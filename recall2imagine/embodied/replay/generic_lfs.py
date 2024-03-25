import time
import pickle
import bz2
from collections import defaultdict, deque
from functools import partial as bind
import pickle

import embodied
import bz2
import numpy as np
import io

from . import saver
from . import selectors, limiters
from .lfs_manager import LFSManager
from . import selectors
from .chunk import Chunk, ChunkSerializer

class FIFO_LFS:
  """
  This class represents a standard FIFO replay buffer where the data is stored on disk.
  After the first chuck of experience arrives, the buffer obtains the data format
  and initializes the replay buffer file. This file contains chunks 
  The main idea behind this implementation is to make the training 
  continuable after interruptions, since we not only need a saved model but also
  the replay buffer.
  
  Therefore, the following (optional) trick is implemented. There are two versions of the buffer. 
  The first is stored in some fast and easily accessible storage (e.g. SSD disk).
  The second is stored in some potentially slower but more persistent storage.
  These versions of the buffer are synced in amortized O(1) time via copy-on-write mechanism.
  If you do not need this dual buffer system, turn it off via `use_lfs=False`.

  For implementation details, see the code comments below.

  Typical buffer size for image envs: 134GB for the 10M buffer; 
  for vector envs: 8GB for the 10M buffer.
  """
  def __init__(
      self, directory, length, capacity, #sampler, limiter,
      overlap=None, online=False, lfs_directory=None, 
      lfs_kwargs=None, samples_per_insert=None,
      use_lfs=False, unlocked_sampling=False,
      tolerance=1e4, min_size=1, batch_size=1,
      num_buffers=2, seed=0):
    assert capacity is None or 1 <= capacity
    if lfs_kwargs is None:
      lfs_kwargs = {}
    # we have to be VERY careful with batch size and
    # num buffers here. if they get corrupted 
    # and diverge from actual batch size and the 
    # bufferization factor, the data agent loads
    # will be a garbage without any notice 
    self.manager = LFSManager(
      tmp_path=directory, lfs_path=lfs_directory,
      readers=batch_size, num_buffers=num_buffers,
      replay_buffer_size=capacity,
      use_lfs=use_lfs, **lfs_kwargs,
      saver_method=self.save, loader_method=self.load
    )
    self.serializer = None
    self.batch_buffer = None
    self.num_buffers = num_buffers
    self.length = length
    self.capacity = capacity
    self.chunks = length
    self.remover = selectors.Fifo()
    self.sampler = selectors.Uniform(seed)
    if samples_per_insert:
      self.limiter = limiters.SamplesPerInsert(
        samples_per_insert, tolerance, min_size, unlocked_sampling)
    else:
      self.limiter = limiters.MinSize(min_size)
    self.stride = 1 if overlap is None else length - overlap
    self.chunk_buffers = defaultdict(bind(Chunk, self.chunks))
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.counters = defaultdict(int)
    self.rng = np.random.default_rng(seed)
    self.was = defaultdict(bool)
    self.table = {}
    self.serializer_pattern = None
    self.bwd_links = {}
    self.fwd_links = {}
    self.inv_table = {}
    self.online = online
    if self.online:
      self.online_queue = deque()
      self.online_stride = length
      self.online_counters = defaultdict(int)
    self.metrics = {
        'samples': 0,
        'sample_wait_dur': 0,
        'sample_wait_count': 0,
        'inserts': 0,
        'insert_wait_dur': 0,
        'insert_wait_count': 0,
    }


  def set_agent(self, agent):
    self._agent = agent

  def __len__(self):
    return len(self.table) * self.length
  
  @property
  def initialized(self):
    return self.manager.initialized

  @property
  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    stats = {
        'size': len(self),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'insert_wait_avg': ratio(m['insert_wait_dur'], m['inserts']),
        'insert_wait_frac': ratio(m['insert_wait_count'], m['inserts']),
        'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
        'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  def add(self, step, worker=0, load=False):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    self.chunk_buffers[worker].append(step)
    self.counters[worker] += 1
    if self.serializer is None:
      self.serializer = ChunkSerializer(self.chunk_buffers[worker])
      self.manager.serializer = self.serializer
      self.manager.initialize(environment_step=step, length=self.length)
    if self.was[worker] >= 2:
      if load:
        assert self.limiter.want_load()[0]
      else:
        dur = wait(self.limiter.want_insert, 'Replay insert is waiting')
        self.metrics['inserts'] += 1                             
        self.metrics['insert_wait_dur'] += dur
        self.metrics['insert_wait_count'] += int(dur > 0)
    if self.counters[worker] < self.length:
      return
    # to think about.
    # when the buffer is restrored from the saved state
    # we get new worker ids here (in the self.was object). Without much pondering, 
    # it seems to me that this should not have any negative effects 
    # on the result but it better to keep that in mind
    self.was[worker] += 1
    self.counters[worker] = 0
    old_buffer = self.chunk_buffers[worker]
    self.chunk_buffers[worker] = Chunk(self.chunks)
    old_buffer.successor = self.chunk_buffers[worker].uuid_b
    _, offset = self.manager.write_chunk(old_buffer)
    if offset in self.inv_table:
      # we are overriding the table entry
      try: # handling race condition
        key = self.inv_table[offset]
        del self.table[key]
        del self.sampler[key]
      except KeyError:
        pass
      if key in self.bwd_links:
        try: # handling race condition
          del self.bwd_links[key]
        except KeyError:
          pass
      else:
        try: # handling race condition
          nxt = self.fwd_links[key]
          del self.bwd_links[nxt]
        except KeyError:
            pass
        try: # handling race condition
          del self.fwd_links[key]
        except KeyError:
            pass
    self.inv_table[offset] = old_buffer.uuid_b
    self.table[old_buffer.uuid_b] = offset
    self.bwd_links[old_buffer.successor] = old_buffer.uuid_b
    self.fwd_links[old_buffer.uuid_b] = old_buffer.successor 
    self.sampler[old_buffer.uuid_b] = offset

  @property
  def ready(self):
    return self.serializer is not None

  def set_agent(self, agent):
    self._agent = agent

  def serialize(self, ):
    data = {
      'fwd_links': self.fwd_links,
      'bwd_links': self.bwd_links,
      'table': self.table,
      'inv_table': self.inv_table,
      'sampler': (self.sampler.indices, self.sampler.keys),
      'was': dict(self.was),
      'offset': self.manager.offset,
      'layers': self.manager.overwrite_layers,
    }
    if self.serializer is not None:
      data['serializer'] = self.serializer.pattern
    else:
      data['serializer'] = self.serializer_pattern
    data = pickle.dumps(data)
    return np.frombuffer(bz2.compress(data), dtype=np.uint8)
  
  def deserialize(self, data):
    data = pickle.loads(bz2.decompress(data.tobytes()))
    self.serializer_pattern = data['serializer']
    if self.manager.serializer is None:
      serializer = ChunkSerializer(pattern=self.serializer_pattern, pattern_obj=None)
      env_step = {k: v[0] for k, v in serializer.dummy_chunk().items()}
      self.serializer = serializer
      self.manager.serializer = serializer
      self.manager.initialize(environment_step=env_step, length=self.length)
    self.fwd_links = data['fwd_links']
    self.bwd_links = data['bwd_links']
    self.table = data['table']
    self.inv_table = data['inv_table']
    self.was = defaultdict(bool)
    self.was.update(data['was'])
    self.sampler.indices, self.sampler.keys = data['sampler']
    self.manager.offset = data['offset']
    self.manager.lfs_offset = data['offset']
    self.manager.overwrite_layers = data['layers']

  def sample(self, flip: int, worker_id: int, batch_buffer):
    dur = wait(self.limiter.want_sample, 'Replay sample is waiting')
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur > 0)
    trying = True
    while trying:
      trying = False
      key = self.sampler()
      while key not in self.bwd_links or self.bwd_links[key] not in self.table:
        key = self.sampler()
      try: # handling race condition
        offset = self.table[key]
      except KeyError:
        trying = True
        continue
      _, chunk = self.manager.read_chunk(flip, offset, worker_id, 0)
      end_pos = self.rng.integers(1, self.length + 1).item()
      if end_pos < self.length:
        try:
          prev_key = self.bwd_links[key]
          prev_offset = self.table[prev_key]
        except KeyError:
          # this is an extremely rare race condition 
          # it happens about once in 300M environment steps
          # if after getting the key from the table,
          # that key is deleted before entering this try 
          # block, this except will be triggered.
          trying = True
          continue
        _, prev_chunk = self.manager.read_chunk(flip, prev_offset, worker_id, 1)
        chunk = {k: np.concatenate([prev_chunk.data[k][end_pos:], chunk.data[k][:end_pos]]) 
                for k in chunk.data.keys()}
        if not (key in self.bwd_links and prev_key in self.table):
          # this is another extremely rare race condition 
          # this checks in the data remains integral
          trying = True
      else:
        chunk = chunk.data
    if 'is_first' in chunk:
      chunk['is_first'][0] = True
    for k in chunk.keys():
      batch_buffer[k][flip, worker_id] = chunk[k]
    return True

  def _remove(self, key):
    wait(self.limiter.want_remove, 'Replay remove is waiting')
    del self.table[key]
    del self.remover[key]
    del self.sampler[key]

  def dataset(self, flip: int, worker_id: int, batch_buffer):
    while True:
      yield self._sample(flip, worker_id, batch_buffer)

  def prioritize(self, ids, prios):
    if hasattr(self.sampler, 'prioritize'):
      self.sampler.prioritize(ids, prios)

  def save(self, wait=True, lfs_only=False):
    table_bytes = self.serialize()
    assert self._agent is not None, 'Please call .set_agent(agent)!!!'
    with io.BytesIO() as stream:
      # the only requirement for the agent saving api 
      # is that it should be able to output a dict of numpy arrays 
      # with parameters
      np.savez(stream, self._agent.save())
      stream.seek(0)
      agent_bytes = np.frombuffer(stream.read(), dtype=np.uint8)
    # we save the weight of the agent and the replay buffer table 
    # into the training state file.
    self.manager.write_prefix(agent_bytes, table_bytes, lfs_only=lfs_only)
    # to keep the training state file synchronized with the 
    # actual training state, we need to flush accumulated 
    # chunks in the long-term storage
    # note that the line below has an effect only if
    # we use a long-term storage
    if not lfs_only: 
      # to avoid infinite recursion the lfs_only flag is introduced here.
      # this is because .maybe_flush triggers another call of this save method,
      # which is needed if .maybe_flush is invoked by data worker 
      # (when we collected enough chunks and we need to store them in lfs;
      #  so to keep the buffer consistent we save the replay buffer table 
      #  right after flushing the latest).
      # in that case 
      self.manager.maybe_flush(force=True, trigger_saving=False)
    return table_bytes

  def maybe_restore(self):
    self.load()

  def load(self, data=None):
    ret = self.manager.read_prefix()
    if len(ret) != 2:
      return False
    agent_bytes, table_bytes = ret
    with io.BytesIO() as stream:
      stream.write(agent_bytes)
      stream.seek(0)
      agent_weights = {k:v for k,v in np.load(stream, allow_pickle=True).items()}['arr_0'].item()
    self._agent.load(agent_weights)
    print('agent loaded!')
    self.deserialize(table_bytes)
    print(f'replay deserialized! The current offset is {self.manager.offset}')
    new_step = (self.manager.offset 
      + self.manager.overwrite_layers 
      * (self.manager.total_chunks - self.manager.prefix_size_stripes)
    ) * self.length
    self._agent.step.load(new_step)
    print(f'Continuing from step {new_step}')

def wait(predicate, message, sleep=0.001, notify=1.0):
  start = time.time()
  notified = False
  while True:
    allowed, detail = predicate()
    duration = time.time() - start
    if allowed:
      return duration
    if not notified and duration >= notify:
      print(f'{message} ({detail})')
      notified = True
    time.sleep(sleep)
