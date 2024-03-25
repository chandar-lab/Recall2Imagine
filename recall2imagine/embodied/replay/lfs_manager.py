import os
import sys
import time
import subprocess
import contextlib
import traceback
import math
import numpy as np
import threading
_200GB = 214_748_364_800 # bytes
_1GB = 1_073_741_824 # bytes
_1MB = 1_048_576
_128KB = _1MB // 8

class LFSManager:
  """
  This class implements the logic behind RL buffer as a large file
  and dual buffers in two storages.  
  """
  def __init__(
      self, tmp_path, lfs_path, 
      replay_buffer_size=1e7,
      stripe_count=8, stripe_size_mb=16,
      readers=16, num_buffers=2, use_lfs=True,
      prefix_size_mb=2048, saver_method=None,
      loader_method=None,
    ):
    """
    Args:
      tmp_path: a path to the fast storage
      lfs_path: a path to the persistent storage
      replay_buffer_size: size of the buffer in terms of RL steps
      stripe_count: an internal parameter for the lustre file system. 
        currently not used, saved for future.
      stripe_size_mb: an internal parameter for the lustre file system. 
        currently not used, saved for future.
      readers: how many data reading threads are there (it's also specified in the batcher object)
      num_buffers: the bufferization factor, to see what's that google "double buffering",
        it's similar to the "prefetch_factor" in pytorch dataloader
      use_lfs: whether to use the big file in large but slow storage
      prefix_size_mb: how big is the space reserved for 
                        * the model, 
                        * replay buffer table (takes a couple of MBs for the buffer of 10M steps)
                        * optimizer state
                      by defaults is 2GB, so works for models of ~100M params, increase if needed
      saver_method: the function that saves the replay buffer, needed internally, do not touch
      loader_method: the function that loads the replay buffer, needed internally, do not touch
    """
    self.serializer = None

    # the current offset in chunks
    self.offset = 0
    self.lfs_offset = 0
    self.overwrite_layers = 0
    self._LFS_ACCUM = 50
    self.replay_buffer_size = replay_buffer_size
    self.readers = readers
    self.num_buffers = num_buffers
    self.prefix_size_mb = prefix_size_mb
    self.prefix_size_b = prefix_size_mb * _1MB
    self.tmp_path = tmp_path
    self.lfs_path = lfs_path
    self.use_lfs = use_lfs
    self.lfs_data = None
    self.last_flush_time = None
    self.lfs_lock = threading.RLock() if use_lfs else contextlib.nullcontext()
    self.init_event = threading.Event()
    self.event_lock = threading.RLock()
    self.saver_method = saver_method
    self.loader_method = loader_method
    os.makedirs(self.tmp_path, exist_ok=True)
    self.tmp_file_path = f'{self.tmp_path}/buffer.blob'
    self.lfs_file_path = f'{self.lfs_path}/buffer.blob'
    if os.path.exists(self.tmp_file_path):
      self.tmp_file = open(self.tmp_file_path, 'r+b')
    self.initialized = False
    self.initializing = False

  def initialize(self, stripe_count=8, environment_step=None,
                 length=1): 
    if not self.initialized:
      with self.lfs_lock:
        if not self.initialized:

          self.initializing = True
          size = sum(v.nbytes for v in environment_step.values())
          size = self.serializer.chunk_size
          UNIT = _128KB if size < _1MB else _1MB
          self.unit = UNIT
          total_buffer_chunks = int(math.ceil(self.replay_buffer_size / length))
          size_units = int(math.ceil(size / UNIT))
          size_bytes = size_units * UNIT
          fifo_size = total_buffer_chunks * size_bytes
          total_buffer_size = fifo_size + self.prefix_size_b
        
          self.stripe_count = stripe_count
          self.stripe_size_units = size_units
          self.stripe_size_b = size_units * UNIT
          self.prefix_size_stripes = self.prefix_size_b // self.stripe_size_b
          self.total_chunks = total_buffer_size // self.stripe_size_b
          prealloc = False
          needs_load = True
          if self.use_lfs:
            os.makedirs(self.lfs_path, exist_ok=True)
            print('checking if LFS file exists!')
            if not os.path.exists(self.lfs_file_path):
              self.lfs_file = open(self.lfs_file_path, 'w+b')
              self.init_stripe()
              prealloc = True
            else:
              print('LFS file exists!')
              self.lfs_file = open(self.lfs_file_path, 'r+b')
              if not os.path.exists(self.tmp_file_path):
                print('TMP file does not exists! copying...')
                subprocess.run(['cp', self.lfs_file_path, self.tmp_file_path])
                # being here means that there is a long term buffer 
                # that we can take. Which means we should deserialize the 
                # address table of the replay buffer which is done by the line below
                needs_load = True
                print('copying done')
          else:
            prealloc = True 
          prealloc_tmp = not os.path.exists(self.tmp_file_path)
          if os.path.exists(self.tmp_file_path):
            self.tmp_file = open(self.tmp_file_path, 'r+b')
          else:
            self.tmp_file = open(self.tmp_file_path, 'w+b')
          if needs_load:
            self.loader_method()
          if prealloc and self.use_lfs:
            print(f'allocating {total_buffer_size / _1GB:.3f}GB replay buffer in long term storage')
            os.posix_fallocate(self.lfs_file.fileno(), 0, total_buffer_size)
          if prealloc_tmp:
            print(f'allocating {total_buffer_size / _1GB:.3f}GB replay buffer in local storage')
            os.posix_fallocate(self.tmp_file.fileno(), 0, total_buffer_size)
          if self.use_lfs:
            os.posix_fadvise(self.lfs_file.fileno(), 0, total_buffer_size, os.POSIX_FADV_SEQUENTIAL)
          os.posix_fadvise(self.tmp_file.fileno(), 0, total_buffer_size, os.POSIX_FADV_RANDOM)

          self.write_buffer = np.zeros(self.stripe_size_b, dtype=np.uint8)
          self.lfs_write_buffer = np.zeros(self.stripe_size_b * self._LFS_ACCUM, dtype=np.uint8)
          self.lfs_write_pointer = 0
          # the shape has the following meaning:
          # num_buffers - bufferization factor: when batch collator reads from i-th buffer; data readers
          #               write to the ((i + 1) % num_buffers)-th one
          # readers - each dataloading worker writes to its own piece of memory. otherwise, race conditions will happen
          # 2 - is because we may need to read left and right havles to cut a chunk of several RL steps from that.
          # stripe_size_b - the size of payload
          self.read_buffers = np.zeros((self.num_buffers, self.readers, 2, self.stripe_size_b), dtype=np.uint8)
          self.initialized = True
          print('initialized!')
          self.initializing = False
      with self.event_lock:
        if not self.init_event.is_set():
          self.init_event.set()

  def init_stripe(self):
    if self.unit != _1MB:
      mbs = int(math.ceil(self.stripe_size_b / _1MB))
    else:
      mbs = self.stripe_size_units
    
  def write_chunk(self, chunk):
    self.serializer.serialize(chunk, self.write_buffer)
    offset = self.offset * self.stripe_size_b + self.prefix_size_b
    written = os.pwrite(self.tmp_file.fileno(), self.write_buffer, offset)
    if self.use_lfs:
      with self.lfs_lock:
        self.lfs_data = self.write_buffer.copy()
    offset = self.offset
    print(f'written {written/_1MB:.2f}MB with offset {offset}')
    self.offset += written // self.stripe_size_b
    self.offset = self.offset \
                  % (self.total_chunks - self.prefix_size_stripes)
    self.maybe_flush()
    self.overwrite_layers += self.offset < offset
    return written, offset
  
  def maybe_flush(self, force=False, trigger_saving=True):
    if self.use_lfs:
      trigger = False
      with self.lfs_lock:
        if self.lfs_data is not None:
          buffer = self.lfs_data
          self.lfs_data = None
          self.lfs_write_buffer[
            self.lfs_write_pointer * self.write_buffer.nbytes : 
            (self.lfs_write_pointer + 1) * self.write_buffer.nbytes] = buffer.copy()
          self.lfs_write_pointer += 1
        if self.lfs_write_pointer == self._LFS_ACCUM or ((self.offset == 0 or force) and self.lfs_write_pointer > 0):
          trigger = True
          offset_ = self.lfs_offset * self.stripe_size_b + self.prefix_size_b
          written_lfs = os.pwrite(
            self.lfs_file.fileno(), 
            self.lfs_write_buffer[
              :self.lfs_write_pointer * self.write_buffer.nbytes], offset_)
          if self.last_flush_time is not None:
            t = time.time()
            delta = f'. last write {t - self.last_flush_time:.3f} seconds ago'
          else:
            delta = ''
          self.last_flush_time = time.time()
          print(f'written {int(math.ceil(written_lfs / _1MB)):.2f}MB to the long-term storage' + delta)
          lfs_stripes = written_lfs // self.stripe_size_b
          self.lfs_offset += lfs_stripes
          self.lfs_offset = self.lfs_offset \
                    % (self.total_chunks - self.prefix_size_stripes)
          self.lfs_write_pointer = 0
          # this is a very important check that ensures the data integrity. if it fails then smth certainly not doing right
          assert self.lfs_offset == self.offset 
      if trigger_saving and trigger: 
        # here we need to guarantee that lfs has the correct state all
        # the time, so we trigger
        self.saver_method(lfs_only=True)

  def read_chunk(self, buffer: int, address: int, worker_id: int, flip: int):
    offset = self.prefix_size_b + address * self.stripe_size_b
    read_bytes = os.preadv(self.tmp_file.fileno(), [self.read_buffers[buffer, worker_id, flip]], offset)
    chunk = self.serializer.deserialize(self.read_buffers[buffer, worker_id, flip])
    return read_bytes, chunk

  def read_prefix(self):
    # try restoring from the lfs
    if self.use_lfs and os.path.exists(self.lfs_file_path) and not os.path.exists(self.tmp_file_path):
      subprocess.run(['cp', self.lfs_file_path, self.tmp_file_path])
      self.tmp_file = open(self.tmp_file_path, 'r+b')
    if not os.path.exists(self.tmp_file_path):
      return ()
    # first we read N
    buf = np.zeros(8, dtype=np.uint8)
    os.preadv(self.tmp_file.fileno(), [buf], 0)
    N = buf.view(np.int64).item()
    if N == 0:
      return () # in that case the prefix has all zero bytes
    # then we read N 64 bit numbers (each telling the size of an object)
    buf = np.zeros(8 * N, dtype=np.uint8)
    os.preadv(self.tmp_file.fileno(), [buf], 8)
    sizes = buf.view(np.int64).copy()
    # finally we read these objects
    buf = np.zeros(sizes.sum().item(), dtype=np.uint8)
    os.preadv(self.tmp_file.fileno(), [buf], 8 * (N + 1))
    accum = 0
    buffers = []
    for size in sizes:
      buffers.append(buf[accum:accum + size].copy())
      accum += size
    return tuple(buffers)

  def write_prefix(self, *buffers, lfs_only=False):
    # the idea is that we receive N buffers with arbitrary data 
    # (but each packed into the flat np array with type np.uint8).
    # we first store N, then N numbers each indicating the size of each buffer
    # and then the buffers themselves. This is a rare operation so we can spend 
    # extra memory on it by concatenating buffers. 
    buffers_size = np.array([len(buffers)] + [len(b) for b in buffers], dtype=np.int64)
    buffers_size = np.frombuffer(buffers_size, dtype=np.uint8)
    buffers = np.concatenate((buffers_size,) + buffers)
    if not lfs_only:
      if not self.initialized:
        # trigerring this means we reached a race condition
        # where actor is about to initialize the replay buffer
        # so we need to wait a little bit.
        print('waiting for the replay to initialize...')
        while self.initializing: 
          time.sleep(60)
        # try acquireing the lock to make sure it initialized
        with self.lfs_lock:
          assert self.initialized
      written_bytes = os.pwrite(self.tmp_file.fileno(), buffers, 0)
      print(f'written {written_bytes/_1MB:.4f}MB to the buffer prefix')
    if self.use_lfs:
      with self.lfs_lock:
        if not hasattr(self, 'lfs_file'):
          # this will be triggered if we did not collect any data yet
          # but are calling the save method so we need to initialize the 
          # lfs file.
          if os.path.exists(self.lfs_file_path):
            # if lfs file is here - all good, we are continuing the experiment
            self.lfs_file = open(self.lfs_file_path, 'r+b')
        lfs_written_bytes = os.pwrite(self.lfs_file.fileno(), buffers, 0)
        if self.last_flush_time is not None:
          t = time.time()
          delta = f'. last write {t - self.last_flush_time:.3f} seconds ago'
        else:
          delta = ''
        self.last_flush_time = time.time()
        print(f'written {lfs_written_bytes/_1MB:.4f}MB to the long-term buffer prefix' + delta)
        # otherwise, that means we started from scratch so we do nothing
