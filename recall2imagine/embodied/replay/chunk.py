import io
from datetime import datetime

import embodied
import numpy as np
import uuid as uuidlib


class Chunk:
  """
  This class represents a contiguous chunk of RL experience 
  and implements some useful methods to work with that - 
  append experience step, save and load.
  """
  def __init__(self, size, successor=None):
    now = datetime.now()
    self.time = now.strftime("%Y%m%dT%H%M%S") + f'F{now.microsecond:06d}'
    self.uuid_b = embodied.uuid()
    self.uuid = str(self.uuid_b)
    self.uuid_b = self.uuid_b.value 
    self.successor = successor
    self.size = size
    self.data = None
    self.length = 0

  def __repr__(self):
    succ = self.successor or str(embodied.uuid(0))
    succ = succ.uuid if isinstance(succ, type(self)) else succ
    return (
        f'Chunk(uuid={self.uuid}, '
        f'succ={succ}, '
        f'len={self.length})')

  def __len__(self):
    return self.length

  def __bool__(self):
    return True

  def append(self, step):
    if not self.data:
      example = {k: embodied.convert(v) for k, v in step.items()}
      self.data = {
          k: np.empty((self.size,) + v.shape, v.dtype)
          for k, v in example.items()}
    for key, value in step.items():
      self.data[key][self.length] = value
    self.length += 1

  def save(self, directory):
    succ = self.successor or str(embodied.uuid(0))
    succ = succ.uuid if isinstance(succ, type(self)) else succ
    filename = f'{self.time}-{self.uuid}-{succ}-{self.length}.npz'
    filename = embodied.Path(directory) / filename
    data = {k: embodied.convert(v) for k, v in self.data.items()}
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **data)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    print(f'Saved chunk: {filename.name}')

  @classmethod
  def load(cls, filename, load_data=True):
    length = int(filename.stem.split('-')[3])
    if load_data:
      with embodied.Path(filename).open('rb') as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    else:
      data = None
    chunk = cls(length)
    chunk.time = filename.stem.split('-')[0]
    chunk.uuid = filename.stem.split('-')[1]
    chunk.successor = filename.stem.split('-')[2]
    chunk.length = length
    chunk.data = data
    chunk.filename = filename
    return chunk

  @classmethod
  def scan(cls, directory, capacity=None, shorten=0):
    directory = embodied.Path(directory)
    filenames, total = [], 0
    for filename in reversed(sorted(directory.glob('*.npz'))):
      if capacity and total >= capacity:
        break
      filenames.append(filename)
      total += max(0, int(filename.stem.split('-')[3]) - shorten)
    return sorted(filenames)

class ChunkSerializer:
  """
  This class represents the serialization behaviour for the Chunk
  class definded above. The serialization format is 
  (next chunk uuid, current chunk uuid, experience payload)
  where the latter is several numpy arrays serialized to bytes.
  The next chunk link is needed to keep the sequential order of data in 
  the dataloader (see the buffer class defined in generic_lfs.py). 
  """
  def __init__(self, pattern_obj: Chunk, pattern=None):
    if pattern is None:
      self.pattern = [(k, v.shape, v.dtype) for k, v in pattern_obj.data.items()]
    else:
      self.pattern = pattern
  
  def dummy_chunk(self):
    return {
      k: np.empty(shape=v, dtype=d) for k,v,d in self.pattern
    }

  @property
  def chunk_size(self):
    return sum(np.dtype(dt).itemsize * np.prod(sh) for _, sh, dt in self.pattern) + 16 * 2
  
  def batch_buffer(self, num_buffers, batch_size, sequence_length):
    return {
      k: np.empty((num_buffers, batch_size, sequence_length, *shape[1:]), dtype=dtype)
      for k, shape, dtype in self.pattern
    }

  def serialize(self, chunk: Chunk, buffer: np.ndarray):
    offset = 0

    succ = chunk.successor or str(embodied.uuid(0))
    succ = succ.uuid if isinstance(succ, type(chunk)) else succ
    succ = succ.value if isinstance(succ, embodied.uuid) else bytes(succ)
    buffer[offset:offset+16] = np.frombuffer(succ, dtype=np.uint8)
    offset += 16

    uuid = chunk.uuid_b
    buffer[offset:offset+16] = np.frombuffer(uuid, dtype=np.uint8)
    offset += 16

    for k, shape, dtype in self.pattern:
      assert shape == chunk.data[k].shape
      assert dtype == chunk.data[k].dtype
      array = chunk.data[k]
      buffer[offset:offset+array.nbytes] = array.view(np.uint8).flat
      offset += array.nbytes

  def deserialize(self, buffer):
    offset = 0
    succ = buffer[offset:offset+16].tobytes()
    offset += 16
    uuid = buffer[offset:offset+16].tobytes()
    offset += 16
    destination = {}
    for k, shape, dtype in self.pattern:
      bytes_size = np.dtype(dtype).itemsize * np.prod(shape)
      destination[k] = buffer[offset:offset+bytes_size].view(dtype).reshape(shape).copy()
      offset += bytes_size
    chunk = Chunk(len(destination[k]))
    chunk.data = destination
    chunk.uuid_b = uuid
    chunk.uuid = embodied.uuid(int.from_bytes(uuid, 'big'))
    chunk.successor = succ
    return chunk