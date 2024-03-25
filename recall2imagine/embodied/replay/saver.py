import concurrent.futures
from multiprocessing.pool import ThreadPool
from collections import defaultdict, deque
from functools import partial as bind
import numpy as np
from tqdm import tqdm
import embodied

from . import chunk as chunklib


class Saver:

  def __init__(self, directory, chunks=1024):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.chunks = chunks
    self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
    self.workers = concurrent.futures.ThreadPoolExecutor(16)
    self.promises = deque()
    self.loading = False

  def add(self, step, worker):
    if self.loading:
      return
    buffer = self.buffers[worker]
    buffer.append(step)
    if buffer.length >= self.chunks:
      self.buffers[worker] = buffer.successor = chunklib.Chunk(self.chunks)
      self.promises.append(self.workers.submit(buffer.save, self.directory))
      for promise in [x for x in self.promises if x.done()]:
        promise.result()
        self.promises.remove(promise)

  def save(self, wait=False):
    for buffer in self.buffers.values():
      if buffer.length:
        self.promises.append(self.workers.submit(buffer.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, capacity, length):
    filenames = chunklib.Chunk.scan(self.directory, capacity, 1)
    if not filenames:
      return
    lazychunks = [chunklib.Chunk.load(f, load_data=False) for f in filenames]
    lazychunks1 = {c.uuid: c for c in lazychunks}
    was = {k: False for k in lazychunks1}
    streamids = {}
    streams = set()
    while any(not x for x in was.values()):
      for chunk in (sorted(lazychunks, key=lambda x: x.time)):
        if not was[chunk.uuid]:
          stream_id = chunk.uuid
          streamids[chunk.uuid] = stream_id
          streams.add(stream_id)
          was[stream_id] = True
          nxt = lazychunks1[chunk.successor] if chunk.successor in lazychunks1 else None
          steps = 0
          while nxt is not None and nxt.uuid in lazychunks1:
            streamids[nxt.uuid] = stream_id
            was[nxt.uuid] = True
            nxt = lazychunks1[nxt.successor] if nxt.successor in lazychunks1 else None
            steps += 1
          print(f'Stream: {stream_id} ({len(streams)}th) - {steps} steps')
    threads = min(len(filenames), 32)
    self.loading = True
    with ThreadPool(threads) as executor:
      for chunk in tqdm(executor.imap_unordered(chunklib.Chunk.load, filenames), total=len(filenames)):
        stream = streamids[chunk.uuid]
        for index in range(chunk.length):
          step = {k: v[index] for k, v in chunk.data.items()}
          yield step, stream
        del chunk
    self.loading = False
