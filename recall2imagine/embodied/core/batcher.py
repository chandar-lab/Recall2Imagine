import queue as queuelib
import sys
import threading
import time
import traceback

import numpy as np


class Batcher:

  def __init__(
      self, sources, workers=0, postprocess=None,
      prefetch_source=4, prefetch_batch=2):
    self._workers = workers
    self._postprocess = postprocess
    if workers:
      # Round-robin assign sources to workers.
      self._running = True
      self._threads = []
      self._queues = []
      assignments = [([], []) for _ in range(workers)]
      for index, source in enumerate(sources):
        queue = queuelib.Queue(prefetch_source)
        self._queues.append(queue)
        assignments[index % workers][0].append(source)
        assignments[index % workers][1].append(queue)
      for args in assignments:
        creator = threading.Thread(
            target=self._creator, args=args, daemon=True)
        creator.start()
        self._threads.append(creator)
      self._batches = queuelib.Queue(prefetch_batch)
      batcher = threading.Thread(
          target=self._batcher, args=(self._queues, self._batches),
          daemon=True)
      batcher.start()
      self._threads.append(batcher)
    else:
      self._iterators = [source() for source in sources]
    self._once = False

  def close(self):
    if self._workers:
      self._running = False
      for thread in self._threads:
        thread.close()

  def __iter__(self):
    if self._once:
      raise RuntimeError(
          'You can only create one iterator per Batcher object to ensure that '
          'data is consumed in order. Create another Batcher object instead.')
    self._once = True
    return self

  def __call__(self):
    return self.__iter__()

  def __next__(self):
    if self._workers:
      batch = self._batches.get()
    else:
      elems = [next(x) for x in self._iterators]
      batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
    if isinstance(batch, Exception):
      raise batch
    return batch

  def _creator(self, sources, outputs):
    try:
      iterators = [source() for source in sources]
      while self._running:
        waiting = True
        for iterator, queue in zip(iterators, outputs):
          if queue.full():
            continue
          queue.put(next(iterator))
          waiting = False
        if waiting:
          time.sleep(0.001)
    except Exception as e:
      e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      outputs[0].put(e)
      raise

  def _batcher(self, sources, output):
    try:
      while self._running:
        elems = [x.get() for x in sources]
        for elem in elems:
          if isinstance(elem, Exception):
            raise elem
        batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
        if self._postprocess:
          batch = self._postprocess(batch)
        output.put(batch)  # Will wait here if the queue is full.
    except Exception as e:
      e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      output.put(e)
      raise


class BatcherSM:
  """
  A class for the batching process that uses shared numpy memory.
  It is much faster than standard Batcher when we are working with
  long sequences in batches because the former sends a lot of data through
  threading.Queue, which is much slower than reading/writing 
  to the shared numpy arrays. 
  """
  def __init__(
      self, replay, workers=0, batch_size=1, 
      batch_sequence_len=1, postprocess=None,
      prefetch_source=4, prefetch_batch=2):
    self._workers = workers
    self._postprocess = postprocess
    # we assume replay was filled with at least one step so the serializer is 
    # initialized already
    self._replay = replay
    self.prefetch_batch = prefetch_batch
    self.batch_size = batch_size
    self._serializer = self._replay.serializer
    self._batch_buffers = None
    self.batch_sequence_len = batch_sequence_len
    self.batch_size = batch_size
    
    if workers:
      # Round-robin assign sources to workers.
      self._running = True
      self._threads = []
      self._queues = []
      self.tasks = queuelib.Queue(prefetch_batch * batch_size)
      self.reports = queuelib.Queue(prefetch_batch * batch_size)
      for _ in range(workers):
        creator = threading.Thread(
          target=self._creator, args=(), daemon=True
        )
        creator.start()
        self._threads.append(creator)
      self._outputs = queuelib.Queue(prefetch_batch)
      batcher = threading.Thread(
          target=self._batcher, args=(),
          daemon=True)
      batcher.start()
      self._threads.append(batcher)
    self._once = False

  def close(self):
    if self._workers:
      self._running = False
      for thread in self._threads:
        thread.close()

  def __iter__(self):
    if self._once:
      raise RuntimeError(
          'You can only create one iterator per Batcher object to ensure that '
          'data is consumed in order. Create another Batcher object instead.')
    self._once = True
    return self

  def __call__(self):
    return self.__iter__()

  def __next__(self):
    if self._workers:
      batch = self._outputs.get()
    else:
      elems = [next(x) for x in self._iterators]
      batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
    if isinstance(batch, Exception):
      raise batch
    return batch

  def _creator(self):
    try:
      while self._running:
        if not self.tasks.empty() and self._replay.ready:
          if self._batch_buffers is None:
            self._serializer = self._replay.serializer
            self._batch_buffers = self._serializer.batch_buffer(
              self.prefetch_batch, self.batch_size, self.batch_sequence_len)
          flip, task_id = self.tasks.get()
          success = self._replay.sample(flip, task_id, self._batch_buffers)
          assert success
          self.reports.put((flip, task_id))
        else:
          time.sleep(0.001)
    except Exception as e:
      e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      raise

  def read_buffer(self, flip):
    # copy here is necessary as right after this read we'll
    # assign another set of data loading tasks which will 
    # overwrite the batch buffers
    return {k: v[flip].copy() for k, v in self._batch_buffers.items()}

  def _batcher(self, ):
    if self.tasks.empty():
      for flip in range(self.prefetch_batch):
        for batch_id in range(self.batch_size):
          self.tasks.put((flip, batch_id))
    batch_completion = [[False for _ in range(self.batch_size)] 
                        for _ in range(self.prefetch_batch)]
    try:
      while self._running:
        while not self.reports.empty():
          flip, task_id = self.reports.get()
          batch_completion[flip][task_id] = True
        completed = [flip for flip in range(self.prefetch_batch) if all(batch_completion[flip])]
        for flip in completed:
          batch = self.read_buffer(flip)
          for k in range(self.batch_size): 
            batch_completion[flip][k] = False
            self.tasks.put((flip, k))
          if self._postprocess:
            batch = self._postprocess(batch)
          self._outputs.put(batch)
        if len(completed) == 0:
          time.sleep(0.001)
    except Exception as e:
      e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      raise
