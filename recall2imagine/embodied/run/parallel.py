import sys
import time
import logging
import threading
from collections import defaultdict

import embodied
import numpy as np


def parallel(agent, replay, logger, make_env, num_envs, args):
  step = logger.step
  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])
  workers = []
  workers.append(embodied.distr.Thread(
      actor, step, agent, replay, logger, args.actor_addr, args))
  workers.append(embodied.distr.Thread(
      learner, step, agent, replay, logger, timer, args))
  if num_envs == 1:
    workers.append(embodied.distr.Thread(
        env, make_env, args.actor_addr, 0, args, timer))
  else:
    for i in range(num_envs):
      workers.append(embodied.distr.Process(
          env, make_env, args.actor_addr, i, args))
  embodied.distr.run(workers)


def actor(step, agent, replay, logger, actor_addr, args):
  metrics = embodied.Metrics()
  scalars = defaultdict(lambda: defaultdict(list))
  videos = defaultdict(lambda: defaultdict(list))
  should_log = embodied.when.Clock(args.log_every)

  _, initial = agent.policy(dummy_data(
      agent.agent.obs_space, (args.actor_batch,)))
  initial = embodied.treemap(lambda x: x[0], initial)
  allstates = defaultdict(lambda: initial)
  agent.sync()
  # step.t = 0

  def callback(obs, env_addrs):
    states = [allstates[a] for a in env_addrs]
    states = embodied.treemap(lambda *xs: list(xs), *states)
    act, states = agent.policy(obs, states)
    act['reset'] = obs['is_last'].copy()
    for i, a in enumerate(env_addrs):
      allstates[a] = embodied.treemap(lambda x: x[i], states)

    trans = {**obs, **act}
    for i, a in enumerate(env_addrs):
      tran = {k: v[i].copy() for k, v in trans.items()}
      replay.add(tran.copy(), worker=a)
      [scalars[a][k].append(v) for k, v in tran.items() if v.size == 1]
      [videos[a][k].append(tran[k]) for k in args.log_keys_video if k in tran]
    step.increment(args.actor_batch)
    # print(f'fps: {args.actor_batch / (time.time() - step.t):.3f}')
    # step.t = time.time()



    for i, a in enumerate(env_addrs):
      if not trans['is_last'][i]:
        continue
      vids = videos.pop(a) if a in videos else {}
      ep = {**scalars.pop(a), **vids}
      ep = {k: embodied.convert(v) for k, v in ep.items()}
      logger.add({
          'length': len(ep['reward']) - 1,
          'score': sum(ep['reward']),
      }, prefix='episode')
      stats = {}
      for key in args.log_keys_video:
        if key != 'none':
          stats[f'policy_{key}'] = ep[key]
      metrics.add(stats, prefix='stats')

    if should_log():
      logger.add(metrics.result())

    return act

  print('[actor] Start server')
  embodied.BatchServer(actor_addr, args.actor_batch, callback).run()


def learner(step, agent, replay, logger, timer, args):
  logdir = embodied.Path(args.logdir)
  ckpt_dir = embodied.Path(args.checkpoint_dir)
  metrics = embodied.Metrics()
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  updates = embodied.Counter()

  checkpoint = embodied.Checkpoint(ckpt_dir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  is_set = False
  while not is_set:
    with replay.manager.event_lock:
      is_set = bool(replay.manager.init_event.is_set())
    if not is_set:
      print('Learner is waiting for the actor to initialize the buffer..')
      time.sleep(30)
    
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save(filename=replay.manager.tmp_file_path)
  # a workaround for the logger step issue when restoring 
  logger.step.value = embodied.Counter(initial=agent.step.value)

  dataset = agent.dataset(replay)
  state = None
  stats = dict(last_time=time.time(), last_step=int(step), batch_entries=0)
  while True:
    batch = next(dataset)
    outs, state, mets = agent.train(batch, state)
    metrics.add(mets)
    updates.increment()
    stats['batch_entries'] += batch['is_first'].size

    if should_sync(updates):
      agent.sync()

    if should_log():
      train = metrics.result()
      report = agent.report(batch)
      report = {k: v for k, v in report.items() if 'train/' + k not in train}
      logger.add(train, prefix='train')
      logger.add(report, prefix='report')
      logger.add(timer.stats(), prefix='timer')
      logger.add(replay.stats, prefix='replay')

      duration = time.time() - stats['last_time']
      actor_fps = (int(step) - stats['last_step']) / duration
      learner_fps = stats['batch_entries'] / duration
      logger.add({
          'actor_fps': actor_fps,
          'learner_fps': learner_fps,
          'train_ratio': learner_fps / actor_fps if actor_fps else np.inf,
      }, prefix='parallel')
      stats = dict(last_time=time.time(), last_step=int(step), batch_entries=0)
      try:
          logger.write(fps=True)
      except:
          print('logging failed')

    if should_save():
      try:
        checkpoint.save()
      except:
        print('saving failed')
        pass


def env(make_env, actor_addr, i, args, timer=None):
  # TODO: Optionally write NPZ episodes.
  print(f'[env{i}] Make env')
  env = make_env()
  if timer:
    timer.wrap('env', env, ['step'])
  actor = embodied.Client(actor_addr)
  act = {k: v.sample() for k, v in env.act_space.items()}
  done = False
  while True:
    act['reset'] = done
    obs = env.step(act)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    done = obs['is_last']
    promise = actor(obs)
    try:
      act = promise()
    except RuntimeError:
      sys.exit(0)
    act = {k: v for k, v in act.items() if not k.startswith('log_')}


def dummy_data(spaces, batch_dims):
  # TODO: Get rid of this function by adding initial_policy_state() and
  # initial_train_state() to the agent API.
  spaces = list(spaces.items())
  data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
  for dim in reversed(batch_dims):
    data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
  return data
