import functools

import embodied
import gym
import gymnasium as gym_
import numpy as np

def flatten(space, onehot=False):
  if isinstance(space, gym_.spaces.Discrete):
    n = space.n
    p = np.eye(n)
    tr = lambda x: p[x]
    space_ = gym_.spaces.Box(0, 1, (n,))
  elif isinstance(space, gym_.spaces.MultiDiscrete):
    p = [np.eye(t) for t in space.nvec]
    tr = lambda x: np.concatenate([eye[j] for eye, j in zip(p, x)], axis=0)
    size = space.nvec.sum()
    space_ = gym_.spaces.Box(0, 1, (size,))
  elif isinstance(space, gym_.spaces.Tuple):
    p = [np.eye(t.n) for t in space.spaces]
    tr = lambda x: np.concatenate([eye[j] for eye, j in zip(p, x)], axis=0)
    space_ = gym_.spaces.Box(0, 1, (sum([len(pp) for pp in p]),))
  return tr, space_

def unmap_action(a, nvec):
  newa = np.zeros(len(nvec), dtype=np.int32)
  for i in reversed(range(len(nvec))):
    newa[i] = a % nvec[i]
    a //= nvec[i]
  return newa

class MyObsPopgym(gym_.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.tr, space_ = flatten(self.observation_space)
    self.maction = self.action_space if isinstance(self.action_space, gym_.spaces.MultiDiscrete) else None
    if self.maction is not None:
      self.action_space = gym_.spaces.Discrete(np.prod(self.action_space.nvec))
    self.observation_space = gym_.spaces.Dict({'observation': space_})
    
  def step(self, action):
    if self.maction is not None:
      action = unmap_action(action, self.maction.nvec)
    obs, reward, ter, trun, info = super().step(action)
    obs = self.tr(obs)
    return {'observation': obs}, reward, ter | trun, info

  def reset(self):
    obs, info = super().reset()
    return {'observation': self.tr(obs)}

class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      try:
        self._env = gym.make(env, **kwargs)
      except:
        self._env = gym_.make(env, **kwargs)
        if 'popgym' in env:
          self._env = MyObsPopgym(self._env)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'step_no': embodied.Space(np.int32),
        # 'ep_no': embodied.Space(np.int32),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, self._done, self._info = self._env.step(action)
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, (gym.spaces.Dict, gym_.spaces.Dict)):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
