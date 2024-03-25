from . import ninjax as nj
from . import jaxutils

from .ssm.siso import SISOBlock
from .ssm.mimo import MIMOBlock
from .nets import RSSM, Linear

import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

cast = jaxutils.cast_to_compute

def init_siso_ssm(N, d_model, n_layers, parallel=True, conv=False, use_norm=False, **kw):
  """
  The wrapper function that wraps SISO SSM into FlaxModule for compatibility with the main code.
  """
  if 'l_max' not in kw:
    kw['l_max'] = 255

  return nj.FlaxModule(
        SISOBlock, 
        layer={
          'N': N, 
          'l_max': kw['l_max'],
          'parallel': parallel,
          'conv': conv
        },
        name='siso_ssm',
        d_model=d_model, 
        n_layers=n_layers,
        dropout=kw.get('dropout', 0.0), 
        prenorm=kw.get('prenorm', False), 
        mlp=kw.get('mlp', False), 
        glu=kw.get('glu', False), 
        use_norm=use_norm,
      )


def init_mimo_ssm(
    P, 
    H, 
    n_blocks=1,
    n_layers=1,
    conj_sym=False, 
    C_init='', 
    discretization='zoh', 
    dt_min=0.001,
    dt_max=0.1,
    clip_eigs=False,
    parallel=True,
    use_norm=False,
    reset_mode=False,
    **kw
  ):
  """
  The wrapper function that wraps MIMO SSM into FlaxModule for compatibility with the main code.

  kw:
    n_blocks: int
    conj_sym: bool
    C_init: 
  """

  if conj_sym:
    P = P // 2

  return nj.FlaxModule(
        MIMOBlock, 
        layer={
        # **{
          'H': H,
          'P': P,
          'n_blocks': n_blocks,
          'C_init': C_init,
          'discretization': discretization,
          'dt_min': dt_min,
          'dt_max': dt_max,
          'conj_sym': conj_sym,
          'clip_eigs': clip_eigs,
          'parallel': parallel,
          'reset_mode': reset_mode,
        },
        # },
        name='mimo_ssm',
        d_model=H, 
        n_layers=n_layers,
        dropout=kw.get('dropout', 0.0), 
        prenorm=kw.get('prenorm', False), 
        mlp=kw.get('mlp', False), 
        glu=kw.get('glu', False), 
        use_norm=use_norm,
      )


class S3M(RSSM):
  """
  This class implements RSSM with SISO/MIMO SSM cell.
  """
  def __init__(self, deter=1024, stoch=32, classes=32, units=1024, hidden=128, unroll=False, initial='learned',
    unimix=0.01, action_clip=1.0, nonrecurrent_enc=False, ssm='mimo', ssm_kwargs=None, **kw):
    if ssm == 'siso':
      self.core = init_siso_ssm(hidden, units, **ssm_kwargs)
    elif ssm == 'mimo':
      self.core = init_mimo_ssm(hidden, units, **ssm_kwargs)
    else:
      raise NotImplementedError("SSM is not implemented")
    self._ssm = ssm
    self._deter = deter
    self._units = units
    self._hidden = hidden
    self._stoch = stoch
    self._classes = classes
    self._n_layers = ssm_kwargs['n_layers']
    self._parallel = ssm_kwargs['parallel']
    self._conv = ssm_kwargs['conv']
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._nonrecurrent_enc = nonrecurrent_enc
    self._action_clip = action_clip
    self._kw = kw
    self._kw['units'] = units

  def initial(self, bs):
    """
    Returns the initial vector for RSSM. 
    """
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          logit=jnp.zeros([bs, self._stoch, self._classes], f32),
          stoch=jnp.zeros([bs, self._stoch, self._classes], f32))
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          mean=jnp.zeros([bs, self._stoch], f32),
          std=jnp.ones([bs, self._stoch], f32),
          stoch=jnp.zeros([bs, self._stoch], f32))
    if self._ssm == 'siso':
      state['hidden'] = jnp.zeros([bs, self._n_layers, self._hidden, self._units], jnp.complex64)
    if self._ssm == 'mimo':
      state['hidden'] = jnp.zeros([bs, self._n_layers, self._hidden], jnp.complex64)
    if self._initial == 'zeros':
      state = cast(state)
      state['hidden'] = state['hidden'].astype(jnp.complex64)
      return state
    elif self._initial == 'learned':
      hidden = self.get('initial_hidden', jnp.zeros, (2,) + state['hidden'][0].shape, f32)
      deter = self.get('initial_deter', jnp.zeros, state['deter'][0].shape, f32)
      hidden = jnp.expand_dims(hidden[0] + 1j * hidden[1], 0)
      state['hidden'] = jnp.repeat(hidden, bs, 0)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(cast(state['deter']))
      return cast(state)
    else:
      raise NotImplementedError(self._initial)

  def _cell(self, x, prev_state):
    """
    Implements one step forward pass of RSSM.

    x.shape == (batch, units)
    prev_state {
      'hidden' : shape (batch, self._hidden, self._units)
    }
    """
    hidden = prev_state['hidden'] # this is x_t-1
    x = x[:, jnp.newaxis]
    # cell expected shape 
    # u: (batch, 1, self._units)
    # xk: (batch, self._hidden, self._units)
    output, hidden = self.core(x, hidden) # y_t, x_t = S*(u_t-1, x_t-1)
    # y: (batch, 1, self._units)
    # xk1: (batch, self._hidden, self._units)
    if isinstance(output, tuple):
      output, _ = output
    output = output[:, 0]
    if self._ssm == 'siso':
      # (batch, hidden, layers, units) -> (batch, layers, hidden, units)
      hidden = jnp.transpose(hidden, (0, 2, 1, 3))
    deter = output
    kw = {'winit': 'normal', 'fan': 'avg', 'act': self._kw['act'], 'units': self._deter}
    deter = self.get('out_proj', Linear, **kw)(deter)
    # kw = {**self._kw, 'units': self._deter}
    # deter = self.get('deter_proj', Linear, **kw)(output)
    return deter, {'deter': deter, 'hidden': hidden}
  

  def _cell_scan(self, x, state, first, zero_state):
    """
    Implements sequential forward pass for RSSM.
    """
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if not self._parallel:
      state, (outs, x) = super()._cell_scan(x, state, first, zero_state)
      return state, (tree_map(swap, outs), swap(x))
    
    if self._ssm == 'siso':
      first = jnp.tile(first[..., None], (1, 1, x.shape[-1]))
    x, outstate = self.core((swap(x), swap(first)), state['hidden'], zero_state['hidden']) # hidden.shape = (batch, layers, hidden, units)
    if isinstance(x, tuple):
      x, _ = x
    if self._ssm == 'siso':
      # (batch, seq, units, layers, hidden) -> (batch, seq, layers, hidden, units)
      outstate = jnp.transpose(outstate, (0, 1, 3, 4, 2))
    kw = {'winit': 'normal', 'fan': 'avg', 'act': self._kw['act'], 'units': self._deter}
    x = self.get('out_proj', Linear, **kw)(x)
    return {'deter': x[:, -1], 'hidden': outstate[:, -1]}, ({'deter': x, 'hidden': outstate}, x)
