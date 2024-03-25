from flax import linen as nn
import jax
import jax.numpy as np
from jax.scipy.signal import convolve
from functools import partial

sg = jax.lax.stop_gradient

def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """
    The initializer function for the discretization step parameter.
    """
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init

# Parallel scan operations
@jax.vmap
def vector_operator_with_dones_initial(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.
    
    On reset, drops previous hidden and uses the initialization vector.

    A is diagonal transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z_j, d_j = q_j
    return A_j * A_i * (1 - d_j) + A_j * d_j, \
           A_j * ((1 - d_j) * b_i + d_j * z_j) + b_j, \
           z_j, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def vector_operator_with_dones_add_initial(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, adds the initialization vector to the previous hidden.
    
    A is diagonal transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z_j, d_j = q_j
    return A_j * A_i * (1 - d_j) + A_j * d_j, \
           A_j * (b_i + d_j * z_j) + b_j, \
           z_j, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def vector_operator_with_dones_sg(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, stops gradients of the previous hidden.
    
    A is diagonal transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z, d_j = q_j
    return A_j * A_i * (1 - d_j) + A_j * d_j, \
           A_j * ((1 - d_j) * b_i + d_j * sg(b_i)) + b_j, \
           z, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def vector_operator_with_dones_antisg(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, drops the value of the previous hidden but keeps the gradient.
    
    A is diagonal transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z, d_j = q_j
    return A_j * A_i * (1 - d_j) + A_j * d_j, \
           A_j * ((1 - d_j) * b_i + d_j * (b_i - sg(b_i))) + b_j, \
           z, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def vector_operator_with_dones_antisg_init(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, drops the value of the previous hidden but keeps the gradient.
    
    A is diagonal transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z, d_j = q_j
    return A_j * A_i * (1 - d_j) + A_j * d_j, \
           A_j * ((1 - d_j) * b_i + d_j * (b_i - sg(b_i) + z)) + b_j, \
           z, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def vector_operator_with_dones_none(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.
    
    On reset, keeps the hidden as is. Therefore, this is equivalent
    to not doing reset at all. This function is implemented just
    to keep the same API with resettable function.

    A is diagonal transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z_j, d_j = q_j
    return A_j * A_i, \
           A_j * b_i + b_j, \
           z_j, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def vector_operator_without_dones(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a diagonal matrix A.
    
    Performs an SSM inference step.
    
    A is diagonal transition matrix
    Bu are inputs projected with B
    Args:
        q_i: tuple containing A_i, Bu_i at position i (P,), (P,)
        q_j: tuple containing A_j, Bu_j at position j (P,), (P,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

@jax.vmap
def matrix_operator_with_dones_initial(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a full matrix A (e.g. with DPLR structure).
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, drops previous hidden and uses the initialization vector.
    
    A is full transition matrix
    Bu are inputs projected with B
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, and d_i at position i (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, and d_j at position j (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z_j, d_j = q_j
    return A_j @ A_i * (1 - d_j) + A_j * d_j, \
           A_j @ ((1 - d_j) * b_i + d_j * z_j) + b_j, \
           z_j, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def matrix_operator_with_dones_add_initial(q_i, q_j):
    """ 
    Binary operator for parallel scan of linear recurrence. 
    Assumes a full matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, adds the initialization vector to the previous hidden.
    
    A is full transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z_j, d_j = q_j
    return A_j @ A_i * (1 - d_j) + A_j * d_j, \
           A_j @ (b_i + d_j * z_j) + b_j, \
           z_j, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def matrix_operator_with_dones_sg(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. 
    Assumes a full matrix A.
    
    Performs a resettable SSM inference step with an initialization 
    provided by init vector.

    On reset, stops gradients of the previous hidden.
    
    A is full transition matrix
    Bu are inputs projected with B
    z is initial vector
    d is done flag
    Args:
        q_i: tuple containing A_i, Bu_i, z_i, and d_i at position i (P,), (P,), (P,), (1,)
        q_j: tuple containing A_j, Bu_j, z_j, and d_j at position j (P,), (P,), (P,), (1,)
    Returns:
        new element ( A_out, Bu_out, _, _ )
    """
    A_i, b_i, _, d_i = q_i
    A_j, b_j, z_j, d_j = q_j
    return A_j @ A_i * (1 - d_j) + A_j * d_j, \
           A_j @ ((1 - d_j) * b_i + d_j * sg(b_i)) + b_j, \
           z_j, \
           (d_i + d_j).clip(0, 1)

@jax.vmap
def matrix_operator_without_dones(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. 
    Assumes a full matrix A.
    
    Performs an SSM inference step.
    
    A is full transition matrix
    Bu are inputs projected with B
    Args:
        q_i: tuple containing A_i, Bu_i at position i (P,), (P,)
        q_j: tuple containing A_j, Bu_j at position j (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j

def fast_scan(Ab, Bb, Cb, u, x0, init, dones=None, conj_sym=False, mode='init'):
    """ 
    Compute the LxH output of discretized SSM given an LxH input.
    Uses parallel scan.
    
    Args:
        Ab (complex64): discretized diagonal state matrix        (P,) or (P,H)
        Bb (complex64): discretized input projection matrix      (P, H)
        Cb (complex64): output projection matrix                 (H, P)
        u (float32): input sequence of features                  (L, H)
        x0 (complex64): initial hidden state                     (P,)
        init (complex64): a vector to use after hidden reset     (P,)
        dones (float32): binary flag indicating episode boundary (L,)
        conj_sym (bool): whether conjugate symmetry is enforced
        mode: (str) state reset strategy. Available options are:
            * 'init' - drop hidden state, use a initial vector
            * 'stop_grad' - stop gradient on reset
            * 'add_init' - keep the hidden on reset but add the inital vector
            * 'anti_sg' - zero the hidden on reset but keep its gradient
            * 'anti_sg_init' - like prev. but uses the initial vector
    Returns:
        ys (float32): the SSM outputs                            (L, H)
    """
    if len(Ab.shape) == 1:
        if mode == 'stop_grad':
            operator_with_dones = vector_operator_with_dones_sg
        elif mode == 'none':
            operator_with_dones = vector_operator_with_dones_none
        elif mode == 'init':
            operator_with_dones = vector_operator_with_dones_initial
        elif mode == 'add_init':
            operator_with_dones = vector_operator_with_dones_add_initial 
        elif mode == 'anti_sg':
            operator_with_dones = vector_operator_with_dones_antisg
        elif mode == 'anti_sg_init':
            operator_with_dones = vector_operator_with_dones_antisg_init
        operator_without_dones = vector_operator_without_dones
        A = Ab * np.ones((u.shape[0], Ab.shape[0]))
        A = np.concatenate((np.ones((1, Ab.shape[0]), dtype=A.dtype), A), axis=0)
        B = jax.vmap(lambda u_: Bb @ u_)(u)
    else:
        if mode == 'stop_grad':
            operator_with_dones = matrix_operator_with_dones_sg
        else:
            operator_with_dones = matrix_operator_with_dones_initial
        operator_without_dones = matrix_operator_without_dones
        A = np.ones((u.shape[0] + 1, Ab.shape[0], Ab.shape[1]))
        Abs = np.tile(Ab[None], (u.shape[0], 1, 1))
        A = np.concatenate((np.eye(Ab.shape[0])[None], Abs), axis=0)

        B = jax.vmap(lambda u_: Bb @ u_)(u[..., None])
    B = np.concatenate((x0[..., np.newaxis, :], B), axis=0)

    if dones is None:
        _, xs = jax.lax.associative_scan(operator_without_dones, (A, B))
    else:
        dones = np.insert(dones, -1, 0)
        zeros = np.repeat(np.expand_dims(init, 0), B.shape[0], 0)
        _, xs, zs, ds = jax.lax.associative_scan(operator_with_dones, (A, B, zeros, dones))

    if conj_sym:
        return jax.vmap(lambda x: 2*(Cb @ x).real)(xs[1:]), xs[1:]
    else:
        return jax.vmap(lambda x: (Cb @ x).real)(xs[1:]), xs[1:]

def slow_scan(Ab, Bb, Cb, u, x0, conj_sym=False):
    """
    Compute the LxH output of discretized SSM given an LxH input.
    Uses sequential scan.
    """
    if len(Ab.shape) == 1: # diagonal case
        if u.shape[0] == 1 or len(u.shape) == 1:
            return scan_diag_SSM_one_step(Ab, Bb, Cb, u, x0, conj_sym)
        def step_diag(x_k_1, u_k):
            x_k = Ab * x_k_1 + Bb @ u_k
            if conj_sym:
                y_k = 2 * (Cb @ x_k)
            else:
                y_k = Cb @ x_k
            return x_k, y_k
        return jax.lax.scan(step_diag, x0, u)
    if u.shape[0] == 1:
        return scan_SSM_one_step(Ab, Bb, Cb, u, x0, conj_sym)
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        if conj_sym:
            y_k = 2 * (Cb @ x_k)
        else:
            y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)

def scan_diag_SSM_one_step(Ab, Bb, Cb, u, x0, conj_sym=False):
    """
    One step inference of discretized SSM.
    Assumes SSM with diagonal matrix A.
    """
    x_k = Ab * x0 + Bb @ u
    y_k = Cb @ x_k
    if conj_sym:
        y_k = 2 * y_k
    return x_k, y_k

def scan_SSM_one_step(Ab, Bb, Cb, u, x0, conj_sym=False):
    """
    One step inference of discretized SSM.
    Assumes SSM with full matrix A.
    """
    x_k = Ab @ x0 + Bb @ u
    y_k = Cb @ x_k
    if conj_sym:
        y_k = 2 * y_k
    return x_k, y_k

def depthwise(layer):
    """
    the decorator that adds an additional depth dimension to a layer.
    Useful for e.g. SISO SSM.
    """
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


class SingleLayer(nn.Module):
    """
    The base SSM layer class.
    """

    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    use_norm: bool = True
    prenorm: bool = True
    mlp: bool = True
    glu: bool = True
    training: bool = True

    def setup(self):
        """
        Initializes the layer parameters
        """
        self.seq = self.layer_cls(**self.layer)
        if self.use_norm:
            self.norm = nn.LayerNorm()
        if self.mlp:
            self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        if self.dropout != 0.0:
            self.drop = nn.Dropout(
                self.dropout,
                broadcast_dims=[0],
                deterministic=not self.training,
            )

    def __call__(self, x, state, init=None):
        """
        Forward pass for the layer
        """
        if isinstance(x, tuple):
            x, dones = x
        else:
            dones = None
        skip = x
        if self.use_norm and self.prenorm:
            x = self.norm(x)
        x, outstate = self.seq(x, state, init, dones)
        if self.dropout != 0:
            x = self.drop(nn.gelu(x))
        else:
            x = nn.gelu(x)
        if self.mlp:
            if self.glu:
                x = self.out(x) * jax.nn.sigmoid(self.out2(x))
            else:
                x = self.out(x)
        if self.dropout != 0.0:
            x = skip + self.drop(x)
        else:
            x = skip + x
        if self.use_norm and not self.prenorm:
            x = self.norm(x)
        return (x, dones), outstate
    
class SequenceBlock(nn.Module):
    """
    A class implementing the multi-layer version of the sequence model.
    """
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    n_layers: int = 1
    use_norm: bool = True
    prenorm: bool = True
    mlp: bool = True
    glu: bool = True
    training: bool = True

    def setup(self):
        """
        Initializes the block layer submodules and parameters
        """
        self.layers = [SingleLayer(self.layer_cls, self.layer, self.dropout, self.d_model, self.use_norm, self.prenorm, self.mlp, self.glu, self.training)
                       for _ in range(self.n_layers)]

    def __call__(self, x, state, zero_state=None):
        """
        Foreard pass for the block
        """
        outstate = []
        if zero_state is not None:
            for hidden, init, layer in zip(state, zero_state, self.layers):
                x, h = layer(x, hidden, init)
                outstate.append(h)
        else:
            for hidden, layer in zip(state, self.layers):
                x, h = layer(x, hidden)
                outstate.append(h)
        outstate = np.stack(outstate, axis=-2) # (batch, len, layers, dim) or (batch * len, layers, dim)
        return x, outstate
    
def batchwise(layer): 
    """
    the decorator that adds an additional batch dimension to a layer.
    """
    return nn.vmap(
        layer,
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
        split_rngs={"params": False, "dropout": True},
    )
