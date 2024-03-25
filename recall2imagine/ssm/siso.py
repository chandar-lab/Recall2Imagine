from functools import partial

from flax import linen as nn
import jax
import jax.numpy as np
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.nn.initializers import normal

from .common import log_step_initializer, \
    slow_scan, \
    fast_scan, \
    depthwise, \
    SequenceBlock, \
    batchwise


def make_HiPPO(N):
    """
    A standard HiPPO initialization.
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    """
    DLPR matrix discretization function.
    """
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()

def make_NPLR_HiPPO(N):
    """
    Creates a HiPPO matrix and discretizes it.
    """
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)


    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N):
    """
    Diagonalize NPLR representation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V

def init(x):
    """
    Factory for constant initializer in Flax
    """
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init

def hippo_initializer(N):
    """
    The initializer function for the DPLR matrices.
    """
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


@depthwise
class SISOLayer(nn.Module):
    """ 
    The SISO SSM (S4).
    """
    N: int
    l_max: int # apparently in rec mode this has no influence on anything
    parallel: bool = False
    conv: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        """
        initializes the SSM parameters.
        """
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))
        # Flax trick to cache discrete form during decoding.
        def init_discrete():
            return discrete_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )
        self.ssm = init_discrete()

    def __call__(self, u, x0, dones=None):
        """
        Forward pass for SSM.

        when parallel=True:
        Shape:
            u.shape: (len, dims)
            u.dtype: float32
            x_0.shape: (N, dims)
            x_0.dtyle: complex64
        Output Shape:
            y.shape: (len, dims)
            y.dtype: float32
            x.shape: (N, dims)
            x.dtype: complex64
        """
        if not self.parallel or u.shape[0] == 1:
            x, y = slow_scan(*self.ssm, u, x0)
            Du = jax.vmap(lambda u: self.D * u)(u)
            return (y.real + Du).reshape(-1), x
        y, x = fast_scan(*self.ssm, u, x0, dones) # .real is happening inside of fast_scan
        Du = jax.vmap(lambda u: self.D * u)(u)
        return (y + Du)[..., 0], x


SISOBlock = partial(SequenceBlock, layer_cls=SISOLayer)
SISOBlock = batchwise(SISOBlock) 
SISOLayerB = batchwise(SISOLayer)