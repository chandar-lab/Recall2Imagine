from functools import partial
import jax
import jax.numpy as np
from jax import random
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh
from flax import linen as nn
from jax.scipy.linalg import block_diag
from .common import SequenceBlock, batchwise, fast_scan, slow_scan

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

# Initializtion Scheme

def make_HiPPO(N):
    """ 
    Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ 
    Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.
    Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
    Returns:
        init function
     """
    def init(key, shape):
        """ 
        Init function
        Args:
            key: jax random key
            shape tuple: desired shape
        Returns:
            sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ 
    Initialize an array of learnable timescale parameters
    Args:
        key: jax random key
        input: tuple containing the array shape H and
            dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    return log_step_initializer(dt_min=dt_min, dt_max=dt_max)(key, shape=(H,1))


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    return lecun_normal()(key, shape=(H, P, 2))


def init_CV(init_fun, rng, shape, V):
    """ 
    Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
    Args:
        init_fun:  the initialization function to use, e.g. lecun_normal()
        rng:       jax random key to be used with init function.
        shape (tuple): desired shape  (H,P)
        V: (complex64)     the eigenvectors used for initialization
    Returns:
        C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ 
    Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ 
    Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


class MIMOLayer(nn.Module):
    P: int
    H: int
    n_blocks: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    parallel: bool = False
    reset_mode: bool = False

    """ 
    The MIMO SSM (S5)
    Args:
        Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
        Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
        V           (complex64): Eigenvectors used for init           (P,P)
        Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
        H           (int32):     Number of features of input seq 
        P           (int32):     state size
        C_init      (string):    Specifies How C is initialized
                        Options: [trunc_standard_normal: sample from truncated standard normal 
                                                    and then multiply by V, i.e. C_tilde=CV.
                                lecun_normal: sample from Lecun_normal and then multiply by V.
                                complex_normal: directly sample a complex valued output matrix 
                                                from standard normal, does not multiply by V]
        conj_sym    (bool):    Whether conjugate symmetry is enforced
        clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                constrain real part of eigenvalues to be negative. 
                                True recommended for autoregressive task/unbounded sequence lengths
                                Discussed in https://arxiv.org/pdf/2206.11893.pdf.
        discretization: (string) Specifies discretization method 
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
        dt_min:      (float32): minimum value to draw timescale values from when 
                                initializing log_step
        dt_max:      (float32): maximum value to draw timescale values from when 
                                initializing log_step
        step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """
        Initializes parameters once and performs discretization each time
        the SSM is applied to a sequence
        """
        block_size = int(self.P // self.n_blocks)

        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size * (1 + self.conj_sym))

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        Lambda = (Lambda * np.ones((self.n_blocks, block_size))).ravel()
        V = block_diag(*([V] * self.n_blocks))
        Vinv = block_diag(*([Vc] * self.n_blocks))
        # just to make sure we don't grad this
        Lambda = sg(Lambda)
        V = sg(V)
        Vinv = sg(Vinv)

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (block_size,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (block_size,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            C = self.param("C", C_init, (self.H, self.P, 2))
            self.C_tilde = C[..., 0] + 1j * C[..., 1]
        else:
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init, rng, shape, V),
                                C_shape)

            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, u, x0, init, dones=None):
        """
        Compute the LxH output of the SSM given an LxH input sequence
        using a parallel scan.
        Args:
            input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """

        if self.parallel and u.shape[0] > 1:
            y, x = fast_scan(
                self.Lambda_bar, self.B_bar, self.C_tilde, 
                u, x0, init, dones, self.conj_sym, self.reset_mode)
            Du = jax.vmap(lambda u: self.D * u)(u)
            return y + Du, x
            
        x, y = slow_scan(self.Lambda_bar, self.B_bar, self.C_tilde, u[0], x0, self.conj_sym)
        Du = jax.vmap(lambda u: self.D * u)(u)
        return y.reshape(-1).real + Du, x


MIMOBlock = partial(SequenceBlock, layer_cls=MIMOLayer)
MIMOBlock = batchwise(MIMOBlock)
MIMOLayerB = batchwise(MIMOLayer)