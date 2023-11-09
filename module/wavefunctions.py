from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random

class Wavefunction(ABC):
    """
    General Wrapper for continuous space wave function.
    """

    def __init__(self, input_shape):
        """
        Initialises the class.
        """
        self.input_shape = input_shape
        self.grad_logpsi = jax.vmap(jax.grad(self.calc_logpsi_single, argnums = 0), in_axes = (None, 0))


    @abstractmethod
    def calc_logpsi(self, parameters, x):
        """
        Evaluates the logarithmn of the wavefunction on batches of basis vectors.
        """
        pass
    
    def calc_logpsi_single(self, parameters, x):
        """
        Calculates logpsi on one sample.
        """
        return jnp.squeeze(self.calc_logpsi(parameters, jnp.expand_dims(x, axis = 0)))
    
    def calc_psi(self, parameters, x):
        """
        Calculates psi on batches of data.
        """
        return jnp.exp(self.calc_logpsi(parameters, x))
    
    def calc_psi_single(self, parameters, x):
        """
        Calculates psi on one sample.
        """
        return jnp.squeeze(self.calc_psi(parameters, jnp.expand_dims(x, axis = 0)))


    def calc_logprob(self, parameters, x):
        """
        Evaluates the log probability of state x.
        """
        return 2 * jnp.real(self.calc_logpsi(parameters, x))
    

    def calc_logprob_single(self, parameters, x):
        """
        Calculates logpprob on one sample.
        """
        return jnp.squeeze(self.calc_logprob(parameters, jnp.expand_dims(x, axis = 0)))

    
    def propose_initials(self, key, parameters, N_chains):
        """
        Proposes N_chains starting values to start Markov chains for sampling.
        """
        return jax.random.uniform(key, (N_chains,) + self.shape)


    

######################################################
#               Example Wavefunctions                #
######################################################

class Orbital(Wavefunction):
    def __init__(self, k, R, d_space = 3):
        """
        Initialises a single orbital placed at position R with decay coefficient k. The single parameter A scales the wavefunction.
        """
        super().__init__(input_shape = (d_space,))

        self.R = R
        self.k = k
    
    def calc_logpsi(self, A , x):
        r = jnp.sqrt(jnp.sum((x - self.R)**2, axis = -1))
        return -self.k * r + A
    

class LCAO(Wavefunction):
    def __init__(self, d_space = 3):
        """
        Creates a LCAO orbital in dimensions d.
        """
        super().__init__(input_shape = (d_space,))


    def init_parameters(self, R, k, lamb):
        """
        Creates the parameter pytree which incorporates shape R = (N_nuclei, d) positions of nuclei,
        shape k = (N_nuclei) potential energy coefficients and shape l = (N_nuclei) linear combination coefficients.
        """
        return {"R": R, "k": k, "lamb": lamb}


    def calc_logpsi(self, parameters, x):
        """
        Evaluates the logarithm of psi at positions shape x = (N_samples, d). The parameters incorporate the scaling
        coefficients for the linear combination of orbitals, and thus have shape = (N_nuclei,).
        """
        x = jnp.expand_dims(x, axis = 1)
        R = jnp.expand_dims(parameters["R"], axis = 0)

        d = jnp.sqrt(jnp.sum((x - R)**2, axis = -1))

        return jax.scipy.special.logsumexp(-parameters["k"] * d, b = parameters["lamb"], axis = -1)
