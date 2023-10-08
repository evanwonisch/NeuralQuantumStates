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
        self.grad_logpsi = jax.vmap(jax.grad(self.calc_logpsi, argnums = 0), in_axes = (None, 0))


    @abstractmethod
    def calc_logpsi(self, parameters, x):
        """
        Evaluates the logarithmn of the wavefunction on (possibly batches of) basis vectors.
        """
        pass

    def calc_psi(self, parameters, x):
        """
        Evaluates the wavefunction on batches of basis vectors.
        """
        return jnp.exp(self.calc_logpsi(parameters, x))
    
    def calc_logprob(self, parameters, x):
        """
        Evaluates the log probability of state x.
        """
        return 2 * jnp.real(self.calc_logpsi(parameters, x))
    
    def propose_initials(self, key, parameters, N_chains):
        """
        Proposes N_chains starting values to start Markov chains for sampling.
        """
        return jax.random.uniform(key, (N_chains,) + self.shape)


      