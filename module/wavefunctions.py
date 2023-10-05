from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random

class Wavefunction(ABC):
    """
    General Wrapper for continuous space wave function.
    """

    @abstractmethod
    def calc_logpsi(self, x):
        """
        Evauluates the logarithmn of the wavefunction at batches of data
        """
        pass

    @partial(jax.jit, static_argnames=['self'])
    def calc_psi(self, x):
        """
        Evaluates the wavefunction at batches of data
        """
        return jnp.exp(self.calc_logpsi(x))
    
    @partial(jax.jit, static_argnames=['self'])
    def calc_logprob(self, x):
        """
        Evaluates the log probability of state x
        """
        return 2 * jnp.real(self.calc_logpsi(x))