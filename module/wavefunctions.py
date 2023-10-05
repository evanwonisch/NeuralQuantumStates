from abc import ABC, abstractmethod
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

    
    def calc_psi(self, x):
        """
        Evaluates the wavefunction at batches of data
        """
        return jnp.exp(self.calc_logpsi(x))