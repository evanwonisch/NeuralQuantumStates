from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random

class Hamiltonian(ABC):
    """
    Implements the local Hamilton Operator operating on a wavefunction.
    """

    @abstractmethod
    def calc_H_loc(self, wavefunction, xs):
        """
        Evaluates the local Hamiltonian for a given wavefuntion on basis vectors xs.
        """
        pass



class Particles(Hamiltonian):
    """
    Implements a continuous-space Hamiltonian without spins
    """

    def __init__(self, masses, potential, d_space = 3):
        """
        Initialises the local Hamiltonian for N particles.

        Params:
        masses      : array of shape (N,) containing masses of particles
        potential   : scalar function taking batches of shape (batch_dim, N * d_space,) and calculating the potential energy
        d_space     : dimensionality of physical space
        """

        self.hbar = 1

        self.masses = jnp.array(masses)
        self.potential = potential
        self.d_space = d_space
        self.N = self.masses.shape[0]

        self.mass_mat = jnp.diag(1/jnp.repeat(self.masses, self.d_space))


    def calc_H_loc(self, wavefunction, parameters, xs):
        """
        Calculates H_loc on a batch of shape = (batch_dim, N * d_space)
        """

        mass_laplace = lambda x: jnp.trace(self.mass_mat * jax.hessian(wavefunction.calc_psi_single, argnums = 1)(parameters, x))
        self.batch_mass_laplace = jax.vmap(mass_laplace, in_axes = 0)

        T = -self.hbar**2*self.batch_mass_laplace(xs) / 2 / wavefunction.calc_psi(parameters, xs)
        V = self.potential(xs)

        return T + V