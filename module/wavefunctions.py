from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random
import math

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


class HydrogenicOrbital(Wavefunction):
    def __init__(self, n, l, m, Z = 1):
        """
        Creates a normalised hydrogenic orbital with quantum numbers n,l,m and nuclear charge Z in 3 dimensions.
        """
        super().__init__(input_shape = (3,))

        assert n > 0, "n has to be positive"
        assert l >= 0, "l has to be non negative"
        assert l < n, "l has to be less than n"
        assert jnp.abs(m) <= l, "abs(m) has to be less or equal to l"

        self.n = n
        self.l = l
        self.m = m
        self.Z = Z

        self.assoc_laguerre = jax.vmap(self.ass_lag, in_axes=[None,None,0])

    def spherical_harmonic(self, l, m, phi, theta):
        """
        Calculates the spherical harmonics Y_l^{m} at positions phi and theta.
        """
        theta = jnp.array(theta)

        if not isinstance(phi, jax.numpy.ndarray):
            phi = jnp.ones_like(theta) * phi
        
        l = jnp.ones_like(phi, dtype="int")*l
        m = jnp.ones_like(phi, dtype="int")*m

        y = jax.scipy.special.sph_harm(m, l, phi, theta)
        return y


    @partial(jax.jit, static_argnames=['self','n','k'])    
    def ass_lag(self, n, k, x):
        """
        Calculazed the n,k-th associated Laguerre polynomial at position x.
        """
        if n < 0:
            raise Exception("n has to be non-negative in asoociated Laguerre polynomials")

        if n == 0:
            return 1

        carry = jnp.array([-x+k+1,1])

        for i in range(1, n):
            A = jnp.array([[(2*i+1+k-x)/(i+1), 1],[-(i+k)/(i+1),0]]).T
            carry = A @ carry

        return carry[0]

    @partial(jax.jit, static_argnames=['self'])    
    def radial_part(self, r):
        """
        Calculates the properly normalised radial part of the wavefunction.
        """
        a = (2*self.Z*r/self.n)**self.l * jnp.exp(-self.Z*r/self.n) * self.assoc_laguerre(self.n-self.l-1, 2*self.l+1, 2*self.Z*r/self.n) * math.factorial(self.n + self.l)
        A = (2*self.Z/self.n)**3*math.factorial(self.n-self.l-1)/2/self.n/math.factorial(self.n+self.l)**3
        return a * jnp.sqrt(A)

    def evaluate(self, position):
        x = position[:,0]
        y = position[:,1]
        z = position[:,2]
        
        r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = jnp.arccos(z / r)
        phi = jnp.arctan2(y, x)

        radial = self.radial_part(r)
        angular = self.spherical_harmonic(self.l, self.m, phi, theta)

        return radial * angular
    
    def calc_logpsi(self, parameters, position):
        return jnp.log(self.evaluate(position))