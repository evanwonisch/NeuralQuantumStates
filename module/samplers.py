from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random


class MCMC(ABC):
    """
    This class wraps the the Metropolis Hasting Sampler.
    """
    def __init__(self, wavefunction, variance):
        self.wavefunction = wavefunction
        self.shape = wavefunction.input_shape
        self.variance = variance

    @abstractmethod
    def propose(self, key, element):
        """
        Proposes a new Sample.
        """
        pass

    def next_element(self, key, element, parameters):
        """
        Performs a Metropolis step. Returns new element and boolean value if proposed sample was accepted.
        """
        key, subkey = jax.random.split(key)

        subsubkey, proposal = self.propose(key, element)

        ratio = jnp.exp(self.wavefunction.calc_logprob_single(parameters, proposal) - self.wavefunction.calc_logprob_single(parameters, element))

        return jnp.where(jax.random.uniform(subkey) < ratio, proposal,  jnp.copy(element)), jnp.where(jax.random.uniform(subkey) < ratio, 1, 0)
    

    @partial(jax.jit, static_argnames=['self', 'N_samples'])
    def sample(self, rkey, parameters, initial, N_samples):
        """
        Samples a markov chain of length N_samples
        """
        
        def f(data, bob):
            next_item, accept = self.next_element(data[0], data[1], data[3])
            key, _ = jax.random.split(data[0])
            return ((key, next_item, data[2] + accept, data[3]), next_item)
        
        carry, samples = jax.lax.scan(f, (rkey, initial, 0, parameters), jnp.zeros((N_samples,) + self.shape))

        return samples, carry[2]/N_samples
    
    @partial(jax.jit, static_argnames=['self', 'N_samples', 'N_chains'])
    def sample_chains(self, key, parameters, N_chains, N_samples):
        vmapped = jax.vmap(lambda key, initial: self.sample(key, parameters, initial, N_samples), in_axes=0)

        keys = jax.random.split(key, N_chains)
        initials = self.wavefunction.propse_initials(key, parameters, N_chains)

        return vmapped(keys, initials)

    
    def chain_quality(self, chains):
        """
        This function compares mean and variance of the chains and returns a similarity matrix.
        Thereby, the distance between the mean of each chain is given in units of the average variance of the respective chains.
        """
        means = jnp.mean(chains, axis = 1)
        vars = jnp.mean(jnp.std(chains, axis = 1), axis = 1)

        d = jnp.mean((jnp.expand_dims(means, axis = 1) - jnp.expand_dims(means, axis = 0))**2, axis = 2)
        v = 0.5*(jnp.expand_dims(vars, axis = 1) + jnp.expand_dims(vars, axis = 0))

        return d/v
    

class MCMCsimple(MCMC):
     
     def propose(self, key, element):
        """
        Proposes a new sample by adding random noise to it.
        """
        key, subkey = jax.random.split(key)

        return subkey, element + jax.random.normal(key, shape = self.shape) * jnp.sqrt(self.variance)
