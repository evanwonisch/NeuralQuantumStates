from abc import ABC, abstractmethod
from functools import partial
import jax.numpy as jnp
import jax.random


class MCMC:
    """
    This class wraps the the Metropolis Hasting Sampler.
    """
    def __init__(self, logprob, shape, variance):
        self.logprob = logprob
        self.shape = shape
        self.variance = variance

    @partial(jax.jit, static_argnames=['self'])
    def propose(self, key, element):
        """
        Proposes a new sample.
        """
        key, subkey = jax.random.split(key)
        return subkey, element + jax.random.normal(subkey, shape = self.shape) * jnp.sqrt(self.variance)

    @partial(jax.jit, static_argnames=['self'])
    def next_element(self, key, element):
        """
        Performs a Metropolis step. Returns new element and boolean value if proposed sample was accepted.
        """
        key, subkey = jax.random.split(key)

        subsubkey, proposal = self.propose(key, element)

        ratio = jnp.exp(self.logprob(proposal) - self.logprob(element))

        return jnp.where(jax.random.uniform(subsubkey) < ratio, proposal,  jnp.copy(element))
    
    @partial(jax.jit, static_argnames=['self', 'N_samples'])
    def sample(self, rkey, initial, N_samples):
        """
        Samples a markov chain of length N_samples
        """
        
        def f(data, bob):
            next_item = self.next_element(data[0], data[1])
            key, _ = jax.random.split(data[0])
            return ((key, next_item), next_item)
        
        return jax.lax.scan(f, (rkey, initial), jnp.zeros((N_samples,) + self.shape))[1]
    
    @partial(jax.jit, static_argnames=['self', 'N_samples', 'N_chains'])
    def sample_chains(self, key, N_chains, N_samples, span = 1):
        vmapped = jax.vmap(lambda key, initial: self.sample(key, initial, N_samples), in_axes=0)

        keys = jax.random.split(key, N_chains)
        initials = jax.random.uniform(key, (N_chains,) + self.shape) * 2 * span - span

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