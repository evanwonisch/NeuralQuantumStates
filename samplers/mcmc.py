from functools import partial
import jax.numpy as jnp
import jax.random

class Sampler:
    """
    This class wraps the the Metropolis Hasting Sampler.
    """
    def __init__(self, rkey, logprob, variance):
        self.rkey = rkey
        self.logprob = logprob
        self.variance = variance

    def next_element(self, element):
        """
        Performs a Metropolis step. Returns new element and boolean value if proposed sample was accepted.
        """
        rkey, subkey = jax.random.split(self.rkey)

        proposal = element + jax.random.normal(rkey, shape = element.shape) * jnp.sqrt(self.variance)

        ratio = jnp.exp(self.logprob(proposal) - self.logprob(element))

        rkey, subkey = jax.random.split(self.rkey)

        self.rkey = subkey # update key

        if jax.random.uniform(rkey) < ratio:
            return proposal, True
        else:
            return jnp.copy(element), False
        
    def sample(self, initial, N_samples, discard = 0):
        """
        Samples a markov chain of length N_samples
        """
        samples = [jnp.array(initial)]
        n = 0
        for i in range(1, N_samples):
            next, accepted = self.next_element(samples[-1])
            samples.append(next)
            n += int(accepted)

        return jnp.array(samples[discard:]), n/N_samples
    
    def sample_chains(self, shape, N_chains, N_samples, span = 1, discard = 0):
        chains = []
        tot_ratio = 0
        for i in range(N_chains):
            rkey, subkey = jax.random.split(self.rkey)
            initial = jax.random.uniform(rkey, shape) * span*2 - span
            chain, ratio = self.sample(initial, N_samples, discard)
            chains.append(chain)
            tot_ratio += ratio

        return jnp.array(chains), tot_ratio/N_chains
    
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