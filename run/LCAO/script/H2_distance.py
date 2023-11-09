import sys
sys.path.insert(0, "../../..")

import jax
import jax.numpy as jnp

import jax.scipy
import jax.scipy.special

from tqdm import tqdm
from functools import partial

import module.samplers
import module.hamiltonians
import module.wavefunctions

import flax.linen as nn

import optax
import numpy as np
from matplotlib import pyplot as plt


def run(dist, key, epoch = 30):

    R = jnp.array([[dist/2,0,0],[-dist/2,0,0]])     # nuclei positions
    k = jnp.array([1.,1.])                          # potential energy coefficients

    def calc_E_nuc():
        acc = 0
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                d = jnp.sqrt(jnp.sum((R[i] - R[j])**2))
                acc += k[i] * k[j] / d


        return acc

    E_nuc = calc_E_nuc()


    #
    # The potential energy function depending on the coordinates
    #
    def potential(x):
        x = jnp.expand_dims(x, axis = 1)
        r = jnp.expand_dims(R,  axis = 0)
        d = jnp.sqrt(jnp.sum((x - r)**2, axis = 2))
        return - jnp.sum(k/d ,axis = 1) + E_nuc


    # the hamiltonian
    hamiltonian = module.hamiltonians.Particles(masses = [1], potential = potential)


    def act(x):
        # return x / (1 + jnp.exp(-x)) - 0.1 * x
        return jnp.log(jnp.cosh(x))

    class NN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32, name="dense1", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(32, name="dense2", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(32, name="dense3", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(32, name="dense4", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(1, name="dense5", bias_init=jax.random.normal)(x)
            x = act(x)
            return -jnp.squeeze(x)
        

    class Ansatz(module.wavefunctions.Wavefunction):
        def __init__(self):
            super().__init__(input_shape = (3,))

            self.nn = NN()
            self.lcao = module.wavefunctions.LCAO(R, k)


        def init_param(self, key):
            nn_param = self.nn.init(key, jnp.empty((2,)))

            return nn_param

        @partial(jax.jit, static_argnames=['self'])
        def eval_nn(self, parameters, x):
            x_coord = x[:, 0]                       # molecular axis
            r = jnp.sqrt(x[:, 1]**2 + x[:, 2]**2 + 1e-4)   # radial distance
            phi = jnp.arctan2(x[:, 2], x[:, 1])     # angle

            coord = jnp.stack((x_coord, r), axis = 1)
            coord2 = jnp.stack((-x_coord, r), axis = 1)

            return self.nn.apply(parameters, coord) + self.nn.apply(parameters, coord2)

        @partial(jax.jit, static_argnames=['self'])
        def calc_logpsi(self, parameters, x):
            return self.lcao.calc_logpsi(jnp.array([1., 1.]), x) + self.eval_nn(parameters, x)
        

    ansatz = Ansatz()
    key, _ = jax.random.split(key)
    param = ansatz.init_param(key)

    sampler = module.samplers.MCMC(ansatz, 0.5)
    subsampling = 10

    def get_samples(N):
        """
        Returns decorrelated samples.
        """
        samples, ratio = sampler.sample(key, param, jax.random.normal(key, shape = (3,)), N*subsampling)
        return samples[0:-1:subsampling], ratio
    

    @partial(jax.jit, static_argnames=['hamiltonian', 'orbital'])
    def calc_grad_E(hamiltonian, orbital, parameters, samples):
        """
        Calculates the gradient of the energy and the energy itself on a batch of samples.
        """

        E_loc = hamiltonian.calc_H_loc(orbital, parameters, samples)
        E = jnp.mean(E_loc)
        dE = jnp.std(E_loc)
        grad_log = orbital.grad_logpsi(parameters, samples)

        m = lambda tree: 2*jnp.real(jnp.mean(jnp.swapaxes(jnp.swapaxes(tree, 0, -1) * (E_loc - E), 0, -1), axis = 0))

        return jax.tree_util.tree_map(m, grad_log), E, dE
    

    stats = {"E":[], "dE":[], "ratio":[], "N_samples": []}
    param = ansatz.init_param(key)

    # define
    optimizer = optax.adam(learning_rate=0.005)

    # initialise
    optimizer_state = optimizer.init(param)


    N = 9000

    for i in tqdm(range(epoch)):
        key, _ = jax.random.split(key)
        samples, ratio = get_samples(N)
        grad, E, dE = calc_grad_E(hamiltonian, ansatz, param, samples)

        stats["E"].append(E)
        stats["dE"].append(dE)
        stats["ratio"].append(ratio)
        stats["N_samples"].append(N)
        
        updates, optimizer_state = optimizer.update(grad, optimizer_state, param)

        param = optax.apply_updates(param, updates)


    N = 10000
    samples, ratio = get_samples(N)
    H_loc = hamiltonian.calc_H_loc(ansatz, param, samples)
    
    return jnp.mean(H_loc), jnp.std(H_loc)/jnp.sqrt(N), jnp.std(H_loc)