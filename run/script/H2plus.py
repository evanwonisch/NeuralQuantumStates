import sys
sys.path.insert(0, "../..")

import jax
import jax.numpy as jnp

import jax.scipy
import jax.scipy.special

from tqdm import tqdm
from functools import partial

import module.samplers
import module.hamiltonians
import module.wavefunctions
import module.misc.cutoffs as cutoffs

import flax.linen as nn

import optax
import numpy as np
from matplotlib import pyplot as plt

import netket.nn.activation
import pickle


def run(dist, verbose = False):
    ## Potential Energy
    key = jax.random.PRNGKey(0)
    R = jnp.array([[dist/2,0,0],[-dist/2,0,0]])     # nuclei positions
    k = jnp.array([1.,1.])                          # potential energy coefficients (nuclear charge)
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
    def potential(x, params):
        x = jnp.expand_dims(x, axis = 1)
        r = jnp.expand_dims(R,  axis = 0)
        d = jnp.sqrt(jnp.sum((x - r)**2, axis = 2))
        return -jnp.sum(k/d ,axis = 1) + E_nuc


    # the hamiltonian
    hamiltonian = module.hamiltonians.Particles(masses = [1], potential = potential)

    ## Construct Neural Network
    def act(x):
        return nn.softplus(x)
    class NN(nn.Module):
        """
        Constructs a neural network.

        possible activation functions:
        netket.nn.activation.log_cosh(x)
        nn.softplus()
        """
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(40, name="dense2", param_dtype="float64", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(40, name="dense3", param_dtype="float64", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(25, name="dense4", param_dtype="float64", bias_init=jax.random.normal)(x)
            x = act(x)
            x = nn.Dense(1, name="dense5", param_dtype="float64", bias_init=jax.random.normal)(x)
            return jnp.squeeze(x)
    ## Construct Ansatz
    class Ansatz(module.wavefunctions.Wavefunction):
        def __init__(self):
            super().__init__(input_shape = (3,))

            self.nn = NN()


        def init_param(self, key):
            a = jax.random.normal(key)**2
            key, key2 = jax.random.split(key)
            b = jax.random.normal(key)**2

            nn_param = self.nn.init(key2, jnp.empty((2,)))

            return {"alpha": a, "beta": b, "nn":nn_param}
        
        def eval_nn(self, parameters, x):
            r1 = jnp.sqrt((x[:, 0] - dist/2)**2 + x[:, 1]**2 + x[:, 2]**2)
            r2 = jnp.sqrt((x[:, 0] + dist/2)**2 + x[:, 1]**2 + x[:, 2]**2)

            coord = jnp.stack((r1, r2), axis = 1)
            coord2 = jnp.stack((r2, r1), axis = 1)

            return self.nn.apply(parameters["nn"], coord) + self.nn.apply(parameters["nn"], coord2)

        
        def calc_logpsi(self, parameters, x):
            r1 = jnp.sqrt((x[:, 0] - dist/2)**2 + x[:, 1]**2 + x[:, 2]**2)
            r2 = jnp.sqrt((x[:, 0] + dist/2)**2 + x[:, 1]**2 + x[:, 2]**2)

            lamb = (r1 + r2) / dist
            mu = (r1 - r2) / dist

            psi = jnp.exp(-0.5*dist*parameters["alpha"]*lamb)*jnp.cosh(0.5*dist*parameters["beta"]*mu)

            nn_out = self.eval_nn(parameters, x)

            return jnp.log(psi) + nn_out
        
    ansatz = Ansatz()
    key, _ = jax.random.split(key)
    param = ansatz.init_param(key)

    ## Sampler
    class SymmetrySampler(module.samplers.MCMC):
        def propose(self, key, element):
            subkey, _ = jax.random.split(key)

            mirrorprob = 0.01

            var = jax.random.normal(key, shape = self.shape) * jnp.sqrt(self.variance)
            inv = -element

            decide = jax.random.uniform(key)

            proposal = jnp.where(decide < mirrorprob, inv, element + var)

            return subkey, proposal
    sampler = SymmetrySampler(ansatz, 0.5)
    subsampling = 10
    def get_samples(N):
        """
        Returns decorrelated samples.
        """
        samples, ratio = sampler.sample(key, param, jax.random.normal(key, shape = (3,)), N*subsampling)
        return samples[0:-1:subsampling], ratio
    ## Optimisation
    @partial(jax.jit, static_argnames=['hamiltonian', 'orbital'])
    def calc_grad_E(hamiltonian, orbital, parameters, samples):
        """
        Calculates the gradient of the energy and the energy itself on a batch of samples.
        """

        E_loc = hamiltonian.calc_H_loc(orbital, parameters, samples)
        E = jnp.mean(E_loc)
        dE = jnp.std(E_loc)

        logpsi_red = lambda parameters: orbital.calc_logpsi(parameters, samples)
        y, v = jax.vjp(logpsi_red, parameters)
        N = samples.shape[0]

        m = lambda tree: 2*jnp.real(tree/N)

        return jax.tree_util.tree_map(m, v(E_loc - E))[0], E, dE
    def eval_S1(orbital, parameters, samples, v):
        f = lambda param: orbital.calc_logpsi(param, samples)
        N = samples.shape[0]

        a = jax.jvp(f, (parameters,), (v,))[1]
        b = netket.jax.vjp(f, parameters)[1](a)[0]
        return jax.tree_util.tree_map(lambda x: 1/N*x, b)

    def eval_S2(orbital, parameters, samples, v):
        f = lambda param: orbital.calc_logpsi(param, samples)
        N = samples.shape[0]
        e = jnp.ones(N)

        a = jnp.sum(jax.jvp(f, (parameters,), (v,))[1])
        b = jax.vjp(f, parameters)[1](e)[0]
        return  jax.tree_util.tree_map(lambda v: 1/N**2 * a * v, b)

    @partial(jax.jit, static_argnames=['orbital'])
    def eval_S(orbital, parameters, samples, v):
        s1 = eval_S1(orbital, parameters, samples, v)
        s2 = eval_S2(orbital, parameters, samples, v)
        return jax.tree_util.tree_map(lambda a,b: a-b, s1, s2)
    eps = 1e-4

    def reg_S(ansatz, param, samples, v):
        u = eval_S(ansatz, param, samples, v)
        return jax.tree_util.tree_map(lambda a,b: a + eps * b, u, v)

    @jax.jit
    def calc_natural_grad(param, samples):
        A = lambda v: reg_S(ansatz, param, samples, v)
        b, E, dE = calc_grad_E(hamiltonian, ansatz, param, samples)
        return jax.scipy.sparse.linalg.cg(A, b)[0], E, dE

    stats = {"E":[], "dE":[], "ratio":[], "N_samples": []}
    param = ansatz.init_param(key)
    N = 800
    epoch = 350

    # define
    optimizer = optax.sgd(learning_rate=0.01)

    # initialise
    optimizer_state = optimizer.init(param)
    for i in tqdm(range(epoch)):
        key, _ = jax.random.split(key)
        samples, ratio = get_samples(N)
        grad, E, dE = calc_natural_grad(param, samples)

        stats["E"].append(E)
        stats["dE"].append(dE)
        stats["ratio"].append(ratio)
        stats["N_samples"].append(N)
        
        updates, optimizer_state = optimizer.update(grad, optimizer_state, param)

        param = optax.apply_updates(param, updates)

    numpy_stats = {}
    for key_ in stats.keys():
        numpy_stats[key_] = np.array(stats[key_])

    N = 10000
    key, _ = jax.random.split(key)
    samples, ratio = get_samples(N)
    H_loc = hamiltonian.calc_H_loc(ansatz, param, samples)
    print("Acceptance Ratio:", ratio)
    print("Expected Energy:", jnp.mean(H_loc),"+/-", jnp.std(H_loc)/jnp.sqrt(N))
    print("Std of Expected Energy:", jnp.std(H_loc))

    if verbose:
        return jnp.mean(H_loc), jnp.std(H_loc)/jnp.sqrt(N), numpy_stats
    
    else:
        return jnp.mean(H_loc), jnp.std(H_loc)/jnp.sqrt(N)