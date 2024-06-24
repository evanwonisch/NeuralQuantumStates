import sys
sys.path.insert(0, "../../..")

import jax
import jax.numpy as jnp

import jax.scipy
import jax.scipy.special

from tqdm import tqdm
from functools import partial

import module
import module.samplers
import module.wavefunctions
import module.misc.cutoffs as cutoffs

import flax.linen as nn

import optax
import numpy as np
from matplotlib import pyplot as plt

import netket.nn.activation
import pickle

import netket as nk

from scipy.sparse.linalg import eigsh

import module.geometry.metric as metric
import module.geometry.christoffel as christoffel
import module.misc.tree_util as tree_util

# key = jax.random.PRNGKey(1)

##--------------------------
## Hamiltonian
##--------------------------
L = 16
graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=graph.n_nodes, total_sz=0)
J = -1.
hamiltonian = nk.operator.LocalOperator(hi)
for (i,j) in graph.edges():
    hamiltonian = hamiltonian - J * nk.operator.spin.sigmax(hi, i) @ nk.operator.spin.sigmax(hi, j)
    hamiltonian = hamiltonian - J * nk.operator.spin.sigmay(hi, i) @ nk.operator.spin.sigmay(hi, j)
    hamiltonian = hamiltonian - J * nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j)


hamiltonian_jax = hamiltonian.to_pauli_strings().to_jax_operator()
ha = hamiltonian.to_sparse()
E_f = -28.569185442467138
def calc_H_loc(orbital, parameters, samples):
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(samples)
    
    logpsi_sigma = orbital.calc_logpsi(parameters, samples)
    logpsi_eta = orbital.calc_logpsi(parameters, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1) 
    
    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)
    
    return res
## Construct Neural Network
def act(x):
    return netket.nn.activation.log_cosh(x)
class NN(nn.Module):
    """
    Constructs a neural network.

    possible activation functions:
    netket.nn.activation.log_cosh(x)
    nn.softplus()
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16*3, name="dense1", param_dtype="float64", bias_init=jax.nn.initializers.normal(stddev=0.1))(x)
        x = act(x)
        x = nn.Dense(2, name="dense2", param_dtype="float64", use_bias=False)(x)

        return jnp.squeeze(x)
## Construct Ansatz
class Ansatz(module.wavefunctions.Wavefunction):
    def __init__(self):
        super().__init__(input_shape = (graph.n_nodes,))

        self.nn = NN()
        # self.phase = NN()
        self.unravel = None


    def init_param(self, key):
        nn_param = self.nn.init(key, jnp.empty((graph.n_nodes,)))
        # key, _ = jax.random.split(key)
        # phase_param = self.nn.init(key, jnp.empty((graph.n_nodes,)))

        # param =  {"nn": nn_param, "phase": phase_param}
        param = {"nn":nn_param}
        flat_param, unravel = jax.flatten_util.ravel_pytree(param)
        self.unravel = unravel
        return flat_param
    

    @partial(jax.jit, static_argnames=['self'])
    def shift_samples(self, samples):
        repeated = jnp.repeat(jnp.expand_dims(samples, axis = 0), g.n_nodes, axis = 0)
        shifted = jax.lax.scan(self.f, 0, repeated)[1]
        return shifted

    def f(self, state, slice):
        out = jnp.roll(slice, state, axis = -1)
        return state + 1, out
    
    def calc_logpsi(self, parameters, x):
        param = self.unravel(parameters)
       
        # shifted = self.shift_samples(x)
        # log_psi = self.nn.apply(parameters["nn"], shifted)
        # sym = jnp.sum(log_psi, axis = 0)

        a = self.nn.apply(param["nn"], x)
        return a[...,0]+1.j*a[...,1]
## Sampler
class SpinSwapSampler(module.samplers.MCMC):
    def propose(self, key, element):
        subkey, _ = jax.random.split(key)

        index = jax.random.randint(subkey, (), 0, graph.n_nodes)
        neigh = (index + 1)
        h = jnp.copy(element[neigh])

        proposal = element.at[neigh].set(element[index])
        proposal = proposal.at[index].set(h)

        return subkey, proposal

## Optimisation
@partial(jax.jit, static_argnames=['orbital'])
def calc_grad_E(orbital, parameters, samples):
    """
    Calculates the gradient of the energy and the energy itself on a batch of samples.
    """

    E_loc = calc_H_loc(orbital, parameters, samples)
    E = jnp.mean(E_loc.real)
    dE = jnp.std(E_loc.real)

    logpsi_red = lambda parameters: jnp.conjugate(orbital.calc_logpsi(parameters, samples))
    y, v = netket.jax.vjp(logpsi_red, parameters)
    N = samples.shape[0]

    m = lambda tree: 2*jnp.real(tree/N)

    return jax.tree_util.tree_map(m, v(E_loc - E))[0], E, dE
def calc_norm_square(ansatz, samples, primals, tangent):
    g = lambda v: jax.tree_util.tree_reduce(lambda c,d : c + d, jax.tree_util.tree_map(lambda a: jnp.sum(a), v))
    return  g(jax.tree_util.tree_map(lambda c, d : c * d, tangent, metric.apply_g(ansatz, samples, primals, tangent)))


def run(key, fname, epoch = 100, N_samples = 1500, lr = optax.constant_schedule(0.009), rcond = optax.constant_schedule(1e-4), beta = optax.constant_schedule(0.3)):
    ansatz = Ansatz()
    param = ansatz.init_param(key)
    sampler = SpinSwapSampler(ansatz, 1)
    def init_chain():
        init = jnp.ones((graph.n_nodes))
        choice = jax.random.choice(key, graph.n_nodes, (int(graph.n_nodes/2),), replace=False)
        return init.at[choice].mul(-1)

    subsampling = graph.n_nodes
    def get_samples(N):
        """
        Returns decorrelated samples.
        """
        samples, ratio = sampler.sample(key, param, init_chain() , N*subsampling)
        return samples[0:-1:subsampling], ratio
    
    stats = {"energy":[], "variance":[], "accept":[], "N_samples": [], "rel_err":[]}

    ## initialise parameters
    samples = get_samples(N_samples)[0]
    grad, E, dE = calc_grad_E(ansatz, param, samples)
    g = metric.get_g(ansatz, samples, param, 0)
    initial_gradient = jnp.linalg.pinv(g, hermitian=True, rcond=rcond(0)) @ grad
    old_param = param + lr(0)/(1 - beta(0)) * initial_gradient

    for i in range(epoch):
        key, _ = jax.random.split(key)

        # samples
        samples, ratio = get_samples(N_samples)

        # old update
        delta_param = param - old_param


        # geodesic correction
        cor = christoffel.geodesic_correction(ansatz, samples, param, delta_param, eps = 0)

        # compute inversde metric
        g = metric.get_g(ansatz, samples, param, eps = 0)
        inv_g = jnp.linalg.pinv(g, hermitian=True, rcond=rcond(i))

        #
        # parallel transport old momentum and damp
        #
        trans = beta(i) * (delta_param - inv_g @ cor)

        #
        # nesterov gradient
        #
        # gradient
        grad, E, dE = calc_grad_E(ansatz, param + trans, samples)
        nat_grad = inv_g @ grad

        # update
        new_param = param + trans - lr(i) * nat_grad
        old_param = param
        param = new_param

        if jnp.any(jnp.isnan(param)):
            raise Exception("The optimisation scheme has diverged.")
        
        # store statistics
        stats["energy"].append(E)
        stats["variance"].append(dE)
        stats["accept"].append(ratio)
        stats["N_samples"].append(N_samples)
        stats["rel_err"].append(jnp.abs(E - E_f)/-E_f)

    ##########################################
    ##### calculate good final energy estimate
    ##########################################

    hamiltonian_jax = hamiltonian.to_pauli_strings().to_jax_operator()
    def calc_H_loc(orbital, parameters, samples):
        eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(samples)
        
        logpsi_sigma = orbital.calc_logpsi(parameters, samples)
        logpsi_eta = orbital.calc_logpsi(parameters, eta)
        logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1) 
        
        res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)
        
        return res
    N_samples = 30000
    key, _ = jax.random.split(key)
    samples, ratio = get_samples(N_samples)
    H_loc = calc_H_loc(ansatz, param, samples).real

    E_estimate = jnp.mean(H_loc)
    u_E_estimate = jnp.std(H_loc)/jnp.sqrt(N_samples)
    rel_err_estimate = (jnp.mean(H_loc) - E_f)/jnp.abs(E_f)
    final_summary = jnp.array([E_estimate, u_E_estimate, rel_err_estimate])
    final_param = param

    ##########################################
    # save the data
    ##########################################
    numpy_stats = {}
    for key_ in stats.keys():
        numpy_stats[key_] = np.array(stats[key_])

    for stat in numpy_stats:
        print("saved to ", fname)
        np.save(fname+"_"+stat, numpy_stats[stat])

    
    np.save(fname+"_"+"param", final_param)
    np.save(fname+"_"+"summary", final_summary)


#
# RUN the stuff
#

index = int(sys.argv[1])
r_seed = int(sys.argv[2])
hyper = jnp.load("/mnt/beegfs/workdir/evan.wonisch/NeuralQuantumStates/run/heisenberg/hyper_cube_fine.npy")

final_lr = hyper[index, 0]
final_rcond = hyper[index, 1]

lr = optax.constant_schedule(final_lr)
rcond = optax.constant_schedule(final_rcond)
beta = optax.constant_schedule(0.9)

key = jax.random.PRNGKey(r_seed)
run(key, "/mnt/beegfs/workdir/evan.wonisch/NeuralQuantumStates/data/gridsearch_fine/beta=0.9/hyper_i="+str(index)+"_seed="+str(r_seed), epoch = 1500, N_samples = 1500, lr = lr, rcond = rcond, beta = beta)