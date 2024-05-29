import jax
import jax.numpy as jnp
import netket
from functools import partial

##-------------------------------------------------------
##-------------------------------------------------------
##
## A package for evaluating the quantum metric tensor
##
##-------------------------------------------------------
##-------------------------------------------------------

#------------------------
# when providing samples
#------------------------

def eval_S1(orbital, parameters, samples, v):
    f = lambda param: orbital.calc_logpsi(param, samples)
    f_conj = lambda param: jnp.conjugate(orbital.calc_logpsi(param, samples))
    N = samples.shape[0]

    a = jax.jvp(f, (parameters,), (v,))[1]
    b = netket.jax.vjp(f_conj, parameters)[1](a)[0]
    return jax.tree_util.tree_map(lambda x: 1/N*x, b)

def eval_S2(orbital, parameters, samples, v):
    f = lambda param: orbital.calc_logpsi(param, samples)
    N = samples.shape[0]
    e = jnp.ones(N)

    a = jnp.sum(jax.jvp(f, (parameters,), (v,))[1]).real
    b = netket.jax.vjp(f, parameters)[1](e)[0]
    return  jax.tree_util.tree_map(lambda v: 1/N**2 * a * v.real, b)

def eval_S(orbital, parameters, samples, v):
    s1 = eval_S1(orbital, parameters, samples, v)
    s2 = eval_S2(orbital, parameters, samples, v)
    return jax.tree_util.tree_map(lambda a,b: jnp.real(a-b), s1, s2)

@partial(jax.jit, static_argnames=['ansatz'])
def apply_g(ansatz, samples, primals, tangent, eps = 1e-4):
    u = eval_S(ansatz, primals, samples, tangent)
    return jax.tree_util.tree_map(lambda a, b: a + eps * b, u, tangent)

@partial(jax.jit, static_argnames=['ansatz'])
def inverse_g(ansatz, samples, primals, cotangent, eps = 1e-4):
    A = lambda tangents : apply_g(ansatz, samples, primals, tangents, eps)
    return jax.scipy.sparse.linalg.cg(A, cotangent)[0]

#-------------------------
# in full summation
#-------------------------


def full_psi(ansatz, param, all_states):
    return ansatz.calc_psi(param, all_states)

def full_lnK(ansatz, param, all_states):
    psi = ansatz.calc_psi(param, all_states)
    return 0.5*jnp.log(jnp.sum(jnp.conj(psi) * psi))

def full_eval_S1(ansatz, parameters, v, all_states):
    f = lambda theta: full_psi(ansatz, theta, all_states)
    K = jnp.exp(full_lnK(ansatz, parameters, all_states))

    a = jax.jvp(f, (parameters,), (v,))[1]
    b = netket.jax.vjp(f, parameters)[1](a)[0]
    return jax.tree_util.tree_map(lambda x: 1 * x.real / K**2, b)

def full_eval_S2(ansatz, parameters, v, all_states):
    f = lambda theta: full_lnK(ansatz, theta, all_states)

    a = jax.jvp(f, (parameters,), (v,))[1]
    b = netket.jax.vjp(f, parameters)[1](jnp.array(1.))[0]
    return  jax.tree_util.tree_map(lambda v: a * v, b)

def full_eval_S(orbital, parameters, v, all_states):
    s1 = full_eval_S1(orbital, parameters, v, all_states)
    s2 = full_eval_S2(orbital, parameters, v, all_states)
    return jax.tree_util.tree_map(lambda a,b: jnp.real(a-b), s1, s2)

@partial(jax.jit, static_argnames=['ansatz'])
def full_apply_g(ansatz, primals, tangent, all_states, eps = 1e-4):
    u = full_eval_S(ansatz, primals, tangent, all_states)
    return jax.tree_util.tree_map(lambda a, b: a + eps * b, u, tangent)

@partial(jax.jit, static_argnames=['ansatz'])
def full_inverse_g(ansatz, primals, cotangent, all_states, eps = 1e-4):
    A = lambda tangents : full_apply_g(ansatz, primals, tangents, all_states, eps)
    return jax.scipy.sparse.linalg.cg(A, cotangent)[0]