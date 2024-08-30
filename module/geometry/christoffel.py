import jax
import jax.numpy as jnp
import netket
from functools import partial
import module.geometry.metric as metric
import module.misc.tree_util as tree_util

##--------------------------------------------------------
##--------------------------------------------------------
##
## A package for computing Christoffel symbols and geodesic corrections
##
##--------------------------------------------------------
##--------------------------------------------------------


##-----------------------
## when providing samples
##-----------------------

def a(ansatz, samples, primals, gamma_dot, eps = 1e-4):
    f = lambda primals : metric.apply_g(ansatz, samples, primals, gamma_dot, eps)
    res = jax.jvp(f, (primals,), (gamma_dot,))
    return res[1]

def a_approx(ansatz, samples, old_samples, gamma, old_gamma, gamma_dot, epsilon, eps = 1e-4):
    x = tree_util.s_mul(1/epsilon, tree_util.t_sub(metric.apply_g(ansatz, samples, gamma, gamma_dot, eps), metric.apply_g(ansatz, old_samples, old_gamma, gamma_dot, eps)))
    return x


def b(ansatz, samples, primals, gamma_dot, eps = 1e-4):

    ## this sums all elements of a tree
    g = lambda v: jax.tree_util.tree_reduce(lambda c,d : c + d, jax.tree_util.tree_map(lambda a: jnp.sum(a), v))

    ## this computes the norm square of gamma dot
    f = lambda primals : g(jax.tree_util.tree_map(lambda c, d : c * d, gamma_dot, metric.apply_g(ansatz, samples, primals, gamma_dot, eps)))
    
    res = jax.vjp(f, primals)[1](jnp.array(1.))[0]
    return res

@partial(jax.jit, static_argnames=['ansatz'])
def geodesic_correction(ansatz, samples, primals, gamma_dot, eps = 1e-4):
    lower = jax.tree_util.tree_map(lambda c,d: 2*c - d, a(ansatz, samples, primals, gamma_dot, eps), b(ansatz, samples, primals, gamma_dot, eps))
    return jax.tree_util.tree_map(lambda k:0.5*k, lower)

@partial(jax.jit, static_argnames=['ansatz'])
def geodesic_correction_approx(ansatz, samples, old_samples, primals, old_primals, gamma_dot, epsilon, eps = 1e-4):
    lower = jax.tree_util.tree_map(lambda c,d: 2*c - d, a_approx(ansatz, samples, old_samples, primals, old_primals, gamma_dot, epsilon, eps), b(ansatz, samples, primals, gamma_dot, eps))
    return jax.tree_util.tree_map(lambda k:0.5*k, lower)

#------------------------------------------------------
#  in ful summation
#------------------------------------------------------

def full_a(ansatz, primals, gamma_dot, all_states, eps = 1e-4):
    f = lambda primals : metric.full_apply_g(ansatz, primals, gamma_dot, all_states, eps)
    res = jax.jvp(f, (primals,), (gamma_dot,))
    return res[1]

def full_b(ansatz, primals, gamma_dot, all_states, eps = 1e-4):

    ## this sums all elements of a tree
    g = lambda v: jax.tree_util.tree_reduce(lambda c,d : c + d, jax.tree_util.tree_map(lambda a: jnp.sum(a), v))

    ## this computes the norm square of gamma dot
    f = lambda primals : g(jax.tree_util.tree_map(lambda c, d : c * d, gamma_dot, metric.full_apply_g(ansatz, primals, gamma_dot, all_states, eps)))
    
    res = jax.vjp(f, primals)[1](jnp.array(1.))[0]
    return res

@partial(jax.jit, static_argnames=['ansatz'])
def full_geodesic_correction(ansatz, primals, gamma_dot, all_states, eps = 1e-4):
    lower = jax.tree_util.tree_map(lambda c,d: 2*c - d, full_a(ansatz, primals, gamma_dot, all_states, eps), full_b(ansatz, primals, gamma_dot, all_states, eps))
    return jax.tree_util.tree_map(lambda k:0.5*k, lower)