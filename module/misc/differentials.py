import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jax.jit, static_argnames=['f'])
def _d2(f, x, v):
    g = lambda x: jax.jvp(f, (x,), (v,))[1]
    primals, tangents = jax.jvp(g, (x,), (v,))

    return tangents

_d2_ = jax.vmap(_d2, in_axes = [None, None, 0])


@partial(jax.jit, static_argnames=['f'])
def hessian(f, x):
    return _d2(f, x, jnp.array([1.,0.,0.])) + _d2(f, x, jnp.array([0.,1.,0.])) + _d2(f, x, jnp.array([0.,0.,1.]))

@partial(jax.jit, static_argnames=['f'])
def hess_diag(f, x):
    return _d2_(f, x, jax.nn.one_hot(jnp.arange(x.shape[0]), 3))