import jax
import jax.numpy as jnp
from jax import jit
import math
from functools import partial


@partial(jax.jit, static_argnames = ["l", "m"])
def spherical_harmonic(l, m, phi, theta):
    """
    Calculates the spehrical harmonics Y_l^m at positions phi and theta, which have to be 1D arrays of the same shape. 
    """
    l_ = jnp.ones_like(phi, dtype="int")*l
    m_ = jnp.ones_like(phi, dtype="int")*m

    y = jax.scipy.special.sph_harm(m_, l_, phi, theta, n_max=l)
    return y


@partial(jax.jit, static_argnames=['n','k'])    
def ass_lag(n, k, x):
    if n < 0:
        raise Exception("n has to be non-negativ in asoociated Laguerre polynomials")

    if n == 0:
        return 1

    carry = jnp.array([-x+k+1,1])

    for i in range(1, n):
        A = jnp.array([[(2*i+1+k-x)/(i+1), 1],[-(i+k)/(i+1),0]]).T
        carry = A @ carry

    return carry[0]

ass_lag = jax.vmap(ass_lag, in_axes=[None, None, 0])

@partial(jax.jit, static_argnames=['n','l','Z'])    
def radial_part(n, l, Z, r):
    a = (2*Z*r/n)**l * jnp.exp(-Z*r/n) * ass_lag(n-l-1, 2*l+1, 2*Z*r/n) * math.factorial(n + l)
    A = (2*Z/n)**3*math.factorial(n-l-1)/2/n/math.factorial(n+l)**3
    return a * jnp.sqrt(A)

@partial(jax.jit, static_argnames=['n','l','m','Z']) 
def evaluate(n, l, m, Z, position):
    x = position[:,0]
    y = position[:,1]
    z = position[:,2]

    r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)

    radial = radial_part(n, l, Z, r)
    angular = spherical_harmonic(l, m, phi, theta)

    return radial * angular