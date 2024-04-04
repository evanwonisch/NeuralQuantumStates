import jax.numpy as jnp


def h_C3(x):
    """
    C3-differentiable cutoff function. Cutoff happens between -1 and 1.
    """
    return 0.5*jnp.where(x < -1, -1, jnp.where(x < 1,  2.1875*x+  -2.1875*x**3+  1.3125*x**5+ -0.3125*x**7  , 1)) +0.5


def h_C2(x):
    """
    C2-differentiable cutoff function. Cutoff happens between -1 and 1.
    """
    return 0.5*jnp.where(x < -1, -1, jnp.where(x < 1, 15/8*x -5/4*x**3 + 3/8*x**5 , 1)) + 0.5