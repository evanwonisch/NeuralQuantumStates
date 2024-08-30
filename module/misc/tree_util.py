#----------------------------------------
# A little package which allows easy manipulation of trees
#----------------------------------------

import jax
import jax.numpy as jnp

euclid_norm_sq = lambda v: jax.tree_util.tree_reduce(lambda c,d : c + d, jax.tree_util.tree_map(lambda a: jnp.sum(a**2), v))

t_sum = lambda u,v: jax.tree_util.tree_map(lambda x,y:x+y, u,v)
t_sub = lambda u,v: jax.tree_util.tree_map(lambda x,y:x-y, u,v)
t_mul = lambda u,v: jax.tree_util.tree_map(lambda x,y:x*y, u,v)
t_div = lambda u,v: jax.tree_util.tree_map(lambda x,y:x/y, u,v)
s_mul = lambda s,v: jax.tree_util.tree_map(lambda x: s*x, v)

t_zeros_like = lambda v: jax.tree_util.tree_map(lambda a: a*0, v)
t_ones_like = lambda v: jax.tree_util.tree_map(lambda a: a*0 + 1, v)

t_dim = lambda tree: sum(x.size for x in jax.tree_util.tree_leaves(tree))