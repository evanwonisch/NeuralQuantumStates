import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


#----------------------------------------
# Metric Tensor
#----------------------------------------
def apply_g(primals, tangent):

    theta = primals[0]
    orig_shape = tangent.shape
    tangent = tangent.reshape(-1, 2, 1)

    g = jnp.array([[0.5, 0],[0, 2*jnp.sin(theta/2)**2]])
    res = jnp.matmul(g, tangent)
    return res.reshape(orig_shape)

def inverse_g(primals, cotangent):
    A = lambda tangents : apply_g(primals, tangents)
    return jax.scipy.sparse.linalg.cg(A, cotangent)[0]


#-----------------------------------------
# Christoffel Symbols
#-----------------------------------------
def a_approx(gamma, old_gamma, gamma_dot, epsilon):
    x = apply_g(gamma, gamma_dot) -  apply_g(old_gamma, gamma_dot)
    return x/epsilon


def b_approx(gamma, old_gamma, gamma_dot, epsilon):
    x1 = jnp.sum(gamma_dot * apply_g(gamma, gamma_dot))
    x2 = jnp.sum(gamma_dot * apply_g(old_gamma, gamma_dot))

    tot = (x1 - x2) / epsilon

    norm_gamma_dot = gamma_dot / jnp.sum(gamma_dot**2)

    return tot * apply_g(gamma, norm_gamma_dot)

@jax.jit
def geodesic_correction_approx(gamma, old_gamma, gamma_dot, epsilon):
    lower = 2 * a_approx(gamma, old_gamma, gamma_dot, epsilon) - b_approx(gamma, old_gamma, gamma_dot, epsilon)*0
    raised = inverse_g(gamma, lower)
    return 0.5*raised


#-------------------------------------------
# Energy and natural gradient
#-------------------------------------------
def calc_E(primals):
    orig_shape = primals.shape
    primals = primals.reshape((-1, 2))

    theta = primals[:,0]
    phi = primals[:,1]


    E = -jnp.sin(theta)*jnp.cos(phi)
    return E.reshape(orig_shape[0:-1])

@jax.jit
def calc_natural_gradient(primals):
    covect = jax.vjp(calc_E, primals)[1](jnp.array(1.))[0]
    return inverse_g(primals, covect)

#--------------------------------------------
# Optimisation schemes
#-------------------------------------------
def optimise_momentum_approx(gamma, N_iter = 150, dt = 0.1, kappa = 1.):
    gamma_dot = -calc_natural_gradient(gamma)

    gammas = [gamma - dt * gamma_dot, gamma]
    for i in range(2, N_iter):
        delta_gam = gammas[i - 1] - gammas[i - 2]
        cor = geodesic_correction_approx(gammas[i - 1], gammas[i - 2], delta_gam/dt, dt)
        natural_grad = calc_natural_gradient(gammas[i - 1])
        gamma = gammas[i-1] - dt**2 * natural_grad +  (1 - kappa * dt) * delta_gam - cor*dt**2
        gammas.append(gamma)

    gammas = jnp.array(gammas)

    return gammas
def optimise_natural_grad(gamma, N_iter = 150, dt = 0.1):
    gammas = [gamma]
    for i in range(1, N_iter):
        natural_grad = calc_natural_gradient(gammas[i - 1])
        gamma = gammas[i-1] - dt * natural_grad
        gammas.append(gamma)

    gammas = jnp.array(gammas)

    return gammas

#--------------------------------------------------
# Generate Data
#--------------------------------------------------
gamma0 = jnp.array([jnp.pi/2 - 0.9,0.4])
dt = 0.1
kappas = jnp.linspace(0.8,2., num = 10)
N = kappas.shape[0]
N_iter = 150

gammass_moment = np.zeros((N, N_iter, 2))
for i, kappa in enumerate(kappas):
    gammas_moment = optimise_momentum_approx(gamma0, N_iter, dt, kappa = kappa)
    gammass_moment[i] = gammas_moment

gammas_nat = optimise_natural_grad(gamma0, N_iter, dt)    

np.save("gammas_momentum_approx_approx", gammass_moment)
np.save("gammas_natural_approx_approx", gammas_nat)
np.save("kappas_approx_approx", kappas)