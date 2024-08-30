from typing import Callable, Optional, Any

import abc
from textwrap import dedent

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

import netket as nk
from netket.utils.types import PyTree, Scalar
from netket.utils import warn_deprecation, timing, struct
from netket.vqs import VariationalState


class SRMomentum(struct.Pytree, mutable=True):
    """Base class for a Linear Preconditioner solving a system :math:`Sx = F`.

    A LinearPreconditioner modifies the gradient :math:`F` in such a way that the new
    gradient :math:`x` solves the linear system `:math:`Sx=F`. The linear operator
    :math:`S` is constructed from the variational state.

    To subtype this class and provide a concrete implementation, one needs to define
    at least the function

    .. code::

        @dataclass
        class MyLinearPreconditioner(AbstractLinearPreconditioner):

            def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = 0):
                # here the lhs of the system should be constructed, for example by
                # returning the geometric tensor or any other object
                # return vstate.quantum_geometric_tensor()

    """

    solver: Any = struct.field(serialize=False)
    """Function used to solve the linear system."""

    solver_restart: bool = False
    """If False uses the last solution of the linear system as a starting point for the solution
    of the next."""

    x0: Optional[PyTree] = None
    """Solution of the last linear system solved."""

    info: Any = struct.field(serialize=False, default=None)
    """Additional information returned by the solver when solving the last linear system."""

    old_momentum: Any
    """Old update or initial step"""

    qgt_fun : Any
    """No idea"""

    alpha : float
    """The learning rate"""

    beta : float
    """The momentum parameter"""

    def __init__(self, initial_momentum, alpha = 1e-2, beta = 0.3, solver=nk.optimizer.solver.pinv_smooth, qgt=nk.optimizer.qgt.QGTAuto(), *, solver_restart=False):
        """
        Constructs the structure holding the parameters for using the
        linear preconditioner.

        Args:
            solver: A callable that solves a linear system of equations.
            solver_restart: If False uses the last solution of the linear
                system as a starting point for the solution of the next
                (default=False).
        """
        self.old_momentum = initial_momentum
        self.solver = solver
        self.solver_restart = solver_restart
        self.qgt_fun = qgt
        self.alpha = alpha
        self.beta = beta

    @timing.timed
    def __call__(
        self,
        vstate: VariationalState,
        gradient: PyTree,
        step: Optional[Scalar] = None,
        *args,
        **kwargs,
    ) -> PyTree:
        
        # get current state variables
        parameters = vstate.parameters
        model_state = vstate.model_state
        samples = vstate.samples.reshape(-1, vstate.samples.shape[-1])
        N_samples = samples.shape[0]

        # get old momentum
        old_momentum = self.old_momentum

        # define evaluation functions 
        logpsi = lambda param : vstate._apply_fun({"params":param, **model_state}, samples)
        logpsi_star = lambda param : jnp.conj(logpsi(param))

        # calculate geodesic correction (Christoffel)
        x = jax.jvp(logpsi, (parameters,), (old_momentum,))[1]
        b = x * (x - 2 * jnp.mean(x).real)
        fn = lambda param: jax.jvp(logpsi, (param,), (old_momentum,))[1]
        hess = jax.jvp(fn, (parameters,), (old_momentum,))[1]
        c = (hess + b) - jnp.mean(hess + b)
        geodesic_correction = jax.tree.map(lambda u : u/N_samples, jax.vjp(logpsi_star, parameters)[1](c)[0])

        # calculate stuff to raise index of
        to_raise = tree_util.tree_map(lambda cor, grad: self.beta*cor + self.alpha * grad, geodesic_correction, gradient)

        # raise index
        x0 = self.x0 if self.solver_restart else None
        qgt = self.qgt_fun(vstate)
        self.x0, self.info = qgt.solve(self.solver, to_raise, x0=x0)
        raised = self.x0

        # combine to new update (shouldn't we return minus the new update?)
        result = tree_util.tree_map(lambda old_mom, rais: self.beta * old_mom - rais, old_momentum, raised)
        self.old_momentum = result

        return result


    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n\tsolver          = {self.solver}, "
            + f"\n\tsolver_restart  = {self.solver_restart},"
            + ")"
        )