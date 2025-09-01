import abc
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer


# change
class BadTrustRegionSpec(Exception):
    """Occurs when something about the Trust Region Specification is invalid"""


@dataclass(frozen=True)
class TrustRegionSpec:
    """
    Params:
        - initial_radius: The initial radius of the trust region model
        - max_radius: The maximum allowed radius of the trust region model. A very large value is
            akin to an unlimited radius
        - nabla0: The minimum value the step-size pk must be; else, the step is rejected. Must be
            >= 0, but should be a fairly small value.
        - nabla1: The minimum value for the model to be considered "Good enough" and not prompt a
            reduction in trust region radius
        - nabla2: The minimum value for a model to be "better than good" and to prompt and increase
            in the trust region radius
        - shrink_factor: If the model is not good, this is the factor by which we will reduce the
            trust region radius. Must be >0.0 and <1.0.
        - growth_factor: If the model is very good, this is the factor by which we will increase the
            trust region radius. Must be > 1.0
    """

    initial_radius: float = 1.0
    max_radius: float = 1e4
    nabla0: float = 1e-4
    nabla1: float = 0.25
    nabla2: float = 0.75

    shrink_factor: float = 0.25
    growth_factor: float = 2.0

    trust_region_subproblem_solver: str = "cg"
    trust_region_subproblem_tol: Optional[float] = 1e-4
    trust_region_subproblem_iter: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            if self.initial_radius <= 0.0:
                err_str = f"Initial radius ({self.initial_radius}) must be >0.0!"
                raise BadTrustRegionSpec(err_str)

            if self.max_radius <= 0.0 or self.max_radius < self.initial_radius:
                err_str = (
                    f"Maximum radius ({self.max_radius}) must be >0 and >= the initial radius "
                    f"({self.initial_radius})"
                )

            if self.nabla0 < 0.0:
                raise BadTrustRegionSpec(f"nabla0 ({self.nabla0}) must be >=0!")

            if not (0.0 <= self.nabla0 <= self.nabla1 <= self.nabla2):
                err_str = (
                    "Nabla's must be set s.t. 0.0 <= nabla0 <= nabla1 <= nabla2, are currently: "
                    f"{self.nabla0}, {self.nabla1}, {self.nabla2}"
                )
                raise BadTrustRegionSpec(err_str)
            if not (0.0 < self.shrink_factor < 1.0):
                err_str = f"Shrink factor ({self.shrink_factor}) must be >0 and <1!"
                raise BadTrustRegionSpec(err_str)
            if self.growth_factor <= 1.0:
                err_str = f"Growth factor ({self.growth_factor}) must be >1!"

            if self.trust_region_subproblem_solver == "cg" and (
                self.trust_region_subproblem_tol < 0.0
                or self.trust_region_subproblem_tol is None
            ):
                err_str = (
                    "If the Trust-Region Subproblem solver is Conjugated-Gradient Steihaug, the "
                    f"tolerance ({self.trust_region_subproblem_tol}) must be specified and >= 0.0!"
                )
                raise BadTrustRegionSpec(err_str)
        except TypeError as type_error:
            raise BadTrustRegionSpec("An invalid value was passed in!") from type_error


@dataclass(frozen=True)
class LineSearchSpec:
    max_searches: int = 10
    extrapolation_factor: Optional[float] = 0.5
    sufficient_decrease: float = 0.9
    curvature_constant: Optional[float] = None


"""
Implements several popular Nonlinear Conjugate Gradient Methods
https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
"""


@dataclass
class _LineSearchReturn:
    xk1: Tensor
    fx1: Tensor
    dxk: Tensor
    new_loss: float


class NonlinearConjugateGradient(Optimizer):
    """
    The only difference between several NLGC methods is just the calculation of the value beta.
    This base class codifies this; inheriting classes must provide _beta_calc().

    args:
        - params: The model parameters
        - lr: The Learning rate. If a line-search is being used, this instead becomes the maximum
            step-size search for the backtracking line-search
        - max_newton: The maximum number of "newton" iterations (e.g. steps) that can occur
            per individual "step"
        - rel_newton_tol: the relative change in gradient for convergence at this step
        - abs_newton_tol: the absolute gradient for convergence
        - line_search_spec: A LineSearchSpec object that describes the backtracking line-search
            parameters
        - viz_steps: Used for plotting purposes, will record and return the values for each step.
            Not advisable for real problems, think Rosenbrock or similar
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.3333,
        max_newton: int = 20,
        rel_newton_tol: float = 1.0e-5,
        abs_newton_tol: float = 1.0e-8,
        line_search_spec: Optional[LineSearchSpec] = None,
        viz_steps: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if max_newton < 1:
            raise ValueError(f"Invalid max_newton: {max_newton} - should be >= 1")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Invalid abs_newton_tol: {abs_newton_tol} - should be >= 0.0"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Invalid abs_newton_tol: {rel_newton_tol} - should be >= 0.0"
            )
        if line_search_spec is not None:
            self._validate_line_search_spec(line_search_spec)

        defaults: dict[str, Any] = dict(
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            line_search_spec=line_search_spec,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "The Nonlinear Conjugate Gradient algorithms don't support per-parameter options "
                "(parameter groups)"
            )
        self._params = self.param_groups[0]["params"]
        self.viz_steps = viz_steps

    def _validate_line_search_spec(self, line_search_spec: LineSearchSpec) -> None:
        extrapolation_factor = line_search_spec.extrapolation_factor
        sufficient_decrease = line_search_spec.sufficient_decrease
        curvature_constant = line_search_spec.curvature_constant
        max_searches = line_search_spec.max_searches

        if extrapolation_factor >= 1.0:
            raise ValueError("Extrapolation factor is a multiplier, must be <1.0!")
        if not 0.0 < sufficient_decrease < 1.0:
            raise ValueError("Sufficient decrease must be strictly in (0, 1)!")
        if curvature_constant is not None:
            # Wolfe
            if not 0.0 < curvature_constant < 1.0:
                raise ValueError("Curvature Constant must be strictly in (0, 1)!")
            if curvature_constant <= sufficient_decrease:
                raise ValueError(
                    "Curvature Constant must be greater than sufficient decrease!"
                )
        if max_searches <= 1:
            raise ValueError(
                "If specifying a line search you must have at least one line search!"
            )

    def _get_flat_grad(self) -> Tensor:
        views: List[Tensor] = []
        for param in self._params:
            if param.grad is None:
                view = param.data.new(param.data.numel()).zero_()
            elif param.grad.data.is_sparse:
                view = param.grad.data.to_dense().view(-1)
            else:
                view = param.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        _ = (gradient, last_gradient, last_conjugate_gradient)

        raise NotImplementedError("This is the base class!")

    @staticmethod
    def _convergence_check(
        grad: Tensor,
        rel_newton_tol: float,
        abs_newton_tol: float,
        og_grad_norm: float,
    ) -> bool:
        g = torch.norm(grad).item()
        return g < abs_newton_tol or g < rel_newton_tol * og_grad_norm

    def step(
        self, closure: Callable[[], float]
    ) -> Union[float, Tuple[float, List[Tensor]]]:
        """The optimizer step function."""
        group = self.param_groups[0]
        lr = group["lr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        line_search_spec = group["line_search_spec"]
        if self.viz_steps:
            steps = []

        with torch.no_grad():
            orig_loss = None
            new_loss = None

            def f(y: Tensor) -> Tensor:
                """Wrapper to obtain the loss"""
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

                return loss

            def F(y: Tensor) -> Tensor:
                """Wrapper to obtain the gradient"""
                nonlocal orig_loss
                x = torch.clone(y)
                x.requires_grad = True

                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)

                with torch.enable_grad():
                    orig_loss = closure()

                z = self._get_flat_grad().clone()
                vector_to_parameters(saved_x, self._params)
                self.zero_grad()

                return z

            x0 = parameters_to_vector(self._params).clone()
            gradient = -1.0 * F(x0)
            # print('gradient.norm()', torch.norm(gradient).item())
            p = gradient.clone()
            rho = torch.vdot(gradient, gradient).real

            original_gradient_norm = torch.norm(gradient).item()
            for _ in range(max_newton):
                if self._convergence_check(
                    gradient, rel_newton_tol, abs_newton_tol, original_gradient_norm
                ):
                    # Converged
                    break
                if line_search_spec is None:
                    denom = torch.vdot(p, p).real
                    if denom.item() < 1.0e-16:
                        # Skip this update if the denominator is too small
                        break
                    alpha = torch.div(rho, denom)
                    x0 += lr * alpha * p
                else:
                    line_search_return = self._backtracking_line_search(
                        x0.clone(),
                        gradient.clone(),
                        p.clone(),
                        lr,
                        f,
                        F,
                        line_search_spec,
                    )
                    xk1 = line_search_return.xk1
                    # fx1 = line_search_return.fx1
                    # dx = line_search_return.dxk
                    new_loss = line_search_return.new_loss
                    x0 = xk1

                if self.viz_steps:
                    steps.append(x0)
                last_gradient = gradient.clone()
                gradient = -1.0 * F(x0).clone()

                beta = self._beta_calc(gradient, last_gradient, p)
                # direction reset
                beta = max(beta, 0.0)
                converged = self._convergence_check(
                    gradient, rel_newton_tol, abs_newton_tol, original_gradient_norm
                )
                if beta == 0.0 and converged:
                    # Converged
                    break
                rho = torch.vdot(gradient, gradient).real

                p = gradient + beta * p

            vector_to_parameters(x0, self._params)
            if new_loss is None:
                new_loss = f(x0)

            if self.viz_steps:
                return new_loss, steps

            return new_loss

    def _backtracking_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        d: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        line_search: LineSearchSpec,
    ) -> _LineSearchReturn:
        max_searches = line_search.max_searches
        extrapolation_factor = line_search.extrapolation_factor
        sufficient_decrease = line_search.sufficient_decrease
        curvature_constant = line_search.curvature_constant
        x_orig = x.clone()
        orig_loss = f(x_orig)
        orig_gradient = fx
        orig_curvature = torch.dot(orig_gradient, d).real
        fx1: Optional[Tensor] = None
        new_loss: Optional[float] = None
        for _ in range(max_searches):
            dx = d.mul(lr)
            x_new = x_orig.add(dx)
            new_loss = f(x_new)
            decreased = (
                orig_loss >= new_loss + sufficient_decrease * lr * orig_curvature
            )
            if decreased:
                if curvature_constant is None:
                    xk1 = x_new
                    break
                fx1 = F(x_new)
                new_curvature = torch.dot(fx1, d).real
                curvature = -new_curvature <= -curvature_constant * orig_curvature
                if curvature:
                    xk1 = x_new
                    break

            lr *= extrapolation_factor
        else:
            warnings.warn(f"Maximum number of line searches ({max_searches}) reached!")
            xk1 = x_new
            new_loss = f(xk1)

        return _LineSearchReturn(xk1=xk1, fx1=fx1, dxk=dx, new_loss=new_loss)


class FletcherReeves(NonlinearConjugateGradient):
    """
    Fletcher, R.; Reeves; C, C.M. "Function Minimization by conjugate gradients" (1964)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        _ = last_conjugate_gradient
        num = torch.dot(gradient, gradient)
        den = torch.dot(last_gradient, last_gradient)
        beta = torch.div(num, den)

        return beta


class Daniels(NonlinearConjugateGradient):
    """
    Daniel, James W., "The Conjugate Gradient Method for Linear and Nonlinear Operator Equations"
    (1967)

    Not currently implemented, requires a Hessian-Vector product.
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        # TODO This will require a bit of thought, and likely a change in function signature
        # the update is (gradient^T * Hessian * gradient) / (last_gradient^T * last_hessian * last_gradient)
        # Some way of storing "last numerator"?
        _ = (gradient, last_gradient, last_conjugate_gradient)
        raise NotImplementedError(
            "This method requires a hessian-vector product, which is not currently supported!"
            " Feel free to open a PR if you require it."
        )


class PolakRibiere(NonlinearConjugateGradient):
    """
    Polak, E.; Ribiere, G., "Note sur la convergence de methodes de directions conjuguees" (1969)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        _ = last_conjugate_gradient
        num = torch.vdot(gradient, torch.sub(gradient, last_gradient)).real
        den = torch.vdot(last_gradient, last_gradient).real
        beta = torch.div(num, den)

        return beta


class HestenesStiefel(NonlinearConjugateGradient):
    """
    Hestenes, M.R.; Stiefel, E. "Methods of Conjugate Gradients for Solving Linear Systems" (1952)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        diff = torch.sub(gradient, last_gradient)
        num = torch.dot(gradient, diff).real
        den = torch.dot(-last_conjugate_gradient, diff).real
        beta = torch.div(num, den)

        return beta


class DaiYuan(NonlinearConjugateGradient):
    """
    Dai, Y.-H; Yuan, Y. "A nonlinear conjugate gradient method with strong global convergence
    property" (1999)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        diff = torch.sub(gradient, last_gradient)
        num = torch.dot(gradient, gradient).real
        den = torch.dot(-last_conjugate_gradient, diff).real
        beta = torch.div(num, den)

        return beta


"""
An implementation of a Hessian-free, conjugate residual newton optimizer.
"""


@dataclass
class _LineSearchReturn:
    """
    xk1: the new parameters
    fx1: the new gradient
    dxk: the change in parameters
    new_loss: the new loss
    """

    xk1: Tensor
    fx1: Tensor
    dxk: Tensor
    new_loss: float


class HFCR_Newton(Optimizer):
    """
    Implements the Inexact Newton algorithm with Hessian matrix free conjugate
    residual method.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
                            parameter groups
        lr (float, optional): learning rate (default=0.3333). If line_search_spec is not none,
            this will be the maximum linesearch step (consider setting =1.0)
        max_cr (int, optional): how many conjugate residual iterations to run (default = 10)
        max_newton (int, optional): how many newton iterations to run (default = 10)
        abs_newton_tol (float, optional): absolute tolerance for Newton iteration convergence (default=1.E-3)
        rel_newton_tol (float, optional): relative tolerance for Newton iteration convergence (default=1.E-3)
        cr_tol (float, optional): tolerance for conjugate residual iteration convergence (default=1.E-3)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.3333,
        max_cr: int = 10,
        max_newton: int = 10,
        abs_newton_tol: float = 1.0e-3,
        rel_newton_tol: float = 1.0e-5,
        cr_tol: float = 1.0e-3,
        line_search_spec: Optional[LineSearchSpec] = None,
    ) -> None:
        # ensure inputs are valid
        if lr < 0.0:
            raise ValueError(f"Invalid learnign rate: {lr} - should be >= 0.0")
        if max_cr < 1:
            raise ValueError(f"Invalid max_cr: {max_cr} - should be >= 1")
        if max_newton < 1:
            raise ValueError(f"Invalid max_newton: {max_newton} - should be >= 1")
        if abs_newton_tol <= 0.0:
            raise ValueError(
                f"Invalid Absolute Newton tolerance: {abs_newton_tol} - must be > 0.0!"
            )
        if rel_newton_tol <= 0.0:
            raise ValueError(
                f"Invalid Relative Newton tolerance: {rel_newton_tol} - must be > 0.0!"
            )
        if line_search_spec is not None:
            extrapolation_factor = line_search_spec.extrapolation_factor
            sufficient_decrease = line_search_spec.sufficient_decrease
            curvature_constant = line_search_spec.curvature_constant
            max_searches = line_search_spec.max_searches

            if extrapolation_factor >= 1.0:
                raise ValueError("Extrapolation factor is a multiplier, must be <1.0!")
            if not 0.0 < sufficient_decrease < 1.0:
                raise ValueError("Sufficient decrease must be strictly in (0, 1)!")
            if curvature_constant is not None:
                # Wolfe
                if not 0.0 < curvature_constant < 1.0:
                    raise ValueError("Curvature Constant must be strictly in (0, 1)!")
                if curvature_constant <= sufficient_decrease:
                    raise ValueError(
                        "Curvature Constant must be greater than sufficient decrease!"
                    )
            if max_searches <= 1:
                raise ValueError(
                    "If specifying a line search you must have at least one line search!"
                )

        defaults = dict(
            lr=lr,
            max_cr=max_cr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            cr_tol=cr_tol,
            line_search_spec=line_search_spec,
        )
        super(HFCR_Newton, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HFCR_Newton doesn't support per-parameter options "
                "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]

    def _Hessian_free_product(
        self, grad_x0: Tensor, x0: Tensor, d: Tensor, closure: Callable[[], float]
    ) -> Tensor:
        """
        Computes the Hessian vector product using finite-difference approximation.
        """
        a = torch.norm(d).item()
        eps = 1.0e-6 * torch.div(torch.norm(x0).item(), a)

        x_new = torch.add(x0, d, alpha=eps)

        grad_x_new = self._get_changed_grad(x_new, closure)

        Hv_free = torch.div(1.0, eps) * (grad_x_new - grad_x0) + torch.mul(
            self.lambda_, d
        )

        return Hv_free

    def _Hessian_free_cr(
        self,
        grad_x0: Tensor,
        x0: Tensor,
        dk: Tensor,
        rk: Tensor,
        max_iter: int,
        tol: float,
        closure: Callable[[], float],
    ) -> Tensor:
        """Use conjugate residual for Hessian free."""
        A: Callable[[Tensor], Tensor] = lambda d: self._Hessian_free_product(
            grad_x0, x0, d, closure
        )
        return self._cr(A, dk, rk, max_iter, tol)

    def _cr(
        self,
        A: Callable[[Tensor], Tensor],
        dk: Tensor,
        rk: Tensor,
        max_cr: int,
        tol: float,
    ) -> Tensor:
        """
        The conjugate residual method to solve ``Ax = b``.

        Args:
            A: Operator implementing the Hessian free Hessian vector product Ax
            dk: Initial guess for x
            rk: Vector b in ``Ax = b``
            max_cr: Maximum iterations
            tol: Termination tolerance for convergence
        """
        r: Tensor = rk.clone()
        p: Tensor = r.clone()
        w: Tensor = A(p)
        q: Tensor = w.clone()

        norm0: float = torch.norm(r).item()
        rho_0: float = torch.dot(q, r).item()

        cr_iter: int = 0

        while cr_iter < max_cr:
            cr_iter += 1
            denom: float = torch.dot(w, w).real.item()
            if denom < 1.0e-16:
                break

            alpha: float = torch.div(rho_0, denom)

            dk.add_(p, alpha=alpha)
            r.sub_(w, alpha=alpha)

            res_i_norm: float = torch.norm(r).item()

            if torch.div(res_i_norm, norm0) < tol or cr_iter == (max_cr - 1):
                break

            q = A(r)
            rho_1: float = torch.dot(q, r).real.item()
            beta: float = torch.div(rho_1, rho_0)
            rho_0 = rho_1

            p = r + beta * p
            w = q + beta * w

        return dk

    def _get_flat_grad(self) -> Tensor:
        """Get flattened gradient."""
        views: List[Tensor] = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        """Get gradient at a different point."""
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        closure()
        grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        return grad

    def step(self, closure: Callable[[], float]) -> float:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The new loss value.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr: float = group["lr"]
        max_cr: int = group["max_cr"]
        max_newton: int = group["max_newton"]
        abs_newton_tol: float = group["abs_newton_tol"]
        rel_newton_tol: float = group["rel_newton_tol"]
        cr_tol: float = group["cr_tol"]
        line_search_spec: Optional[LineSearchSpec] = group["line_search_spec"]

        with torch.no_grad():
            x0: Tensor = parameters_to_vector(self._params)
            grad_x0: Tensor = -1.0 * self._get_flat_grad()
            rk: Tensor = grad_x0.clone()
            dk: Tensor = torch.zeros_like(x0)

            original_gradient_norm: float = torch.norm(grad_x0).item()

            for newton_iter in range(max_newton):
                if self._convergence_check(
                    grad_x0, rel_newton_tol, abs_newton_tol, original_gradient_norm
                ):
                    break

                dk = self._Hessian_free_cr(grad_x0, x0, dk, rk, max_cr, cr_tol, closure)

                if line_search_spec is None:
                    x0.add_(dk, alpha=lr)
                else:
                    line_search_return = self._backtracking_line_search(
                        x0,
                        grad_x0,
                        dk,
                        lr,
                        closure,
                        self._get_changed_grad,
                        line_search_spec,
                    )
                    x0 = line_search_return.xk1

                grad_x0 = -1.0 * self._get_flat_grad()
                rk = grad_x0.clone()

            vector_to_parameters(x0, self._params)
            return closure()


"""
A collection of Quasi-Newton methods that rely on matrix-free operators to approximate a
matrix-vector product rather than forming the Hessian explicitly.
"""


class LineSearchWarning(UserWarning):
    """Raise when an error occurs with a Line Search"""


class B0p:
    """
    An expression for the initial Hessian B0 that uses a taylor series approximation
    to determine the Matrix-vector product:
        B0p := (1/mu)(F(x0+mu*p) - F(x0))
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        closure: Callable[[], float],
        mu: float = 1e-6,
    ) -> None:
        self._params = params
        self.closure = closure
        self.mu = mu

    def __call__(self, p: Tensor) -> Tensor:
        Fx = self._get_flat_grad()
        Fx_mu_p = self._get_perturbed_grad(p)
        diff = torch.sub(Fx_mu_p, Fx)
        b0p = torch.div(diff, self.mu)

        return b0p

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_perturbed_grad(self, p: Tensor) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vec = torch.add(current_params, torch.mul(self.mu, p))
        vector_to_parameters(vec, self._params)
        _ = self.closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self._zero_grad()

        return new_flat_grad

    def _zero_grad(self) -> None:
        for p in self._params:
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()


@dataclass
class _LineSearchReturn:
    """
    xk1: the new parameters
    fx1: the new gradient
    dxk: the change in parameters
    new_loss: the new loss
    """

    xk1: Tensor
    fx1: Tensor
    dxk: Tensor
    new_loss: float


class QuasiNewtonWarning(RuntimeWarning):
    """
    Something went wrong in the quasi-newton method that's recoverable
    """


class QuasiNewton(Optimizer):
    """
    A base class for the other quasi-newton methods to inherit from, providing
    common code (as they really only vary by their matrix update methods).
    By direct, we mean solving Bd=-F(x) for d
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 1e-3,
        rel_newton_tol: float = 1e-5,
        krylov_tol: float = 1e-3,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
        mu: float = 1e-6,
        verbose: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning Rate ({lr} must be > 0!")
        if max_newton < 1:
            raise ValueError(f"Max Newton ({max_newton} must be > 0!")
        if max_krylov < 1:
            raise ValueError(f"Max Krylov ({max_krylov} must be > 0!")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Absolute Newton Tolerance ({abs_newton_tol} must be > 0!"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Relative Newton Tolerance ({rel_newton_tol} must be > 0!"
            )
        if krylov_tol < 0.0:
            raise ValueError(f"Krylov tolerance ({krylov_tol} must be > 0!")

        if mu <= 0.0:
            raise ValueError(f"Finite difference size mu ({mu}) for B0p must be >0.0!")

        if matrix_free_memory is not None and matrix_free_memory < 1:
            raise ValueError(
                f"Matrix-free memory size ({matrix_free_memory}) must be None (unlimited) or >0!"
            )

        if line_search is not None:
            if line_search.max_searches < 1:
                raise ValueError(
                    f"Line search max search ({line_search.max_searches}) must be >0!"
                )
            if (
                line_search.extrapolation_factor is not None
                and line_search.extrapolation_factor >= 1.0
            ):
                raise ValueError(
                    f"Extrapolation factor ({line_search.extrapolation_factor}) must be <= 1.0!"
                )
            if not 0.0 < line_search.sufficient_decrease < 1.0:
                raise ValueError(
                    (
                        f"Line search sufficient decrease ({line_search.sufficient_decrease}) must "
                        "be in (0.0, 1.0)!"
                    )
                )
            if (
                line_search.curvature_constant is not None
                and not 0.0 < line_search.curvature_constant < 1.0
            ):
                raise ValueError(
                    (
                        "Line search curvature constant specified ("
                        f"{line_search.curvature_constant}); must be <= 1.0!"
                    )
                )

        defaults = dict(
            lr=lr,
            max_krylov=max_krylov,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            line_search=line_search,
            mu=mu,
        )

        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "The Quasi-Newton methods don't support per-parameter "
                "options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.solver = Solver(max_krylov, krylov_tol)
        self.mf_op = MatrixFreeOperator(lambda p: p, n=matrix_free_memory)
        self.verbose = verbose

    def step(self, closure: Callable[[], float]) -> float:
        group = self.param_groups[0]
        lr = group["lr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        line_search = group["line_search"]

        def b0p(p: Tensor) -> Tensor:
            return p

        try:
            self.mf_op.change_B0p(b0p)
        except AttributeError:
            pass

        def f(y: Tensor) -> Tensor:
            """
            Convenience method for finding the loss at y
            """
            with torch.no_grad():
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

            return loss

        def F(y: Tensor) -> Tensor:
            """
            Convenience method for finding the gradient at y
            F(y) = df(y)/dx
            """
            nonlocal closure
            return self._get_changed_grad(y, closure)

        x0 = parameters_to_vector(self._params)
        fx = F(x0)
        original_gradient_norm = torch.norm(fx).item()

        x = x0.clone()
        d = fx.clone()

        new_loss = None
        fx1 = None

        for _ in range(max_newton):
            if (
                torch.norm(fx).item() <= abs_newton_tol
                or torch.norm(fx).item() <= original_gradient_norm * rel_newton_tol
            ):
                # converged
                break
            d = self.solver(self.mf_op, d, -fx)
            if not torch.isfinite(d).all():
                if self.verbose:
                    msg = (
                        "Solver produced invalid step, assumptions of matrix structure likely "
                        "violated. Resetting matrix free operator and taking gradient step."
                    )
                    print(msg)
                d = -fx.clone()
                self.mf_op.reset()

            if line_search is None:
                dx = torch.mul(lr, d)
                xk1 = torch.add(x, dx)
            elif line_search.curvature_constant is None:
                line_search_return = self._backtracking_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss
            else:
                line_search_return = self._wolfe_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss

            if not torch.isfinite(xk1).all():
                if self.verbose:
                    msg = (
                        "Something broke when stepping. Resetting the MF operator and skipping "
                        "this step. If this is occuring regularly, this may be an unstable "
                        "configuration."
                    )
                    print(msg)
                self.mf_op.reset()
                xk1 = x.clone()

            if fx1 is None:
                fx1 = F(xk1)
            self.mf_op.update(fx, fx1, dx)
            x = xk1
            fx = fx1
            fx1 = None

        vector_to_parameters(x, self._params)
        if new_loss is None:
            new_loss = f(x)

        return new_loss

    def _backtracking_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        d: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        line_search: LineSearchSpec,
    ) -> _LineSearchReturn:
        max_searches = line_search.max_searches
        extrapolation_factor = line_search.extrapolation_factor
        sufficient_decrease = line_search.sufficient_decrease
        curvature_constant = line_search.curvature_constant
        x_orig = x.clone()
        orig_loss = f(x_orig)
        orig_gradient = fx
        orig_curvature = torch.dot(orig_gradient, d).real
        fx1: Optional[Tensor] = None
        new_loss: Optional[float] = None
        for _ in range(max_searches):
            dx = d.mul(lr)
            x_new = x_orig.add(dx)
            new_loss = f(x_new)
            decreased = (
                orig_loss >= new_loss + sufficient_decrease * lr * orig_curvature
            )
            if decreased:
                if curvature_constant is None:
                    xk1 = x_new
                    break
                fx1 = F(x_new)
                new_curvature = torch.dot(fx1, d).real
                curvature = -new_curvature <= -curvature_constant * orig_curvature
                if curvature:
                    xk1 = x_new
                    break

            lr *= extrapolation_factor
        else:
            warnings.warn(f"Maximum number of line searches ({max_searches}) reached!")
            xk1 = x_new
            new_loss = f(xk1)

        return _LineSearchReturn(xk1=xk1, fx1=fx1, dxk=dx, new_loss=new_loss)

    def _wolfe_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        pk: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        spec: LineSearchSpec,
    ) -> _LineSearchReturn:
        """Basically the same as the SciPy implementation"""

        grad = torch.empty_like(pk)

        def phi(alpha: float) -> float:
            return f(x + alpha * pk)

        def dphi(alpha: float) -> float:
            nonlocal grad
            grad = F(x + alpha * pk)
            return torch.dot(grad, pk)

        amax = lr
        c1 = spec.sufficient_decrease
        c2 = spec.curvature_constant
        phi0 = phi(0.0)
        dphi0 = dphi(0.0)
        alpha0 = 0.0
        alpha1 = 1.0
        phi_a1 = phi(alpha1)
        phi_a0 = phi0
        dphi_a0 = dphi0

        for i in range(spec.max_searches):
            if alpha1 == 0 or alpha0 == amax:
                alpha_star = None
                phi_star = phi0
                phi0 = None
                dphi_star = None
                if alpha1 == 0:
                    msg = (
                        "Rounding errors prevent the Wolfe line search from converging"
                    )
                else:
                    msg = (
                        "The line search algorithm could not find a solution <= the learning rate "
                        f"({lr})"
                    )
                warnings.warn(msg, LineSearchWarning)
                break

            not_first_iteration = i > 0
            if (phi_a1 > phi0 + c1 * alpha1 * dphi0) or (
                (phi_a1 >= phi_a0) and not_first_iteration
            ):
                alpha_star, phi_star, dphi = self._zoom(
                    alpha0,
                    alpha1,
                    phi_a0,
                    phi_a1,
                    dphi_a0,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            dphi_a1 = dphi(alpha1)
            if abs(dphi_a1) <= -c2 * dphi0:
                alpha_star = alpha1
                phi_star = phi_a1
                dphi_star = dphi_a1
                break

            if dphi_a1 >= 0:
                alpha_star, phi_star, dphi_star = self._zoom(
                    alpha1,
                    alpha0,
                    phi_a1,
                    phi_a0,
                    dphi_a1,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            alpha2 = 2 * alpha1
            alpha2 = min(alpha2, amax)
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi(alpha1)
            dphi_a0 = dphi_a1

        else:
            alpha_star = alpha1
            phi_star = phi_a1
            # dphi_star = None
            warnings.warn(
                "The Wolfe Line Search algorithm did not converge", LineSearchWarning
            )

        if alpha_star is None:
            alpha_star = 0.0
        delta_x = alpha_star * pk
        xk1 = x + delta_x

        retval = _LineSearchReturn(xk1=xk1, fx1=grad, dxk=delta_x, new_loss=phi_star)
        return retval

    def _zoom(
        self,
        a_lo: float,
        a_hi: float,
        phi_lo: float,
        phi_hi: float,
        derphi_lo: float,
        phi: Callable[[float], float],
        derphi: Callable[[float], float],
        phi0: float,
        derphi0: float,
        c1: float,
        c2: float,
    ) -> Tuple[float, float, float]:
        """
        Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
        Taken from SciPy's Optimization toolbox

        Notes
        -----
        Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
        'Numerical Optimization', 1999, pp. 61.

        """

        maxiter = 10
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0
        a_rec = 0
        while True:
            dalpha = a_hi - a_lo
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            if i > 0:
                cchk = delta1 * dalpha
                a_j = self._cubicmin(
                    a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                )
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j
            phi_aj = phi(a_j)
            if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj

            else:
                derphi_aj = derphi(a_j)
                if abs(derphi_aj) <= -c2 * derphi0:
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
                    break

                if derphi_aj * (a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo

                else:
                    phi_rec = phi_lo
                    a_rec = a_lo

                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj

            i += 1
            if i > maxiter:
                # Failed to find a conforming step size
                a_star = None
                val_star = None
                valprime_star = None
                break

        return a_star, val_star, valprime_star

    def _cubicmin(
        self, a: float, fa: float, fpa: float, b: float, fb: float, c: float, fc: float
    ) -> float:
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

        If no minimizer can be found, return None.

        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        device = self._params[0].device
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = torch.empty((2, 2)).to(device)
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            [A, B] = torch.matmul(
                d1,
                torch.tensor([fb - fa - C * db, fc - fa - C * dc]).flatten().to(device),
            )
            A = A / denom
            B = B / denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + torch.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _quadmin(self, a: float, fa: float, fpa: float, b: float, fb: float) -> float:
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.

        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)

        with torch.set_grad_enabled(True):
            _ = closure()
            new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad


class QuasiNewtonTrust(Optimizer):
    """
    A base class for the other quasi-newton methods to inherit from, providing
    common code (as they really only vary by their matrix update methods).
    By direct, we mean solving Bd=-F(x) for d
    This appears to work for the SR1 Dual, even though the dual of the matrix is also the inverse.
    Neat.

    Uses a Trust Region method, instead of a fixed step size or line-search
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        trust_region: TrustRegionSpec,
        lr: float = 1.0,
        max_newton: int = 10,
        abs_newton_tol: float = 1e-3,
        rel_newton_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        mu: float = 1e-6,
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning Rate ({lr} must be > 0!")
        if max_newton < 1:
            raise ValueError(f"Max Newton ({max_newton} must be > 0!")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Absolute Newton Tolerance ({abs_newton_tol} must be > 0!"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Relative Newton Tolerance ({rel_newton_tol} must be > 0!"
            )
        if mu <= 0.0:
            raise ValueError(f"Finite difference size mu ({mu}) for B0p must be >0.0!")

        if matrix_free_memory is not None and matrix_free_memory < 1:
            raise ValueError(
                f"Matrix-free memory size ({matrix_free_memory}) must be None (unlimited) or >0!"
            )

        defaults = dict(
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            trust_region=trust_region,
            mu=mu,
        )

        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "The Quasi-Newton methods don't support per-parameter "
                "options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.mf_op = MatrixFreeOperator(lambda p: p, n=matrix_free_memory)
        self.trust_region_spec = trust_region
        if self.trust_region_spec.trust_region_subproblem_solver == "cauchy":
            self._trust_region_subproblem = CauchyPoint
        elif self.trust_region_spec.trust_region_subproblem_solver == "dogleg":
            self._trust_region_subproblem = Dogleg
        elif self.trust_region_spec.trust_region_subproblem_solver == "cg":
            self._trust_region_subproblem = ConjugateGradientSteihaug
        else:
            raise BadTrustRegionSpec(
                "Invalid trust-region subproblem solver requested: "
                f"{self.trust_region_spec.trust_region_subproblem_solver}"
            )

        self._trust_region_radius = self.trust_region_spec.initial_radius

    def step(self, closure: Callable[[], float]) -> float:
        group = self.param_groups[0]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]

        # b0p = B0p(self._params, closure)
        def b0p(p: Tensor) -> Tensor:
            return p

        try:
            self.mf_op.change_B0p(b0p)
        except AttributeError:
            pass

        def f(y: Tensor) -> Tensor:
            """
            Convenience method for finding the loss at y
            """
            with torch.no_grad():
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

            return loss

        def F(y: Tensor) -> Tensor:
            """
            Convenience method for finding the gradient at y
            F(y) = df(y)/dx
            """
            nonlocal closure
            return self._get_changed_grad(y, closure)

        x0 = parameters_to_vector(self._params)
        x = x0.clone()
        fx = F(x0)
        original_gradient_norm = torch.norm(fx).item()
        m = self._trust_region_subproblem(
            x, f, F, self.mf_op, self.trust_region_spec.trust_region_subproblem_iter
        )
        newton_iter = 0

        # We use while loops to allow for a repeated iteration in the event of a bad model
        while newton_iter < max_newton:
            newton_iter += 1
            if (
                torch.norm(fx).item() <= abs_newton_tol
                or torch.norm(fx).item() <= original_gradient_norm * rel_newton_tol
            ):
                # converged
                break
            p, hits_boundary = m.solve(self._trust_region_radius)
            predicted_value = m(p)
            x_proposed = x + p
            m_proposed = self._trust_region_subproblem(
                x_proposed,
                f,
                F,
                self.mf_op,
                self.trust_region_spec.trust_region_subproblem_iter,
            )
            actual_reduction = m.fun - m_proposed.fun
            predicted_reduction = m.fun - predicted_value
            if predicted_reduction < 0.0:
                warnings.warn("Predicted improvement was negative!")
                break
            rho = actual_reduction / predicted_reduction
            if rho < self.trust_region_spec.nabla1:
                self._trust_region_radius *= self.trust_region_spec.shrink_factor
            elif rho > 0.75 and hits_boundary:
                self._trust_region_radius = min(
                    self.trust_region_spec.growth_factor * self._trust_region_radius,
                    self.trust_region_spec.max_radius,
                )

            if rho > self.trust_region_spec.nabla0:
                # accept the step
                self.mf_op.update(
                    m.grad.clone(), m_proposed.grad.clone(), (x_proposed - x).clone()
                )
                x = x_proposed.clone()
                m = m_proposed

        vector_to_parameters(x, self._params)
        new_loss = m.fun

        return new_loss

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad


class InverseQuasiNewton(Optimizer):
    """
    A base class for the inverse quasi-newton methods to inherit from, providing
    common code (as they really only vary by their matrix update methods).
    By inverse, we mean they rely on the direct calculation of the inverse matrix H=B^{-1}
    to find d = -HF(x)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        max_newton: int = 10,
        abs_newton_tol: float = 1e-3,
        rel_newton_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
        trust_region: Optional[TrustRegionSpec] = None,
        verbose: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Learning Rate ({lr} must be > 0!")
        if max_newton < 1:
            raise ValueError(f"Max Newton ({max_newton} must be > 0!")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Absolute Newton Tolerance ({abs_newton_tol} must be > 0!"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Relative Newton Tolerance ({rel_newton_tol} must be > 0!"
            )
        if matrix_free_memory is not None and matrix_free_memory < 1:
            raise ValueError(
                f"Matrix-free memory size ({matrix_free_memory}) must be None (unlimited) or >0!"
            )

        # Line search and Trust Region validation
        if line_search is not None and trust_region is not None:
            raise ValueError(
                "Line search and trust region are mutually exclusive; "
                "please only provide one specification!"
            )
        if line_search is not None:
            if line_search.max_searches < 1:
                raise ValueError(
                    f"Line search max search ({line_search.max_searches}) must be >0!"
                )
            if (
                line_search.extrapolation_factor is not None
                and line_search.extrapolation_factor >= 1.0
            ):
                raise ValueError(
                    f"Extrapolation factor ({line_search.extrapolation_factor}) must be <= 1.0!"
                )
            if not 0.0 < line_search.sufficient_decrease < 1.0:
                raise ValueError(
                    (
                        f"Line search sufficient decrease ({line_search.sufficient_decrease}) must "
                        "be in (0.0, 1.0)!"
                    )
                )
            if (
                line_search.curvature_constant is not None
                and not 0.0 < line_search.curvature_constant < 1.0
            ):
                raise ValueError(
                    (
                        "Line search curvature constant specified ("
                        f"{line_search.curvature_constant}); must be <= 1.0!"
                    )
                )

        if trust_region is not None:
            raise NotImplementedError("Trust region methods not yet supported!")

        defaults = dict(
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            line_search=line_search,
            trust_region=trust_region,
        )

        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "The Quasi-Newton methods don't support per-parameter "
                "options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.mf_op = InverseMatrixFreeOperator(n=matrix_free_memory)
        self.verbose = verbose

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        group = self.param_groups[0]
        lr = group["lr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        line_search = group["line_search"]

        new_loss = None

        def f(y: Tensor) -> Tensor:
            """Convenience method to get the loss"""
            with torch.no_grad():
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

            return loss

        def F(y: Tensor) -> Tensor:
            """Convenience method to get the gradient"""
            x = torch.clone(y)
            saved_x = parameters_to_vector(self._params)
            vector_to_parameters(x, self._params)
            with torch.set_grad_enabled(True):
                _ = closure()
            z = torch.clone(self._get_flat_grad())
            vector_to_parameters(saved_x, self._params)
            self.zero_grad()

            return z

        x0 = parameters_to_vector(self._params)
        fx = F(x0)
        original_gradient_norm = torch.norm(fx).item()

        x = x0.clone()
        d = fx.clone()

        new_loss = None
        fx1 = None

        for _ in range(max_newton):
            if (
                torch.norm(fx).item() <= abs_newton_tol
                or torch.norm(fx).item() <= original_gradient_norm * rel_newton_tol
            ):
                # converged
                break
            d = -(self.mf_op * fx)
            if not torch.isfinite(d).all():
                if self.verbose:
                    msg = (
                        "Matrix-Free operator produced an invalid step, assumptions of matrix "
                        "structure likely violated.  Resetting matrix free operator and taking "
                        "gradient step."
                    )
                    print(msg)
                    print(f"Currently at {len(self.mf_op.memory)} entries.")
                self.mf_op.reset()
                d = -fx.clone()

            if line_search is None:
                dx = torch.mul(lr, d)
                xk1 = torch.add(x, dx)
                fx1 = F(xk1)
            elif line_search.curvature_constant is None:
                line_search_return = self._backtracking_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss
            else:
                line_search_return = self._wolfe_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss

            if not torch.isfinite(xk1).all():
                if self.verbose:
                    msg = (
                        "Something broke when stepping. Resetting the MF operator and skipping this "
                        "step. If this is occuring regularly, this may be an unstable configuration."
                    )
                    print(msg)
                self.mf_op.reset()
                xk1 = x.clone()

            if fx1 is None:
                fx1 = F(xk1)
            self.mf_op.update(fx, fx1, dx)
            x = xk1
            fx = fx1

        vector_to_parameters(x, self._params)
        if new_loss is None:
            new_loss = f(x)

        return new_loss

    def _backtracking_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        d: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        line_search: LineSearchSpec,
    ) -> _LineSearchReturn:
        max_searches = line_search.max_searches
        extrapolation_factor = line_search.extrapolation_factor
        sufficient_decrease = line_search.sufficient_decrease
        curvature_constant = line_search.curvature_constant

        x_orig = x.clone()
        orig_loss = f(x_orig)
        orig_gradient = fx
        orig_curvature = torch.dot(orig_gradient, d).real
        fx1: Optional[Tensor] = None
        new_loss: Optional[float] = None

        for _ in range(max_searches):
            dx = d.mul(lr)
            x_new = x_orig.add(dx)
            new_loss = f(x_new)
            decreased = (
                new_loss <= orig_loss + sufficient_decrease * lr * orig_curvature
            )
            if decreased:
                if curvature_constant is None:
                    xk1 = x_new
                    break
                fx1 = F(x_new)
                new_curvature = torch.dot(fx1, d).real
                curvature = -new_curvature <= -curvature_constant * orig_curvature
                if curvature:
                    xk1 = x_new
                    break

            lr *= extrapolation_factor
        else:
            warnings.warn(f"Maximum number of line searches ({max_searches}) reached!")
            xk1 = x_new

        return _LineSearchReturn(xk1=xk1, fx1=fx1, dxk=dx, new_loss=new_loss)

    def _wolfe_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        pk: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        spec: LineSearchSpec,
    ) -> _LineSearchReturn:
        """Basically the same as the SciPy implementation"""

        grad = torch.empty_like(pk)

        def phi(alpha: float) -> float:
            return f(x + alpha * pk)

        def dphi(alpha: float) -> float:
            nonlocal grad
            grad = F(x + alpha * pk)
            return torch.dot(grad, pk)

        amax = lr
        c1 = spec.sufficient_decrease
        c2 = spec.curvature_constant
        phi0 = phi(0.0)
        dphi0 = dphi(0.0)
        alpha0 = 0.0
        alpha1 = 1.0
        phi_a1 = phi(alpha1)
        phi_a0 = phi0
        dphi_a0 = dphi0

        for i in range(spec.max_searches):
            if alpha1 == 0 or alpha0 == amax:
                alpha_star = None
                phi_star = phi0
                phi0 = None
                dphi_star = None
                if alpha1 == 0:
                    msg = (
                        "Rounding errors prevent the Wolfe line search from converging"
                    )
                else:
                    msg = (
                        "The line search algorithm could not find a solution <= the learning rate "
                        f"({lr})"
                    )
                warnings.warn(msg, LineSearchWarning)
                break

            not_first_iteration = i > 0
            if (phi_a1 > phi0 + c1 * alpha1 * dphi0) or (
                (phi_a1 >= phi_a0) and not_first_iteration
            ):
                alpha_star, phi_star, dphi = self._zoom(
                    alpha0,
                    alpha1,
                    phi_a0,
                    phi_a1,
                    dphi_a0,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            dphi_a1 = dphi(alpha1)
            if abs(dphi_a1) <= -c2 * dphi0:
                alpha_star = alpha1
                phi_star = phi_a1
                dphi_star = dphi_a1
                break

            if dphi_a1 >= 0:
                alpha_star, phi_star, dphi_star = self._zoom(
                    alpha1,
                    alpha0,
                    phi_a1,
                    phi_a0,
                    dphi_a1,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            alpha2 = 2 * alpha1
            alpha2 = min(alpha2, amax)
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi(alpha1)
            dphi_a0 = dphi_a1

        else:
            alpha_star = alpha1
            phi_star = phi_a1
            # dphi_star = None
            warnings.warn(
                "The Wolfe Line Search algorithm did not converge", LineSearchWarning
            )

        if alpha_star is None:
            alpha_star = 0.0
        delta_x = alpha_star * pk
        xk1 = x + delta_x

        retval = _LineSearchReturn(xk1=xk1, fx1=grad, dxk=delta_x, new_loss=phi_star)
        return retval

    def _zoom(
        self,
        a_lo: float,
        a_hi: float,
        phi_lo: float,
        phi_hi: float,
        derphi_lo: float,
        phi: Callable[[float], float],
        derphi: Callable[[float], float],
        phi0: float,
        derphi0: float,
        c1: float,
        c2: float,
    ) -> Tuple[float, float, float]:
        """
        Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
        Taken from SciPy's Optimization toolbox

        Notes
        -----
        Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
        'Numerical Optimization', 1999, pp. 61.

        """

        maxiter = 10
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0
        a_rec = 0
        while True:
            dalpha = a_hi - a_lo
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            if i > 0:
                cchk = delta1 * dalpha
                a_j = self._cubicmin(
                    a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                )
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j
            phi_aj = phi(a_j)
            if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj

            else:
                derphi_aj = derphi(a_j)
                if abs(derphi_aj) <= -c2 * derphi0:
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
                    break

                if derphi_aj * (a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo

                else:
                    phi_rec = phi_lo
                    a_rec = a_lo

                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj

            i += 1
            if i > maxiter:
                # Failed to find a conforming step size
                a_star = None
                val_star = None
                valprime_star = None
                break

        return a_star, val_star, valprime_star

    def _cubicmin(
        self, a: float, fa: float, fpa: float, b: float, fb: float, c: float, fc: float
    ) -> float:
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

        If no minimizer can be found, return None.

        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        device = self._params[0].device
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = torch.empty((2, 2)).to(device)
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            [A, B] = torch.matmul(
                d1,
                torch.tensor([fb - fa - C * db, fc - fa - C * dc]).flatten().to(device),
            )
            A = A / denom
            B = B / denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + torch.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _quadmin(self, a: float, fa: float, fpa: float, b: float, fb: float) -> float:
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.

        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad


class SymmetricRankOneQuasiNewton(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Symmetric_rank-one
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = SymmetricRankOneMatrixFree(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateResidual(max_krylov, krylov_tol)


class SymmetricRankOneInverseQuasiNewton(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Symmetric_rank-one
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
            trust_region=trust_region,
        )
        self.mf_op = SymmetricRankOneInverseMatrixFree(matrix_free_memory)


class SymmetricRankOneTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = SymmetricRankOneMatrixFree(lambda p: p, n=matrix_free_memory)


class SymmetricRankOneDualTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        # SR1 dual is the inverse
        self.mf_op = SymmetricRankOneInverseMatrixFree(matrix_free_memory)


class BFGS(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 0.001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = BFGS(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class BFGSInverse(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = BFGSInverse(matrix_free_memory)


class BFGSTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = BFGS(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class DavidonFletcherPowellQuasiNewton(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 0.001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = DavidonFletcherPowellQuasiNewton(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class DavidonFletcherPowellInverseQuasiNewton(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = DavidonFletcherPowellInverseMatrixFree(matrix_free_memory)


class DavidonFletcherPowellTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = DavidonFletcherPowellQuasiNewton(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class Broyden(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Brodyden%27s_method
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = Broyden(lambda p: p, n=matrix_free_memory)
        self.solver = None
        err_msg = (
            "Need to implement a solver that handles non-symmetric, non-PD solver "
            "such as GMRES, Arnoldi, or GCR."
        )
        raise NotImplementedError(err_msg)


class BrodyenInverse(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = BroydenInverseMatrixFree(matrix_free_memory)


class BroydenTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = Broyden(lambda p: p, n=matrix_free_memory)
        self.solver = None
        err_msg = (
            "Need to implement a solver that handles non-symmetric, non-PD solver "
            "such as GMRES, Arnoldi, or GCR."
        )
        raise NotImplementedError(err_msg)


"""
A Data Class for specifying Line Search behaviors
"""


"""
A collection of Matrix-Free operators that act as Hessians in Quasi-Newton
methods.
"""


# A constant used to check if the denominator is too small for an update
_R = 1e-8


class NonCommutativeOperatorError(RuntimeError):
    """
    A custom exception for the Matrix Free operators to indicate that
    "left" multiplication (p^TA vs. Ap) is not defined.
    """


class MatrixFreeOperator:
    """
    A matrix free operator for direct problems
    args:
        - B0p: A callable that accepts a tensor and returns a tensor.
            Alternatively, update this later if required with the change_B0p method
        - n: The size of the memory. Setting it to None will make this a
            full-memory model; an integer retains the n most recent updates
    """

    def __init__(
        self, B0p: Callable[[Tensor], Tensor], n: Optional[int] = None
    ) -> None:
        self.B0p = B0p
        if n is None:
            self.memory = []
        else:
            self.memory = deque(maxlen=n)

    def change_B0p(self, B0p: Callable[[Tensor, Callable[[], float]], Tensor]) -> None:
        """
        A hack, basically...B0p relies on a closure, but that closure
        is handed in to the "step" method of the optimizer. Either we need
        to significantly redesign the interface...or we'll just use this.
        """
        self.B0p = B0p

    def reset(self) -> None:
        """
        Convenience function to reset the state of the operator
        """
        self.memory.clear()

    def update(
        self, grad_fk_: Tensor, grad_fk_plus_one_: Tensor, delta_x_: Tensor
    ) -> None:
        """
        Given relevant vectors, update the memory of this object
        """
        raise NotImplementedError("Update method must be provided in base class!")

    def multiply(self, p: Tensor) -> Tensor:
        """
        Implement the actual matrix vector multiplication
        """
        raise NotImplementedError("Multiply method must be provided in base class!")

    def __mul__(self, p: Tensor) -> Tensor:
        return self.multiply(p)

    def __rmul__(self, p: Tensor) -> None:
        raise NonCommutativeOperatorError(
            "Matrix free methods don't commute! In the case of something like q^T*A*p, try q^T(A*p)"
        )

    def construct_full_matrix(
        self,
        m: int,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        These are all matrix-vector approximations. We can "rehydrate" the underlying matrix by
        multiplying it times the standard basis vectors and "stacking" the results.

        Note that this is really for debugging/exploratory purposes, and very impractical
        or impossible for large systems (recall the O(n^2) memory requirements!).
        """
        H = torch.zeros((m, m), dtype=dtype, device=device)
        mat_device = self.memory[0][0].device
        for i in range(m):
            basis_i = torch.zeros((m,), dtype=dtype)
            basis_i[i] = 1.0
            basis_i = basis_i.to(mat_device)
            col = self.multiply(basis_i).to(device)
            H[:, i] = col

        return H

    def determinant(
        self,
        m: int,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Construct the full matrix and calculate its determinant
        """
        mat = self.construct_full_matrix(m, dtype=dtype, device=device)
        det = torch.linalg.det(mat)

        return det


class InverseMatrixFreeOperator:
    """
    A matrix free operator for inverse problems
    """

    def __init__(self, n: Optional[int] = None) -> None:
        if n is None:
            self.memory = []
        else:
            self.memory = deque(maxlen=n)

    def reset(self) -> None:
        """
        Convenience function to reset the state of the operator
        """
        self.memory.clear()

    def update(
        self, grad_fk_: Tensor, grad_fk_plus_one_: Tensor, delta_x_: Tensor
    ) -> None:
        """
        Given relevant vectors, update the memory of this object
        """
        raise NotImplementedError("Update method must be provided in base class!")

    def multiply(self, p: Tensor) -> Tensor:
        """
        Implement the actual matrix vector multiplication
        """
        raise NotImplementedError("Multiply method must be provided in base class!")

    def __mul__(self, p: Tensor) -> Tensor:
        return self.multiply(p)

    def __rmul__(self, p: Tensor) -> None:
        raise NonCommutativeOperatorError("Matrix free methods don't commute!")


class SymmetricRankOneMatrixFree(MatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_f_k = grad_f_k_.detach().clone()
        grad_f_k_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_f_k_plus_one - grad_f_k
        Bk_delta_x = self.multiply(delta_x)
        vk = yk - Bk_delta_x
        den = torch.dot(vk, delta_x)
        if torch.abs(delta_x @ vk) < 1e-6 * torch.norm(delta_x) * torch.norm(vk):
            # Don't apply the update
            return

        self.memory.append((vk.clone(), den.clone()))

    def multiply(self, p: Tensor) -> Tensor:
        Bp = self.B0p(p.clone())
        for vk, den in self.memory:
            Bp += (vk * (vk @ p)) / den

        return Bp


class SymmetricRankOneInverseMatrixFree(InverseMatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_fk_plus_one - grad_fk
        vk = delta_x - self.multiply(yk)
        den = torch.dot(vk, yk)
        self.memory.append((vk, den))

    def multiply(self, p: Tensor) -> Tensor:
        """
        Assumes Hk0 = I
        """
        # TODO evaulate alternate Hk0 choices (Nocedal & Wright have a suggestion)
        Hkp = p.clone()
        for vk, den in self.memory:
            Hkp += vk * torch.dot(vk, p) / den

        return Hkp


class BFGSMatrixFree(MatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        B_delta_x = self.multiply(delta_x)
        den1 = torch.dot(y, delta_x)
        den2 = torch.dot(delta_x, B_delta_x)
        self.memory.append((y, B_delta_x, den1, den2))

    def multiply(self, p: Tensor) -> Tensor:
        Bp = self.B0p(p.clone())
        for y, B_delta_x, den1, den2 in self.memory:
            Bp += ((y * torch.dot(y, p)) / den1) + (
                (B_delta_x * torch.dot(B_delta_x, p)) / den2
            )

        return Bp


class BFGSInverseMatrixFree(InverseMatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        Hy = self.multiply(y)
        den = torch.dot(y, delta_x)
        self.memory.append((y, Hy, delta_x, den))

    def multiply(self, p: Tensor) -> Tensor:
        Hp = p.clone()
        for y, Hy, delta_x, den in self.memory:
            term1 = -(delta_x * torch.dot(y, Hp)) / den
            term2 = -(Hy * torch.dot(delta_x, p)) / den
            term3 = (delta_x * torch.dot(y, Hy) * torch.dot(delta_x, p)) / (den**2)
            term4 = (delta_x * torch.dot(delta_x, p)) / den

            Hp += term1 + term2 + term3 + term4

        return Hp


class DavidonFletcherPowellMatrixFree(MatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        B_delta_x = self.multiply(delta_x)
        self.memory.append((B_delta_x, delta_x, y))

    def multiply(self, p: Tensor) -> Tensor:
        Bp = self.B0p(p.clone())
        for B_delta_x, delta_x, y in self.memory:
            den = torch.dot(y, delta_x)
            yTp = torch.dot(y, p)
            B_deltax_yTp = torch.mul(B_delta_x, yTp)

            num2 = torch.mul(y, torch.dot(delta_x, Bp))
            num3 = torch.mul(y, yTp)
            num4 = torch.dot(y, torch.mul(delta_x, B_deltax_yTp))

            num1_sum = torch.add(torch.add(B_deltax_yTp, num2), num3)
            term1 = torch.div(num1_sum, den)
            term2 = torch.div(num4, torch.mul(den, den))
            Bp = torch.add(Bp, torch.add(term1, term2))

        return Bp


class DavidonFletcherPowellInverseMatrixFree(InverseMatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        Hy = self.multiply(y)
        den1 = torch.dot(delta_x, y)
        den2 = torch.dot(y, Hy)
        self.memory.append((y, Hy, delta_x, den1, den2))

    def multiply(self, p: Tensor) -> Tensor:
        Hkp = p.clone()
        for y, Hy, delta_x, den1, den2 in self.memory:
            Hkp += (delta_x * torch.dot(delta_x, p) / den1) - (
                Hy * torch.dot(y, Hkp) / den2
            )

        return Hkp


class BroydenMatrixFree(MatrixFreeOperator):
    """
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_fk_plus_one - grad_fk
        vk = yk - self.multiply(delta_x)
        left = vk / torch.dot(delta_x, delta_x)
        self.memory.append((left, delta_x))

    def multiply(self, p: Tensor) -> Tensor:
        Bkp = self.B0p(p.clone())
        for left_k, delta_xk in self.memory:
            Bkp += left_k * torch.dot(delta_xk, p)

        return Bkp


class BroydenInverseMatrixFree(InverseMatrixFreeOperator):
    """
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_fk_plus_one - grad_fk
        vk = delta_x - self.multiply(yk)
        den = torch.dot(delta_x, self.multiply(yk))
        self.memory.append((vk, delta_x, den))

    def multiply(self, p: Tensor) -> Tensor:
        Hkp = p.clone()
        for vk, delta_x, den in self.memory:
            Hkp += vk * torch.dot(delta_x, Hkp) / den

        return Hkp


"""
Mostly for debugging, full matrix operators.
"""


class MatrixOperator(abc.ABC):
    """
    Base class to create Hessians (matrix operators) for use in the
    """

    def __init__(self, B0: Tensor, n: Optional[int] = None) -> None:
        self.B0 = B0
        self.matrix = torch.clone(B0)
        self.n = n
        if self.n is not None:
            err = "Limited memory matrix operators not implemented, using full history!"
            warn(err, RuntimeWarning)

    def reset(self) -> None:
        self.matrix = torch.clone(self.B0)

    def multiply(self, x: Tensor) -> Tensor:
        return torch.matmul(self.matrix, x)

    def __mul__(self, x: Tensor) -> Tensor:
        return self.multiply(x)

    def __rmul__(self, x: Tensor) -> None:
        return torch.matmul(x, self.matrix)

    @abc.abstractmethod
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        """
        the quasi-newton update
        """


class BroydenMatrix(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        self.matrix += torch.outer(
            (yk - self.multiply(delta_x)) / torch.dot(delta_x, delta_x), delta_x
        )


class BroydenInverseMatrix(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        num = (delta_x - self.multiply(yk)) @ delta_x @ self.matrix
        den = delta_x @ self.matrix @ yk
        self.matrix += num / den


class SymmetricRankOneMatrix(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        Bk_delta_x = self.multiply(delta_x)
        vk = yk - Bk_delta_x
        num = torch.outer(vk, vk)
        den = torch.dot(vk, delta_x)
        if torch.abs(delta_x @ vk) < 1e-6 * torch.norm(delta_x) * torch.norm(vk):
            # Don't apply the update
            return

        self.matrix += num / den


class SymmetricRankOneInverseMatrix(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        tmp = delta_x - torch.matmul(self.matrix, yk)
        num = torch.matmul(tmp, tmp.T)
        den = torch.dot(tmp, yk)

        self.matrix += num / den


class DavidonFletcherPowellMatrix(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        dim = grad_f_k.shape[0]
        Identity = torch.eye(dim)
        num1 = torch.matmul(yk, delta_x.T)
        num2 = torch.matmul(delta_x, yk.T)
        den = torch.dot(yk, delta_x)
        left = Identity - num1 / den
        right = Identity - num2 / den
        product = torch.matmul(torch.matmul(left, self.matrix), right)
        summand = torch.matmul(yk, yk.T) / den

        self.matrix += product + summand


class DavidonFletcherPowellInverseMatrix(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        num1 = torch.matmul(delta_x, delta_x.T)
        den1 = torch.dot(delta_x, yk)
        num2 = torch.matmul(
            torch.matmul(torch.matmul(self.matrix, yk), yk.T), self.matrix
        )
        den2 = torch.matmul(torch.matmul(yk.T, self.matrix), yk)

        self.matrix += num1 / den1 - num2 / den2


"""
A collection of linear system solvers for use in our optimization routines.
"""


class Solver(abc.ABC):
    """
    An iterative solver for problems of the form Ax=b where A may be a matrix-free
    operator.

    args:
        - max_iter: The maximum iterations the solver can attempt per call
        - tolerance: the tolerance for determining convergence
    """

    def __init__(self, max_iter: int, tolerance: float) -> None:
        self.max_iter = max_iter
        self.tol = tolerance

    def __call__(self, A: MatrixFreeOperator, x: Tensor, b: Tensor) -> Tensor:
        return self.solve(A, x, b)

    @abc.abstractmethod
    def solve(self, A: MatrixFreeOperator, x0: Tensor, b: Tensor) -> Tensor:
        """
        The base solve function.
        args:
            A: A matrix or matrix-free operator that supports matrix multiplicaton
            x0, b: the vectors to solve for.
        """


class ConjugateGradient(Solver):
    def solve(self, A: MatrixFreeOperator, x0: Tensor, b: Tensor) -> Tensor:
        rk = b - (A * x0)
        if torch.norm(rk).item() <= self.tol:
            return x0

        xk = x0.clone()
        pk = rk.clone()
        rk_inner = torch.dot(rk, rk)
        for _ in range(self.max_iter):
            Apk = A * pk
            alpha = torch.div(rk_inner, torch.dot(pk, Apk))
            xk = torch.add(xk, pk, alpha=alpha)
            rk = torch.sub(rk, Apk, alpha=alpha)

            rk_inner_new = torch.dot(rk, rk)
            if rk_inner_new.item() <= self.tol:
                return xk

            beta = torch.div(rk_inner_new, rk_inner)
            pk = torch.add(rk, pk, alpha=beta)
            rk_inner = rk_inner_new

        return xk


class ConjugateResidual(Solver):
    def solve(self, A: MatrixFreeOperator, x0: Tensor, b: Tensor) -> Tensor:
        rk = b - A * x0
        xk = x0
        if torch.norm(rk) <= self.tol:
            return x0
        pk = torch.clone(rk)
        Apk = A * pk
        for _ in range(self.max_iter):
            alpha = torch.div(torch.dot(rk, A * rk), torch.dot(Apk, Apk))
            xk1 = torch.add(xk, pk, alpha=alpha)
            rk1 = torch.sub(rk, Apk, alpha=alpha)
            if torch.norm(rk1) <= self.tol:
                xk = xk1
                break
            beta = torch.div(torch.dot(rk1, A * rk1), torch.dot(rk, A * rk))
            pk1 = torch.add(rk1, pk, alpha=beta)
            Apk = torch.add(A * rk1, Apk, alpha=beta)
            xk = xk1
            pk = pk1
            rk = rk1

        return xk


"""
A Data Class for specifying Trust Region behaviors
"""


class QuadraticSubproblem:
    """
    Taken from Scipy. A Function object representing the quadratic subproblem, with
    neat lazy evaluation techniques for the coeff values
    Modified to work in our case
    """

    # TODO typehints
    def __init__(
        self,
        x: Tensor,
        loss: Callable[[Tensor], float],
        grad: Callable[[Tensor], Tensor],
        hess: Tensor,
        max_iter: int,
    ) -> None:
        self._x = x
        self._f = None
        self._g = None
        self._h = hess
        self._g_mag = None

        self._fun = loss
        self._grad = grad
        self._max_iter = max_iter
        if self._max_iter <= 0:
            raise ValueError(
                "Maximum iterations in trust region subproblem must be >= 1"
            )

    def __call__(self, p: Tensor) -> float:
        return self.fun + torch.dot(self.grad, p) + 0.5 * torch.dot(p, self.hess * p)

    @property
    def fun(self) -> float:
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def grad(self) -> Tensor:
        if self._g is None:
            self._g = self._grad(self._x)
        return self._g

    @property
    def hess(self) -> Tensor:
        # This won't be none, we're doing this differently
        return self._h

    @property
    def grad_mag(self) -> float:
        if self._g_mag is None:
            self._g_mag = torch.linalg.norm(self.grad)
        return self._g_mag

    def get_boundaries_intersections(
        self, z: Tensor, d: Tensor, trust_radius: float
    ) -> List[float]:
        """
        Solve ||z+t*d|| == trust_radius
        return both values of t, sorted from low to high
        """
        a = torch.dot(d, d)
        b = 2 * torch.dot(z, d)
        c = torch.dot(z, z) - trust_radius**2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        aux = b + torch.copysign(sqrt_discriminant, b)
        ta = -aux / (2 * a)
        tb = -2 * c / aux
        return sorted([ta, tb])

    def solve(self, trust_radius: float) -> Tuple[Tensor, bool]:
        raise NotImplementedError(
            "The solve method must be provided by an inheriting class!"
        )


class ConjugateGradientSteihaug(QuadraticSubproblem):
    """
    The Conjugate Gradient Steihaug method. Very similar to the normal CG method,
    but handles the case where the dBd <= 0 by instead assuming a linear model
    """

    def solve(self, trust_radius: float) -> Tuple[Tensor, bool]:
        # get the norm of jacobian and define the origin
        p_origin = torch.zeros_like(self.grad)

        # define a default tolerance
        tolerance = min(0.5, torch.sqrt(self.grad_mag)) * self.grad_mag

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.grad_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.grad
        d = -r

        # Search for the min of the approximation of the objective function.
        for _ in range(self._max_iter):
            # do an iteration
            Bd = self.hess * d
            dBd = torch.dot(d, Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = torch.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if torch.norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = torch.dot(r_next, r_next)
            if torch.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next
        if torch.norm(z) >= trust_radius:
            # Find t >= 0 to get the boundary point such that
            # ||z + t d|| == trust_radius
            ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
            p_boundary = z + tb * d
            hits_boundary = True
            return p_boundary, hits_boundary
        return (z_next, False)


class CauchyPoint(QuadraticSubproblem):
    def solve(self, trust_radius: float) -> Tuple[Tensor, bool]:
        _ = trust_radius
        raise NotImplementedError("Haven't implemented Cauchy-Point yet")


class Dogleg(QuadraticSubproblem):
    def solve(self, trust_radius: float) -> Tuple[Tensor, bool]:
        _ = trust_radius
        raise NotImplementedError("Haven't implemented Dogleg yet")


"""
Implements several variants of the Levenberg algorithm.
    - LevenbergEveryStep adjusts the value of lambda with each step of the Newton iteration
    - LevenbergEndStep takes several steps and then adjusts lambda, retracting all if needed.
    - LevenbergBatch requires an additional call to adjust lambda after all batches have been
      used for that training epoch.
"""


# TODO Eventually write this using inheritance...
class LevenbergEveryStep(Optimizer):
    """
    Implements the Levenberg method using a quasi-newton method and Hessian Free
    conjugate residual solver. Adjusts lambda per newton step.
        lr: The learning rate. Must be >0.0
        lambda0: The initial value of lambda, or the influence of gradient vs. Hessian. Must be >0.0
        max_lambda: The largest value lambda can scale to. Must be >0.0 and > min_lambda
        min_lambda: The smallest value lambda can scale to. Must be >0.0 and < max_lambda
        nu: The multiplier used to scale lambda as the optimizer steps improve or fail to improve
            the loss. Must be >1.0
        max_cr: The maximum number of conjugate residual iterations that can be taken when solving
            Ax=b. Must be >=1.
        max_newton: The maximum number of newton iterations that can be taken for this batch. Must
            be >=1
        cr_tol: The tolerance to consider the conjugate residual as converged, prompting early
            return. Must be >=0.0
        newton_tol: The tolerance to consider the newton iteration as converged, prompting early
            return. Must be >=0.0
        debug: Whether to track the values for lambda and loss as the algorithm progresses. If
            enabled, the step() method will return orig loss as well as these values in a tuple.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.3333,
        lambda0: float = 1.0,
        max_lambda: float = 1000.0,
        min_lambda: float = 1e-6,
        nu: float = 2.0**0.5,
        max_cr: int = 10,
        max_newton: int = 1,
        newton_tol: float = 1.0e-3,
        cr_tol: float = 1.0e-3,
        debug: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if lambda0 <= 0.0:
            raise ValueError(f"Invalid lambda0: {lambda0} - should be > 0.0")
        if max_lambda <= 0.0:
            raise ValueError(f"Invalid max_lambda: {max_lambda} - should be > 0.0")
        if min_lambda <= 0.0:
            raise ValueError(f"Invalid min_lambda: {min_lambda} - should be > 0.0")
        if max_lambda <= min_lambda:
            raise ValueError(
                f"Invalid max and min lambda: {max_lambda}, {min_lambda} - max_lambda should be > min_lambda"
            )
        if nu <= 1.0:
            raise ValueError(f"Invalid nu: {nu} - should be > 1.0")
        if max_cr < 1:
            raise ValueError(f"Invalid max_cr: {max_cr} - should be >= 1")
        if cr_tol < 0.0:
            raise ValueError(f"Invalid cr_tol: {cr_tol} - should be >= 0.0")
        if max_newton < 1:
            raise ValueError(f"Invalid max_newton: {max_newton} - should be >= 1")
        if newton_tol < 0.0:
            raise ValueError(f"Invalid newton_tol: {newton_tol} - should be >= 0.0")

        defaults = dict(
            lr=lr,
            lambda_=lambda0,
            max_lambda=max_lambda,
            min_lambda=min_lambda,
            nu=nu,
            max_cr=max_cr,
            max_newton=max_newton,
            newton_tol=newton_tol,
            cr_tol=cr_tol,
            debug=debug,
        )

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HFCR_Newton doesn't support per-parameter options "
                "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.lambda_ = lambda0

    def _Hessian_free_product(
        self,
        grad_x0: Tensor,
        x0: Tensor,
        d: Tensor,
        lambda_: float,
        closure: Callable[[], float],
    ) -> Tensor:
        """
        Computes the Hessian vector product by using a finite-difference approximation
        Hd = (1 / eps) * delta_f(x + eps * d) - delta_f(x), where H is the Hessian,
        d is the vector which has the same dimension as x, delta_f is the first order
        derivative, eps is a small scalar.

        Arguments:
        grad_x0 (torch.tensor): flatten tensor, denotes the flat grad of x0
        x0 (torch.tensor): flatten tensor, denotes the flat parameters
        d(torch.tensor): flatten tensor.

        Return:
        Hv_free (torch.tensor): Flat tensor
        """
        # calculate eps
        a = torch.norm(d).item()
        eps = 1.0e-6 * torch.div(torch.norm(x0).item(), a)

        x_new = torch.add(x0, d, alpha=eps)

        grad_x_new = self._get_changed_grad(x_new, closure)

        Hv_free = torch.div(1.0, eps) * (grad_x_new - grad_x0) + torch.mul(lambda_, d)

        return Hv_free

    def _Hessian_free_cr(
        self,
        grad_x0: Tensor,
        x0: Tensor,
        dk: Tensor,
        rk: Tensor,
        lambda_: float,
        max_iter: int,
        tol: float,
        closure: Callable[[], float],
    ) -> Tensor:
        """
        Use conjugate residual for Hessian free.
        """

        def A(d: Tensor) -> Tensor:
            return self._Hessian_free_product(grad_x0, x0, d, lambda_, closure)

        return self._cr(A, dk, rk, max_iter, tol)

    @staticmethod
    def _cr(
        A: Callable[[Tensor], Tensor], dk: Tensor, rk: Tensor, max_cr: int, tol: float
    ) -> Tensor:
        """
        The conjugate residual method method to solve ``Ax = b``, where A is
        required to be symmetric.

        Arguments:
            A (callable): An operator implementing the Hessian free
                Hessian vector product Ax.
            dk (torch.Tensor): An initial guess for x.
            rk (torch.Tensor): The vector b in ``Ax = b``.
            max_cr (int): maximum iterations.
            tol (float, optional): Termination tolerance for convergence.

        Return:
            dk (torch.Tensor): The approximation x in ``Ax = b``.
        """

        r = rk.clone()
        p = r.clone()
        w = A(p)
        q = w.clone()

        norm0 = torch.norm(r).item()
        rho_0 = torch.dot(q, r).item()

        cr_iter = 0

        while cr_iter < max_cr:
            cr_iter += 1
            denom = torch.dot(w, w).real.item()
            if denom < 1.0e-16:
                break

            alpha = torch.div(rho_0, denom)

            dk.add_(p, alpha=alpha)
            r.sub_(w, alpha=alpha)

            res_i_norm = torch.norm(r).item()

            if torch.div(res_i_norm, norm0) < tol or cr_iter == (max_cr - 1):
                break

            q = A(r)
            rho_1 = torch.dot(q, r).item()

            if abs(rho_1) < 1.0e-16:
                break

            beta = torch.div(rho_1, rho_0)
            rho_0 = rho_1
            p.mul_(beta).add_(r)
            w.mul_(beta).add_(q)

        return dk

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        """
        Calculate the gradient of model parameters given the new parameters.
        Note that we are not really changing model parameters at this moment.

        Argument:
        vec (torch.tensor): a flatten tensor, the new model parameters.
        closure: used to re-evaluate model.

        Return:
        new_flat_grad (torch.tensor): a flatten tensor, the gradient of such
                                      new parameters.
        """
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad

    def step(self, closure: Callable[[], float]) -> float:
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        assert len(self.param_groups) == 1

        group = self.param_groups[0]

        lr = group["lr"]
        min_lambda = group["min_lambda"]
        max_lambda = group["max_lambda"]
        nu = group["nu"]
        max_cr = group["max_cr"]
        max_newton = group["max_newton"]
        newton_tol = group["newton_tol"]
        cr_tol = group["cr_tol"]
        debug = group["debug"]

        # evaluate initial
        orig_loss = closure()

        # Completely original params, if multiple newton steps are taken
        x0 = parameters_to_vector(self._params)
        # "Current" params for the newton step
        x1 = x0.clone()

        rk = self._get_flat_grad().neg()

        res_norm_1 = torch.norm(rk).item()
        res_norm_0 = torch.norm(rk).item()

        n_iter = 0
        # Insurance so each epoch will at least try 1 iteration
        if self.lambda_ < min_lambda:
            self.lambda_ *= nu
        if self.lambda_ > max_lambda:
            self.lambda_ /= nu

        if debug:
            losses = []

        while (
            n_iter < max_newton
            and torch.div(res_norm_1, res_norm_0) > newton_tol
            and min_lambda < self.lambda_ < max_lambda
        ):
            if debug:
                iter_losses = []
            n_iter += 1

            dk = torch.zeros_like(rk)

            grad_xk = self._get_changed_grad(x1, closure)

            # Hessian free Conjugate Residual
            dk = self._Hessian_free_cr(
                grad_xk, x1, dk, rk, self.lambda_, max_cr, cr_tol, closure
            )

            # Update parameters
            x1.add_(dk, alpha=lr)

            vector_to_parameters(x1, self._params)
            new_loss = closure()
            vector_to_parameters(x0, self._params)
            if debug:
                iter_losses.append(new_loss.item())

            if new_loss < orig_loss:
                # Decrease the direct influence of the gradient
                self.lambda_ /= nu
                # update grad based on new parameters
                rk = self._get_changed_grad(x1, closure).neg()
                res_norm_1 = torch.norm(rk).item()
                x0 = x1.clone()
                if debug:
                    losses.append(iter_losses)

            else:
                # Retract the step
                n_iter -= 1
                x1 = x0.clone()
                # Increase the direct influence of the gradient
                self.lambda_ *= nu

        # set parameters
        vector_to_parameters(x0, self._params)
        # if debug:
        #     return (
        #         orig_loss,
        #         lambdas,
        #         losses,
        #     )

        return orig_loss
