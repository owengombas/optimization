import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from lib.optimization_function import (
    OptimizationFunction,
    OptimizationObjectiveFunction,
    OptimizationEqualityConstraint,
    OptimizationInequalityConstraint,
)


class OptimizationProblem:
    _objective_function: OptimizationObjectiveFunction
    _inequality_constraints: List[OptimizationInequalityConstraint]
    _equality_constraints: List[OptimizationEqualityConstraint]
    _dual_function: OptimizationFunction = None
    _X: np.ndarray
    _domain: List[np.ndarray]
    _tolerance: float

    @property
    def domain(self):
        return self._domain

    @property
    def X(self):
        return self._X

    @property
    def dim(self):
        return len(self._domain) + 1

    @property
    def objective_function(self) -> OptimizationFunction:
        return self._objective_function

    @property
    def equality_constraints(self) -> List[OptimizationEqualityConstraint]:
        return self._equality_constraints

    @property
    def inequality_constraints(self) -> List[OptimizationInequalityConstraint]:
        return self._inequality_constraints

    @property
    def dual_function(self) -> OptimizationFunction:
        return self._dual_function

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @domain.setter
    def domain(self, domain: List[np.ndarray]):
        self._domain = domain
        self._reset()

    @dual_function.setter
    def dual_function(self, dual_function: OptimizationFunction):
        self._dual_function = dual_function

    @tolerance.setter
    def tolerance(self, tolerance: float):
        self._tolerance = tolerance
        self._reset()

    def __init__(
        self,
        variables_domain: List[np.ndarray],
        objective_function: OptimizationObjectiveFunction,
        inequality_constraints: List[OptimizationInequalityConstraint] = [],
        equality_constraints: List[OptimizationEqualityConstraint] = [],
        dual_function: OptimizationFunction = None,
        tolerance: float = 1e-6,
    ):
        self._objective_function = objective_function
        self._inequality_constraints = inequality_constraints
        self._equality_constraints = equality_constraints
        self._dual_function = dual_function
        self._domain = variables_domain
        self._tolerance = tolerance

        self._reset()

    def _reset(self):
        self.objective_function.forget_memoized("function_constrainted")
        self._compute_variables()
        self._compute_all_functions()
        self.compute_all_gradients()

    def _compute_variables(self):
        self._X = np.meshgrid(*self._domain)

    def _compute_all_functions(self):
        self.objective_function.evaluate_function(self.X, recompute=True)

        for constraint in self._equality_constraints:
            constraint.evaluate_equality_constraint(
                self.X, tolerance=self.tolerance, recompute=True
            )

        for constraint in self._inequality_constraints:
            constraint.evaluate_inequality_constraint(self.X, recompute=True)

    def compute_all_gradients(self):
        self.objective_function.gradient.evaluate_function(self.X, recompute=True)

        for constraint in self._equality_constraints:
            constraint.gradient.evaluate_function(self.X, recompute=True)

        for constraint in self._inequality_constraints:
            constraint.gradient.evaluate_function(self.X, recompute=True)

    def get_fesable_objective_function(self, memoize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.objective_function.is_memoized("function_constrainted") or not memoize:
            Z = self.objective_function.retrieve_memoized_values()
            Z_feasable = self.objective_function.retrieve_memoized_feasable_set()

            for constraint in self._equality_constraints + self._inequality_constraints:
                Z_constraint = constraint.retrieve_constrainted_memorized_values()
                Z = np.where(np.isnan(Z_constraint), np.nan, Z)
                Z_feasable = np.where(np.isnan(Z_constraint), np.nan, Z_feasable)

            if memoize:
                self.objective_function.memoize_function(
                    "function_constrainted", Z, Z_feasable
                )
            else:
                return Z, Z_feasable

        return self.objective_function.retrieve_memoized_values(
            "function_constrainted"
        ), self.objective_function.retrieve_memoized_feasable_set(
            "function_constrainted"
        )

    def find_min(self):
        values, _ = self.get_fesable_objective_function()
        if np.isnan(values).all():
            return np.nan, None
        idx = np.unravel_index(np.nanargmin(values, axis=None), values.shape)
        coordinates_values = [x[idx] for x in self._X]
        value = values[idx]
        return value, coordinates_values

    def lagrangians(
        self,
        lambdas_and_mus: np.ndarray,
    ) -> Tuple[List[OptimizationFunction], List[np.ndarray], List[np.ndarray]]:
        lagrangians: List[OptimizationFunction] = []

        for i in range(len(lambdas_and_mus)):
            lambdas = lambdas_and_mus[i, : len(self.inequality_constraints)]
            mus = lambdas_and_mus[i, len(self.inequality_constraints) :]
            lagrangians.append(
                self.lagrangian(
                    lambdas=lambdas,
                    mus=mus,
                )
            )

        return lagrangians

    def lagrangian(
        self,
        lambdas: np.ndarray = np.array([]),
        mus: np.ndarray = np.array([]),
    ) -> OptimizationFunction:
        """
        Compute the Lagrangian function
        L(x, lambdas, mus) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \sum_{i=1}^p \mu_i g_i(x)
        where h_i(x) <= 0 are the inequality constraints and g_i(x) = 0 are the equality constraints
        """
        # sum of the inequality constraints over number of constraints l_i * (f1(x) + f2(x) + ... + fm(x)) using numpy
        # sum of the equality constraints over number of constraints m_i * (f1(x) + f2(x) + ... + fm(x))

        def lagrangian_fn(_X: np.ndarray) -> np.ndarray:
            return (
                self.objective_function.evaluate_function(_X, memoized_id=None)[0]
                + np.sum(
                    [
                        lambdas[i]
                        * (
                            self.inequality_constraints[
                                i
                            ].evaluate_function(_X, memoized_id=None)[0]
                        )
                        for i in range(len(self.inequality_constraints))
                    ],
                    axis=0,
                )
                + np.sum(
                    [
                        mus[i]
                        * (
                            self.equality_constraints[
                                i
                            ].evaluate_function(_X, memoized_id=None)[0]
                        )
                        for i in range(len(self.equality_constraints))
                    ],
                    axis=0,
                )
            )

        def lagrangian_gradient(_X: np.ndarray) -> np.ndarray:
            return (
                self.objective_function.gradient.evaluate_function(_X, memoized_id=None, recompute=True)[0]
                + np.sum(
                    [
                        lambdas[i]
                        * (
                            self.inequality_constraints[
                                i
                            ].gradient.evaluate_function(_X, memoized_id=None, recompute=True)[0]
                        )
                        for i in range(len(self.inequality_constraints))
                    ],
                    axis=0,
                )
                + np.sum(
                    [
                        mus[i]
                        * (
                            self.equality_constraints[
                                i
                            ].gradient.evaluate_function(_X, memoized_id=None, recompute=True)[0]
                        )
                        for i in range(len(self.equality_constraints))
                    ],
                    axis=0,
                )
            )

        lagrangian = OptimizationFunction(
            lagrangian_fn,
            name=f"L({lambdas}, {mus})",
            gradient=OptimizationFunction(
                lagrangian_gradient,
                name=f"∇L({lambdas}, {mus})",
            ),
        )

        return lagrangian

    def lagrangian_infimum(
        self, lagrangians: List[OptimizationFunction]
    ) -> OptimizationFunction:
        """
        Return the list of minima of the Lagrangians for each value of lambda and mu
        """

        def lagrangian_infimum_fn(_X: np.ndarray) -> np.ndarray:
            lagrangians_values = np.array(
                [l.evaluate_function(_X)[0] for l in lagrangians]
            )
            min_idx = np.nanargmin(lagrangians_values, axis=1)
            min_values = np.nanmin(lagrangians_values, axis=1)
            return min_idx, min_values

        return OptimizationFunction(
            lagrangian_infimum_fn,
            name=f"lagrangian_infimum",
        )

    def lower_bound_property(
        self, lagrangians: List[OptimizationFunction]
    ) -> Tuple[bool, float, float]:
        """
        Check if the lower bound property is satisfied
        """
        lagrangians_values = np.array([l.evaluate_function()[0] for l in lagrangians])
        lower_bound = np.nanmin(lagrangians_values)
        min_value, _ = self.find_min()
        return lower_bound <= self.find_min()[0], lower_bound, min_value

    def kkt(
        self,
        lagrangian: OptimizationFunction,
        lambdas_and_mus: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[bool, np.ndarray]:
        # Primal feasibility
        x_feasible_primal = self.get_fesable_objective_function(memoize=False)[1]
        x_feasible_primal = x_feasible_primal[:, ~np.isnan(x_feasible_primal).any(axis=0)]

        # Complemetarity slackness
        x_feasible_slackness = x_feasible_primal.copy()
        for i, constraint in enumerate(self.inequality_constraints):
            x_feasible_slackness = np.where(
                np.abs(
                    lambdas_and_mus[i]
                    * constraint.evaluate_function(x_feasible_primal, memoized_id=None)[0]
                ) > tolerance,
                np.nan,
                x_feasible_slackness,
            )

        x_feasible_slackness = x_feasible_slackness[:, ~np.isnan(x_feasible_slackness).any(axis=0)]

        # Vanishing gradient
        values, _ = lagrangian.gradient.evaluate_function(x_feasible_slackness, memoized_id=None)
        vectors = values.T
        norms = [np.linalg.norm(v) for v in vectors]
        x_feasible_gradient = x_feasible_slackness[:, np.array(norms) < tolerance]

        return x_feasible_gradient.T

    def kkts(
        self,
        lagrangians: List[OptimizationFunction],
        lambdas_and_mus: np.ndarray,
        tolerance: float = 1e-6,
    ) -> List[Tuple[np.ndarray, bool, np.ndarray]]:
        values = [
            (lambdas_and_mus[i], self.kkt(lagrangians[i], lambdas_and_mus[i], tolerance))
            for i in range(len(lagrangians))
        ]
        values = filter(lambda x: len(x[1]) > 0, values)
        return list(values)

    def cartesian_product(arrays: List[np.ndarray]):
        return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))
