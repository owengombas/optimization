import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uuid
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

    @domain.setter
    def domain(self, X: np.ndarray):
        self._domain = X
        self._compute_variables()
        self._compute_all_functions()

    @dual_function.setter
    def dual_function(self, dual_function: OptimizationFunction):
        self._dual_function = dual_function

    def __init__(
        self,
        variables_domain: List[np.ndarray],
        objective_function: OptimizationObjectiveFunction,
        inequality_constraints: List[OptimizationInequalityConstraint] = [],
        equality_constraints: List[OptimizationEqualityConstraint] = [],
        dual_function: OptimizationFunction = None,
    ):
        self._objective_function = objective_function
        self._inequality_constraints = inequality_constraints
        self._equality_constraints = equality_constraints
        self._dual_function = dual_function
        self._domain = variables_domain

        self._compute_variables()
        self._compute_all_functions()

    def _compute_variables(self):
        self._X = np.meshgrid(*self._domain)
    
    def _compute_all_functions(self):        
        self.objective_function.evaluate_function(self.X)
        
        for constraint in self._equality_constraints:
            constraint.evaluate_equality_constraint(self.X)
        
        for constraint in self._inequality_constraints:
            constraint.evaluate_inequality_constraint(self.X)

    def get_fesable_objective_function(self) -> np.ndarray:
        if not self.objective_function.is_memorized("function_constrainted"):
            Z = self.objective_function.retrieve_memoized_values()

            for constraint in self._equality_constraints + self._inequality_constraints:
                Z_constraint = constraint.retrieve_constrainted_memorized_values()
                Z = np.where(np.isnan(Z_constraint), np.nan, Z)
                self.objective_function.memorize_values("function_constrainted", Z)
        
            return Z
        else:
            return self.objective_function.retrieve_memoized_values("function_constrainted")

    def find_min(self):
        values = self.get_fesable_objective_function()
        if np.isnan(values).all():
            return np.nan, None
        idx = np.unravel_index(np.nanargmin(values, axis=None), values.shape)
        coordinates_values = [x[idx] for x in self._X]
        value = values[idx]
        return value, coordinates_values

    def lagrangian(
        self, lambdas: np.ndarray = np.array([]), mus: np.ndarray = np.array([]),
        on_feasible_region: bool = False
    ) -> OptimizationFunction:
        """
        Compute the Lagrangian function
        L(x, lambdas, mus) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \sum_{i=1}^p \mu_i g_i(x)
        where h_i(x) <= 0 are the inequality constraints and g_i(x) = 0 are the equality constraints
        """
        # sum of the inequality constraints over number of constraints l_i * (f1(x) + f2(x) + ... + fm(x)) using numpy
        # sum of the equality constraints over number of constraints m_i * (f1(x) + f2(x) + ... + fm(x))

        def lagrangian_fn(_X: np.ndarray):
            return self.objective_function.evaluate_function(self.X) + np.sum(
                [
                    lambdas[i] * (self.inequality_constraints[i].retrieve_constrainted_memorized_values() if on_feasible_region else self.inequality_constraints[i].retrieve_memoized_values())
                    for i in range(len(self.inequality_constraints))
                ],
                axis=0,
            ) + np.sum(
                [
                    mus[i] * (self.equality_constraints[i].retrieve_constrainted_memorized_values() if on_feasible_region else self.equality_constraints[i].retrieve_memoized_values())
                    for i in range(len(self.equality_constraints))
                ],
                axis=0,
            )
        
        return OptimizationFunction(
            lagrangian_fn,
            name=f"lagrangian_{lambdas}_{mus}",
        )

    def lagrangian_infimum(
        self, lagrangians: List[OptimizationFunction]
    ) -> OptimizationFunction:
        """
        Return the list of minima of the Lagrangians for each value of lambda and mu
        """
        def lagrangian_infimum_fn(_X: np.ndarray) -> np.ndarray:
            lagrangians_values = np.array([l.evaluate_function() for l in lagrangians])
            min_idx = np.nanargmin(lagrangians_values, axis=1)
            min_values = np.nanmin(lagrangians_values, axis=1)
            return min_idx, min_values
        
        return OptimizationFunction(
            lagrangian_infimum_fn,
            name=f"lagrangian_infimum",
        )

    def lower_bound_property(self, lagrangians: List[OptimizationFunction]) -> Tuple[bool, float, float]:
        """
        Check if the lower bound property is satisfied
        """
        lagrangians_values = np.array([l.evaluate_function() for l in lagrangians])
        lower_bound = np.nanmin(lagrangians_values)
        min_value, _ = self.find_min()
        return lower_bound <= self.find_min()[0], lower_bound, min_value
