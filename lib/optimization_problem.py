import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uuid


class OptimizationProblem:
    _objective_function: Callable[[np.ndarray], np.ndarray]
    _inequality_constraints: List[Callable[[np.ndarray], np.ndarray]]
    _equality_constraints: List[Callable[[np.ndarray], np.ndarray]]
    _domain: List[np.ndarray]
    _X: np.ndarray
    _memoized_functions: Dict[str, np.ndarray] = {}
    _g: Callable[[np.ndarray], np.ndarray] = None

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
    def objective_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._objective_function
    
    @property
    def equality_constraints(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return self._equality_constraints
    
    @property
    def inequality_constraints(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return self._inequality_constraints
    
    @property
    def objective_function_values(self) -> np.ndarray:
        return self.compute_function_values(self._objective_function)
    
    @property
    def g(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._g

    def __init__(
        self,
        variables_domain: List[np.ndarray],
        objective_function: Callable[[np.ndarray], np.ndarray],
        inequality_constraints: List[Callable[[np.ndarray], np.ndarray]] = [],
        equality_constraints: List[Callable[[np.ndarray], np.ndarray]] = [],
    ):
        self._objective_function = objective_function
        self._inequality_constraints = inequality_constraints
        self._equality_constraints = equality_constraints
        self._domain = variables_domain
        self._compute_variables()

        self._memoized_functions = {}
        self._objective_function.__name__ = "objective_function"
        for index, constraint in enumerate(self._equality_constraints):
            constraint.__name__ = "$equality_constraint_" + str(index)
        for index, constraint in enumerate(self._inequality_constraints):
            constraint.__name__ = "$inequality_constraint_" + str(index)

        for fn in [self._objective_function] + self._inequality_constraints + self._equality_constraints:
            self._memoized_functions[fn.__name__] = None

    def set_g(self, g: Callable[[np.ndarray], np.ndarray]):
        self._g = g
        
    def _compute_variables(self):
        self._X = np.meshgrid(*self._domain)
    
    def compute_function_values(
            self,
            function: Callable[[np.ndarray], np.ndarray],
            equality_error: float = 0.3,
            force_recompute: bool = False,
        ) -> np.ndarray:
        if self._memoized_functions[function.__name__] is None or force_recompute:
            self._memoized_functions[function.__name__] = function(self.X)
            if "$inequality_constraint" in function.__name__:
                self._memoized_functions[function.__name__] = np.where(
                    self._memoized_functions[function.__name__] <= 0,
                    self._memoized_functions[function.__name__],
                    np.nan
                )
            if "$equality_constraint" in function.__name__:
                self._memoized_functions[function.__name__] = np.where(
                    abs(self._memoized_functions[function.__name__]) <= equality_error,
                    self._memoized_functions[function.__name__],
                    np.nan
                )

        return self._memoized_functions[function.__name__]

    def get_fesable_objective_function(self) -> np.ndarray:
        if "objective_function_constrainted" not in self._memoized_functions:
            Z = self.compute_function_values(self._objective_function)

            for constraint in self._equality_constraints + self._inequality_constraints:
                Z_constraint = self.compute_function_values(constraint)
                Z = np.where(
                    np.isnan(Z_constraint),
                    np.nan,
                    Z
                )
                
            self._memoized_functions["objective_function_constrainted"] = Z

        return self._memoized_functions["objective_function_constrainted"]

    def find_min(self):
        values = self.get_fesable_objective_function()
        if np.isnan(values).all():
            return np.nan, None
        idx = np.unravel_index(np.nanargmin(values, axis=None), values.shape)
        coordinates_values = [x[idx] for x in self._X]
        value = values[idx]
        return value, coordinates_values

    def slater_condition(self) -> bool:
        """
        Check if the Slater condition is satisfied,
        if the inequality constraints are strictly feasible (< 0),
        taking into account the equality constraints to be feasible (== 0)
        """
        pass
    
    def lagrangian(self, lambdas: np.ndarray = np.array([]), mus: np.ndarray = np.array([])) -> np.ndarray:
        """
        Compute the Lagrangian function
        L(x, lambdas, mus) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \sum_{i=1}^p \mu_i g_i(x)
        where h_i(x) <= 0 are the inequality constraints and g_i(x) = 0 are the equality constraints
        """
        # sum of the inequality constraints over number of constraints l_i * (f1(x) + f2(x) + ... + fm(x)) using numpy
        # sum of the equality constraints over number of constraints m_i * (f1(x) + f2(x) + ... + fm(x))
        inequ = np.array([np.zeros(self.objective_function_values.shape)])
        equ = np.array([np.zeros(self.objective_function_values.shape)])

        if len(self.inequality_constraints):
            inequ = np.array([lambdas[i] * self.inequality_constraints[i](self.X) for i in range(len(self.inequality_constraints))])
        
        if len(self.equality_constraints):
            equ = np.array([mus[i] * self.equality_constraints[i](self.X) for i in range(len(self.equality_constraints))])
        
        inequ = np.sum(inequ, axis=0)
        equ = np.sum(equ, axis=0)

        return np.sum(np.array([self.objective_function(self.X), inequ, equ]), axis=0)
    
    def lagrangian_infimum(self, lagrangian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the list of minima of the Lagrangians for each value of lambda and mu
        """
        min_idx = np.nanargmin(lagrangian, axis=1)
        min_values = np.nanmin(lagrangian, axis=1)
        return min_idx, min_values
    
    def lower_bound_property(self, lagrangians) -> Tuple[bool, float, float]:
        """
        Check if the lower bound property is satisfied
        """
        lower_bound = np.nanmin(lagrangians)
        min_value, _ = self.find_min()
        return lower_bound <= self.find_min()[0], lower_bound, min_value
    