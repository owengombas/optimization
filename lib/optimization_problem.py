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
        
    def _compute_variables(self):
        self._X = np.meshgrid(*self._domain)
    
    def compute_function_values(
            self,
            function: Callable[[np.ndarray], np.ndarray],
            equality_error: float = 0.3,
            force_recompute: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    def lagrangian(self, lambas: np.ndarray, mus: np.ndarray) -> np.ndarray:
        """
        Compute the Lagrangian function
        """
        f0 = self.compute_function_values(self._objective_function)
        a = [lambas[i] * self._inequality_constraints[i](self.X) for i in range(len(self._inequality_constraints))]
        b = [mus[i] * self._equality_constraints[i](self.X) for i in range(len(self._equality_constraints))]

        return f0 + a + b
