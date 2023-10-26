from __future__ import annotations
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uuid


class OptimizationFunction:
    _function: Callable[[np.ndarray], np.ndarray]
    _gradient: "OptimizationFunction"
    _hessian: "OptimizationFunction"
    _name: str
    _memoized_values: Dict[str, np.ndarray]
    _expression: str

    @property
    def function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._function

    @property
    def gradient(self) -> "OptimizationFunction":
        return self._gradient

    @property
    def hessian(self) -> "OptimizationFunction":
        return self._hessian

    @property
    def name(self) -> str:
        return self._name

    @property
    def expression(self) -> str:
        return self._expression

    @function.setter
    def function(self, function: Callable[[np.ndarray], np.ndarray]):
        self._function = function

    @gradient.setter
    def gradient(self, gradient: "OptimizationFunction"):
        self._gradient = gradient

    @hessian.setter
    def hessian(self, hessian: "OptimizationFunction"):
        self._hessian = hessian

    @name.setter
    def name(self, name: str):
        self._name = name

    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        gradient: "OptimizationFunction" = None,
        hessian: "OptimizationFunction" = None,
        name: str = "",
        expression: str = "",
    ):
        self._function = function
        self._gradient = gradient
        self._hessian = hessian
        self._name = name
        self._memoized_values = {}
        self._expression = expression

    def _call_and_save(
        self,
        x: np.ndarray,
        function: Callable[[np.ndarray], np.ndarray],
        condition: Callable[[np.ndarray], np.ndarray],
        memoized_id: str,
    ) -> np.ndarray:
        values = function(x)

        if condition is not None:
            values = np.where(condition(values), values, np.nan)
        
        if memoized_id is not None:
            if memoized_id not in self._memoized_values:
                self._memoized_values[memoized_id] = values
            
        return values

    def evaluate_function(
        self,
        x: np.ndarray = np.array([]),
        memoized_id: str = "function",
        condition: Callable[[np.ndarray], np.ndarray] = None,
    ) -> np.ndarray:
        return self._call_and_save(
            x=x,
            function=self.function,
            memoized_id=memoized_id,
            condition=condition,
        )

    def evaluate_gradient(
        self,
        x: np.ndarray,
        memoized_id: str = "gradient",
        condition: Callable[[np.ndarray], np.ndarray] = None,
    ) -> np.ndarray:
        if self.gradient is None:
            raise Exception("Gradient is not defined")
        return self._gradient.evaluate_function(
            x, memoized_id=memoized_id, condition=condition
        )

    def evaluate_hessian(
        self,
        x: np.ndarray,
        memoized_id: str = "hessian",
        condition: Callable[[np.ndarray], np.ndarray] = None,
    ) -> np.ndarray:
        if self.hessian is None:
            raise Exception("Hessian is not defined")
        return self._hessian.evaluate_function(
            x, memoized_id=memoized_id, condition=condition
        )

    def find_min(self, memoized_id: str = "function") -> Tuple[int, float]:
        if "min" not in self._memoized_values:
            index_min = np.nanargmin(self._memoized_values[memoized_id])
            min_value = self._memoized_values[memoized_id][index_min]
            self._memoized_values["min"] = (index_min, min_value)
        return self._memoized_values["min"]

    def find_max(self, memoized_id: str = "function") -> Tuple[int, float]:
        if "max" not in self._memoized_values:
            index_max = np.nanargmax(self._memoized_values[memoized_id])
            max_value = self._memoized_values[memoized_id][index_max]
            self._memoized_values["max"] = (index_max, max_value)
        return self._memoized_values["max"]

    def reset_memoized_values(self):
        self._memoized_values = {}

    def retrieve_memoized_values(self, memoized_id: str = "function") -> np.ndarray:
        return self._memoized_values[memoized_id]

    def memorize_values(self, memoized_id: str, values: np.ndarray):
        self._memoized_values[memoized_id] = values

    def is_memorized(self, memoized_id: str) -> bool:
        return memoized_id in self._memoized_values

    def __str__(self):
        return f"{self.name} = {self.expression}"

    def __repr__(self):
        return self.__str__()


class OptimizationObjectiveFunction(OptimizationFunction):
    pass


class OptimizationEqualityConstraint(OptimizationFunction):
    def evaluate_equality_constraint(
        self,
        x: np.ndarray,
        error: float = 0.3,
        recompute: bool = False,
        memoized_id: str = "function_equality_constraint",
    ) -> np.ndarray:
        if memoized_id not in self._memoized_values or recompute:
            all_values = self._function(x)
            self._memoized_values["function"] = all_values
            self._memoized_values[memoized_id] = np.where(
                np.abs(all_values) <= error, all_values, np.nan
            )
        return self._memoized_values[memoized_id]

    def retrieve_constrainted_memorized_values(self) -> np.ndarray:
        return self._memoized_values["function_equality_constraint"]


class OptimizationInequalityConstraint(OptimizationFunction):
    def evaluate_inequality_constraint(
        self,
        x: np.ndarray,
        recompute: bool = False,
        memoized_id: str = "function_inequality_constraint",
    ) -> np.ndarray:
        if memoized_id not in self._memoized_values or recompute:
            all_values = self._function(x)
            self._memoized_values["function"] = all_values
            self._memoized_values[memoized_id] = np.where(
                all_values <= 0, all_values, np.nan
            )
        return self._memoized_values[memoized_id]

    def retrieve_constrainted_memorized_values(self) -> np.ndarray:
        return self._memoized_values["function_inequality_constraint"]
