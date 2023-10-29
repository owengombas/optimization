from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union


class OptimizationFunction:
    _function: Callable[[np.ndarray], np.ndarray]
    _gradient: "OptimizationFunction"
    _hessian: "OptimizationFunction"
    _name: str
    _memoized_values: Dict[str, Dict[str, np.ndarray]]
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
        self._feasable_set = None

    def _call_and_save(
        self,
        x: np.ndarray,
        function: Callable[[np.ndarray], np.ndarray],
        memoized_id: str,
        condition: Callable[[np.ndarray], np.ndarray] = None,
        recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if memoized_id is not None and self.is_memoized(memoized_id) and not recompute:
            return (
                self.retrieve_memoized_values(memoized_id),
                self.retrieve_memoized_feasable_set(memoized_id),
            )

        values = function(x)
        feasible_set = x

        if condition is not None:
            feasible_set = np.where(condition(values), x, np.nan)
            values = np.where(condition(values), values, np.nan)

        if not self.is_memoized(memoized_id) or recompute:
            self.memoize_function(memoized_id, values, feasible_set)

        return values, feasible_set

    def evaluate_function(
        self,
        x: np.ndarray = np.array([]),
        memoized_id: str = "function",
        condition: Callable[[np.ndarray], np.ndarray] = None,
        recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._call_and_save(
            x=x,
            function=self.function,
            memoized_id=memoized_id,
            condition=condition,
            recompute=recompute,
        )

    def find_min(self, memoized_id: str = "function") -> Tuple[int, float]:
        index_min = np.nanargmin(self._memoized_values[memoized_id])
        min_value = self._memoized_values[memoized_id][index_min]
        return index_min, min_value

    def find_max(self, memoized_id: str = "function") -> Tuple[int, float]:
        index_max = np.nanargmax(self._memoized_values[memoized_id])
        max_value = self._memoized_values[memoized_id][index_max]
        return index_max, max_value

    def reset_memoized_values(self):
        self._memoized_values = {}

    def retrieve_memoized_values(self, memoized_id: str = "function") -> np.ndarray:
        return self._memoized_values[memoized_id]["values"]

    def retrieve_memoized_feasable_set(
        self, memoized_id: str = "function"
    ) -> np.ndarray:
        return self._memoized_values[memoized_id]["feasible_set"]

    def memoize_function(
        self, memoized_id: str, values: np.ndarray, feasible_set: np.ndarray
    ):
        self._memoized_values[memoized_id] = {
            "values": values,
            "feasible_set": feasible_set,
        }

    def is_memoized(self, memoized_id: str) -> bool:
        if memoized_id is None:
            return False
        return memoized_id in self._memoized_values
    
    def forget_memoized(self, memoized_id: str):
        if memoized_id in self._memoized_values:
            del self._memoized_values[memoized_id]

    def __str__(self):
        if self.expression == "":
            return self.name
        return f"{self.name} = {self.expression}"

    def __repr__(self):
        return self.__str__()


class OptimizationObjectiveFunction(OptimizationFunction):
    pass


class OptimizationEqualityConstraint(OptimizationFunction):
    def evaluate_equality_constraint(
        self,
        x: np.ndarray,
        tolerance: float = 0.3,
        memoized_id: str = "function_equality_constraint",
        recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._call_and_save(
            x=x,
            function=self._function,
            memoized_id="function",
            recompute=recompute,
        )

        return self._call_and_save(
            x=x,
            function=self._function,
            condition=lambda x: np.abs(x) <= tolerance,
            memoized_id=memoized_id,
            recompute=recompute
        )

    def retrieve_constrainted_memorized_values(self) -> np.ndarray:
        return self.retrieve_memoized_values("function_equality_constraint")

    def retrieve_constrainted_memorized_feasable_set(self) -> np.ndarray:
        return self.retrieve_memoized_values("function_equality_constraint")

    def __str__(self):
        if self.expression == "":
            return f"{self.name}"
        return f"{self.name}: {self.expression} = 0"


class OptimizationInequalityConstraint(OptimizationFunction):
    def evaluate_inequality_constraint(
        self,
        x: np.ndarray,
        tolerance: float = 1e-6,
        memoized_id: str = "function_inequality_constraint",
        recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._call_and_save(
            x=x,
            function=self._function,
            memoized_id="function",
            recompute=recompute,
        )

        return self._call_and_save(
            x=x,
            function=self._function,
            condition=lambda x: x <= tolerance,
            memoized_id=memoized_id,
            recompute=recompute
        )

    def retrieve_constrainted_memorized_values(self) -> np.ndarray:
        return self.retrieve_memoized_values("function_inequality_constraint")

    def retrieve_constrainted_memorized_feasable_set(self) -> np.ndarray:
        return self.retrieve_memoized_feasable_set("function_inequality_constraint")

    def __str__(self):
        if self.expression == "":
            return f"{self.name}"
        return f"{self.name}: {self.expression} <= 0"
