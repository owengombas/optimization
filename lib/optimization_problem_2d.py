import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.optimization_problem import OptimizationProblem
from lib.optimization_function import (
    OptimizationFunction,
    OptimizationObjectiveFunction,
    OptimizationEqualityConstraint,
    OptimizationInequalityConstraint,
)


class OptimizationProblem2D(OptimizationProblem):
    @property
    def x(self) -> np.ndarray:
        return self.domain[0]

    def plot_feasible_region_along_constraints(self) -> go.Figure:
        f = self.plot_functions()
        f_2 = self.plot_feasible_region()
        f.add_traces(f_2.data)
        return f

    def plot_feasible_region(self) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Scatter(
                    y=self.get_fesable_objective_function(),
                    x=self.x,
                    mode="lines",
                    name="Feasible region",
                )
            ]
        )

        _, min_coords = self.find_min()
        if min_coords is not None:
            fig.add_trace(
                go.Scatter(
                    y=[self.objective_function.evaluate_function(min_coords, memoized_id=None)],
                    x=[min_coords[0]],
                    mode="markers",
                    name="Min",
                )
            )

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100),
        )

        return fig

    def plot_functions(self) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Scatter(
                    y=self.objective_function.retrieve_memoized_values(),
                    x=self.x,
                    mode="lines",
                    name=self.objective_function.name,
                )
            ]
        )

        for constraint in self.equality_constraints + self.inequality_constraints:
            fig.add_trace(
                go.Scatter(
                    y=constraint.retrieve_constrainted_memorized_values(),
                    x=self.x,
                    mode="lines",
                    name=constraint.name,
                )
            )

        fig.update_layout(
            title="All functions",
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100),
        )

        return fig

    def plot_feasible_region_along_constraints_with_lagrangian(
        self,
        lambdas: np.ndarray,
        lagrangians: List[OptimizationFunction],
        lagrangians_infimum: OptimizationFunction,
        lambdas_dual_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[go.Figure, go.Figure]:
        plot = self.plot_feasible_region_along_constraints()

        for i, l in enumerate(lagrangians):
            plot.add_trace(
                go.Scatter(
                    x=self.domain[0],
                    y=l.evaluate_function(),
                    mode="lines",
                    marker=dict(
                        size=2,
                        color="red",
                        opacity=0.1,
                    ),
                    opacity=0.1,
                    name=f"Lagrangian for lambda = {lambdas[i]}",
                )
            )

        lagrangians_infimum_values = lagrangians_infimum.evaluate_function(lagrangians)
        plot.add_trace(
            go.Scatter(
                x=self.domain[0][lagrangians_infimum_values[0]],
                y=lagrangians_infimum_values[1],
                mode="lines",
                marker=dict(
                    size=1,
                    color="orange",
                    opacity=1,
                ),
                opacity=1,
                name=f"g(lambda)",
            )
        )

        g = go.Figure(
            data=[
                go.Scatter(
                    x=lambdas,
                    y=lagrangians_infimum_values[1],
                    mode="lines+markers+text",
                    line=dict(
                        color="orange",
                        width=1,
                    ),
                    marker=dict(
                        size=5,
                        color="orange",
                        opacity=1,
                    ),
                    opacity=1,
                    name=f"g(lambda) from computed lagrangians",
                    text=[f"{lagrangians_infimum_values[1][i]:.2f}" for i in range(len(lambdas))],
                    textposition="top right",
                ),
            ]
        )

        dual_function_values = self.dual_function.evaluate_function(lambdas if lambdas_dual_function is None else lambdas_dual_function)
        if self.dual_function is not None:
            g.add_trace(
                go.Scatter(
                    x=lambdas if lambdas_dual_function is None else lambdas_dual_function,
                    y=dual_function_values,
                    mode="lines",
                    marker=dict(
                        size=2,
                        color="red",
                        opacity=1,
                    ),
                    opacity=1,
                    name=f"{str(self.dual_function)}",
                ),
            )
        
        g.update_layout(
            title="g(lambda)",
            xaxis_title="lambda",
            yaxis_title="g(lambda)",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100),
        )

        return plot, g
