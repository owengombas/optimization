import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.optimization_problem import OptimizationProblem


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
                    y=[self.objective_function(min_coords)],
                    x=[min_coords[0]],
                    mode="markers",
                    name="Min",
                )
            )

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
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
                    y=self.objective_function_values,
                    x=self.x,
                    mode="lines",
                    name=self.objective_function.__name__,
                )
            ]
        )

        for constraint in self.equality_constraints + self.inequality_constraints:
            fig.add_trace(
                go.Scatter(
                    y=self.compute_function_values(constraint),
                    x=self.x,
                    mode="lines",
                    name=constraint.__name__,
                )
            )

        fig.update_layout(
            title="All functions",
            xaxis_title="x1",
            yaxis_title="y",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100),
        )

        return fig

    def plot_feasible_region_along_constraints_with_lagrangian(
        self,
        lambdas: np.ndarray,
        lagrangians: np.ndarray,
        lagrangians_infimum: np.ndarray,
        langrangian_real: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[go.Figure, go.Figure]:
        plot = self.plot_feasible_region_along_constraints()

        for i, l in enumerate(lagrangians):
            plot.add_trace(
                go.Scatter(
                    x=self.domain[0],
                    y=l,
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

        plot.add_trace(
            go.Scatter(
                x=self.domain[0][lagrangians_infimum[0]],
                y=lagrangians_infimum[1],
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
                    y=lagrangians_infimum[1],
                    mode="lines",
                    marker=dict(
                        size=2,
                        color="orange",
                        opacity=1,
                    ),
                    opacity=1,
                    name=f"g(lambda)",
                ),
            ]
        )

        if self.g is not None:
            g.add_trace(
                go.Scatter(
                    x=lambdas if langrangian_real is None else langrangian_real,
                    y=self.g(lambdas if langrangian_real is None else langrangian_real),
                    mode="lines",
                    marker=dict(
                        size=2,
                        color="red",
                        opacity=1,
                    ),
                    opacity=1,
                    name=f"real g(lambda)",
                ),
            )

        return plot, g
