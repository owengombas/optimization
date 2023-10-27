import plotly.graph_objects as go
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


class OptimizationProblem3D(OptimizationProblem):
    @property
    def x(self) -> np.ndarray:
        return self.domain[0]

    @property
    def y(self) -> np.ndarray:
        return self.domain[1]

    def plot_feasable_countours(
        self,
    ) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
        # plot the contour of the objective function
        fig = go.Figure(
            data=[
                go.Contour(
                    z=self.objective_function.retrieve_memoized_values(),
                    x=self.x,
                    y=self.y,
                    contours=dict(
                        start=0,
                        end=10,
                        size=1,
                    ),
                    line_smoothing=1,
                    opacity=0.2,
                    name=f"{str(self.objective_function)}",
                )
            ]
        )

        # Highlight the feasible region, make the unfeasible region darker
        fig.add_trace(
            go.Contour(
                z=self.get_fesable_objective_function()[0],
                x=self.x,
                y=self.y,
                contours=dict(
                    start=0,
                    end=10,
                    size=1,
                ),
                line_smoothing=1,
                opacity=1,
                showscale=False,
                name="Feasible region on the objective function",
            )
        )

        _, coords = self.find_min()
        if coords is not None:
            fig.add_trace(
                go.Scatter(
                    x=[coords[0]],
                    y=[coords[1]],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                        opacity=1,
                    ),
                    name="Min",
                )
            )

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="x2",
        )

        return fig

    def plot_feasable_3d(self) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=self.objective_function.retrieve_memoized_values(),
                    x=self.x,
                    y=self.y,
                    opacity=0.2,
                    name=f"{str(self.objective_function)}",
                )
            ]
        )

        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )

        fig.add_trace(
            go.Surface(
                z=self.get_fesable_objective_function()[0],
                x=self.x,
                y=self.y,
                opacity=1,
                showscale=False,
                name="Feasible region on the objective function",
            )
        )

        min_value, coords = self.find_min()
        if coords is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[coords[0]],
                    y=[coords[1]],
                    z=[min_value],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                        opacity=1,
                    ),
                    name="Min",
                )
            )

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="x2",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100),
        )

        return fig

    def plot_functions_3d(self) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=self.objective_function.retrieve_memoized_values(),
                    x=self.x,
                    y=self.y,
                    opacity=0.8,
                    colorscale="Viridis",
                    name=f"{str(self.objective_function)}",
                )
            ]
        )

        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )

        for constraint in self.equality_constraints:
            fig.add_trace(
                go.Surface(
                    z=constraint.retrieve_constrainted_memorized_values(),
                    x=self.x,
                    y=self.y,
                    opacity=1,
                    showscale=False,
                    name=f"{str(constraint)}",
                )
            )

        for constraint in self.inequality_constraints:
            fig.add_trace(
                go.Surface(
                    z=constraint.retrieve_constrainted_memorized_values(),
                    x=self.x,
                    y=self.y,
                    opacity=1,
                    showscale=False,
                    name=f"{str(constraint)}",
                )
            )

        fig.update_layout(
            title="All functions",
            xaxis_title="x1",
            yaxis_title="x2",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100),
        )

        return fig

    def plot_feasable_along_with_langrangians(self, lagrangians: List[OptimizationFunction]) -> go.Figure:
        plot = self.plot_feasable_3d()

        for l in lagrangians:
            plot.add_trace(
                go.Scatter3d(
                    z=l.evaluate_function()[0],
                    x=self.x,
                    y=self.y,
                    opacity=1,
                    name=f"str{l}",
                )
            )

        return plot
