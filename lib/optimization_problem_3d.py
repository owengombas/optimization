import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.optimization_problem import OptimizationProblem

class OptimizationProblem3D(OptimizationProblem):    
    @property
    def x(self) -> np.ndarray:
        return self.domain[0]

    @property
    def y(self) -> np.ndarray:
        return self.domain[1]
    
    def plot_feasable_countours(self) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:        
        # plot the contour of the objective function
        fig = go.Figure(
            data=[
                go.Contour(
                    z=self.objective_function_values,
                    x=self.x,
                    y=self.y,
                    contours=dict(
                        start=0,
                        end=10,
                        size=1,
                    ),
                    line_smoothing=1,
                    opacity=0.2,
                )
            ]
        )

        # Highlight the feasible region, make the unfeasible region darker
        fig.add_trace(
            go.Contour(
                z=self.get_fesable_objective_function(),
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
                )
            )

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="x2",
        )

        return fig

    def plot_feasable_3d(self) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=self.objective_function_values,
                    x=self.x,
                    y=self.y,
                    opacity=0.2,
                )
            ]
        )

        fig.add_trace(
            go.Surface(
                z=self.get_fesable_objective_function(),
                x=self.x,
                y=self.y,
                opacity=1,
                showscale=False,
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
                )
            )

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="x2",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100)
        )

        return fig

    def plot_functions_3d(self) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=self.objective_function_values,
                    x=self.x,
                    y=self.y,
                    opacity=0.8,
                    colorscale="Viridis",
                )
            ]
        )

        for constraint in self.equality_constraints + self.inequality_constraints:
            fig.add_trace(
                go.Surface(
                    z=self.compute_function_values(constraint),
                    x=self.x,
                    y=self.y,
                    opacity=1,
                    showscale=False,
                )
            )
        
        fig.update_layout(
            title="All functions",
            xaxis_title="x1",
            yaxis_title="x2",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100)
        )

        return fig

    def add_lagrangian_trace_3d(
        self,
        fig: go.Figure,
        L: np.ndarray
    ) -> go.Figure:
        fig.add_trace(
            go.Scatter3d(
                x=self.x,
                y=self.y,
                z=L,
                mode="markers",
                showlegend=False,
                line=dict(
                    color="red",
                    width=2,
                ),
            )
        )
        return fig

    def add_axis_3d_plot(self, fig: go.Figure, z: np.ndarray = None) -> go.Figure:
        fig.add_trace(
            go.Scatter3d(
                x=[np.min(self.x), np.max(self.x)],
                y=[0, 0],
                z=[0, 0],
                mode='lines',
                showlegend=False,
                marker=dict(
                    color='black',
                    size=10,
                    symbol='cross'
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[np.min(self.y), np.max(self.y)],
                z=[0, 0],
                mode='lines',
                showlegend=False,
                marker=dict(
                    color='black',
                    size=10,
                    symbol='cross'
                ),

            )
        )
        if z is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0],
                    y=[0, 0],
                    z=[np.min(z), np.max(z)],
                    mode='lines',
                    showlegend=False,
                    marker=dict(
                        color='black',
                        size=10,
                        symbol='cross'
                    ),
                )
            )
        return fig