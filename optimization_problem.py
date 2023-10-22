import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Any, Optional, Generic, TypeVar, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OptimizationProblem:
    def __init__(
        self,
        objective_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        inequality_constraints: List[Callable[[np.ndarray, np.ndarray], np.ndarray]],
        equality_constraints: List[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    ):
        self.objective_function = objective_function
        self.inequality_constraints = inequality_constraints
        self.equality_constraints = equality_constraints
    
    def apply_constraints(self, X1: np.ndarray, X2: np.ndarray, equality_error: float = 0.1) -> np.ndarray:
        Z = np.zeros(X1.shape)

        for constraint in self.equality_constraints:
            Z = np.where(abs(constraint(X1, X2)) <= equality_error, Z, np.nan)
        
        for constraint in self.inequality_constraints:
            Z = np.where(constraint(X1, X2) <= 0, Z, np.nan)
        
        Z = np.where(np.isnan(Z), np.nan, self.objective_function(X1, X2))

        return Z

    def find_min(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.isnan(Z).all():
            return np.nan, np.nan
        return np.unravel_index(np.nanargmin(Z, axis=None), Z.shape)
    
    def _add_min_marker(self, fig: go.Figure, X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, dim: int) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
        assert dim in [2, 3], "dim must be 2 or 3"
        
        min_idx = self.find_min(Z)

        if np.isnan(min_idx).any():
            return fig, np.nan, np.nan, np.nan

        if dim == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=[X1[min_idx]],
                    y=[X2[min_idx]],
                    z=[Z[min_idx]],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                        opacity=1,
                    ),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[X1[min_idx]],
                    y=[X2[min_idx]],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                        opacity=1,
                    ),
                )
            )

        return fig, X1[min_idx], X2[min_idx], Z[min_idx]
    
    def get_z(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        X1, X2 = np.meshgrid(x_1, x_2)
        Z = self.objective_function(X1, X2)
        return X1, X2, Z

    def plot_3d(self, x_1: np.ndarray, x_2: np.ndarray, equality_error=0.2) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
        X1, X2, Z = self.get_z(x_1, x_2)
        
        fig = go.Figure(
            data=[
                go.Surface(
                    z=Z,
                    x=x_1,
                    y=x_2,
                    opacity=0.2,
                )
            ]
        )

        Z = self.apply_constraints(X1, X2, equality_error)

        fig.add_trace(
            go.Surface(
                z=Z,
                x=x_1,
                y=x_2,
                opacity=1,
                showscale=False,
            )
        )

        # Add markers for the optimal points
        # find the minimum of the objective function
        fig, x1_min, x2_min, z_min = self._add_min_marker(fig, X1, X2, Z, 3)

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="x2",
        )

        return fig, x1_min, x2_min, z_min

    def plot_2d(self, x_1: np.ndarray, x_2: np.ndarray, equality_error=0.1) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
        X1, X2, Z = self.get_z(x_1, x_2)
        
        # plot the contour of the objective function
        fig = go.Figure(
            data=[
                go.Contour(
                    z=Z,
                    x=x_1,
                    y=x_2,
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


        Z = self.apply_constraints(X1, X2, equality_error)

        # Add markers for the optimal points
        # find the minimum of the objective function
        fig, x1_min, x2_min, z_min = self._add_min_marker(fig, X1, X2, Z, 2)

        # Highlight the feasible region, make the unfeasible region darker
        fig.add_trace(
            go.Contour(
                z=Z,
                x=x_1,
                y=x_2,
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

        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="x2",
        )

        return fig, x1_min, x2_min, z_min
