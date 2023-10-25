import plotly.graph_objects as go
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
    
    def plot_feasible_region(self) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Scatter(
                    y=self.get_fesable_objective_function(),
                    x=self.x,
                    mode="lines",
                    name="Feasible region"
                )
            ]
        )
        fig.update_layout(
            title="Feasible region",
            xaxis_title="x1",
            yaxis_title="y",
            width=800,
            height=500,
            margin=dict(r=20, l=10, b=10, t=100)
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
            margin=dict(r=20, l=10, b=10, t=100)
        )

        return fig