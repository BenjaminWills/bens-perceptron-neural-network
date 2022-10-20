import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from Mathematics_fundamentals.calculus.differentiation.differentiation import \
    Differentiation
from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)

plt.style.use("dark_background")


class Visualisation:
    def make_axis(self, all=False):
        fig, ax = plt.subplots()
        if all:
            return ax, fig
        return ax

    def matplotlib_config(self, ax, equal=False):
        if equal == True:
            ratio = "equal"
        else:
            ratio = "auto"
        ax.set_aspect(ratio)
        ax.grid(True, which="both", linestyle=":")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.yaxis.tick_left()
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        ax.xaxis.tick_bottom()

    def get_function_visualisation(self, function, start, end, steps, ax):
        if start >= end:
            return None
        plotting_space = np.linspace(start, end, steps)
        ax.plot(plotting_space, function(plotting_space))
        self.matplotlib_config(ax)

    def get_conic_visualisation(self, function, start, end, steps, ax):
        # Conic sections are one to many functions. So we need to plot them differently.
        plotting_space = np.linspace(start, end, steps)
        ax.plot(plotting_space, function(plotting_space)[0])
        ax.plot(plotting_space, function(plotting_space)[1])
        self.matplotlib_config(ax)

    def visualise_function_linear_transform(
        self, function, start, end, steps, ax, transformation_matrix
    ):
        space = np.linspace(start, end, steps)
        lin = Linalg()
        points = lin.transform_function(space, function, transformation_matrix)
        self.matplotlib_config(ax)
        ax.scatter(points[0], points[1], marker=".", s=0.2)

    def visualise_conic_linear_transform(
        self, function, start, end, steps, ax, transformation_matrix
    ):
        space = np.linspace(start, end, steps)
        lin = Linalg()
        pos_points = lin.transform_conic_function(
            space, function, transformation_matrix
        )[0]
        neg_points = lin.transform_conic_function(
            space, function, transformation_matrix
        )[1]
        self.matplotlib_config(ax)
        ax.scatter(pos_points[0], pos_points[1], marker=".", s=0.2)
        ax.scatter(neg_points[0], neg_points[1], marker=".", s=0.2)

    def visualise_tangent(self, function, x, start, end, steps, ax):
        diff = Differentiation()
        func = Functions()
        slope = diff.get_derivative(function, x)
        intercept = function(x) - x * slope

        def tangent(x):
            return func.line(x, slope, intercept)

        self.matplotlib_config(ax)
        self.get_function_visualisation(tangent, start, end, steps, ax)
        self.get_function_visualisation(function, start, end, steps, ax)
        return "plotted."

    def visualise_conic_tangent(self, function, x, start, end, steps, ax):
        diff = Differentiation()
        func = Functions()
        conic_derivative = diff.get_conic_derivative(function, x)
        function_values = function(x)
        slope_one = conic_derivative[0]
        intercept_one = function_values[0] - x * slope_one
        slope_two = conic_derivative[1]
        intercept_two = function_values[1] - x * slope_two

        def upper_tangent(x):
            return func.line(x, slope_one, intercept_one)

        def lower_tangent(x):
            return func.line(x, slope_two, intercept_two)

        self.matplotlib_config(ax, equal=True)
        self.get_function_visualisation(upper_tangent, start, end, steps, ax)
        self.get_function_visualisation(lower_tangent, start, end, steps, ax)
        self.get_conic_visualisation(function, start, end, steps, ax)
        return "plotted."

    def show_visualisation(self):
        plt.show()


# TODO:
# - NEWTON RAPHSON GRAPH
# -
