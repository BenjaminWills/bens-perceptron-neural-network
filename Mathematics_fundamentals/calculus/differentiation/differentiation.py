
import math

import numpy as np

from Mathematics_fundamentals.numbers.numbers import Real


class Differentiation:
    def get_derivative(self, function, x):
        dx = 10**-5
        vert_change = function(x + dx) - function(x)
        slope = vert_change / dx
        return slope

    def get_conic_derivative(self, function, x):
        dx = 10**-5
        shifted = function(x + dx)
        original = function(x)
        first_diff = shifted[0] - original[0]
        second_diff = shifted[1] - original[1]
        first_slope = first_diff / dx
        second_slope = second_diff / dx
        return [first_slope, second_slope]

    def newton_raphson_method(self, function, initial_point, verbose=False):
        x0 = initial_point
        max_iterations = 100000
        iteration_count = 0
        while abs(function(x0)) > 10**-5:
            if iteration_count == max_iterations:
                return "DIVERGENT: possibly a saddle point nearby.", x0
            slope = self.get_derivative(function, x0)
            func_value = function(x0)
            x0 = x0 - (func_value / slope)
            iteration_count += 1
        x0 = np.round(x0, 4)
        if verbose:
            return (
                f"""Iteration count: {iteration_count}, root: ({x0},{function(x0)})"""
            )
        return [[x0], [function(x0)]]

    def get_nth_derivative(self, function, x, order=1):
        # at each point we need to find another derivative of a derivative.
        if order == 1:
            return self.get_derivative(function, x)
        else:

            def deriv(x):
                return self.get_derivative(function, x)

            return self.get_nth_derivative(deriv, x, order - 1)

    def taylor_series(self, function, x, order, centre=0):
        output = function(centre)
        for i in range(1, order):
            output += (
                self.get_nth_derivative(function, centre, order)
                * ((x - centre) ** order)
            ) / math.factorial(order)
        return output
