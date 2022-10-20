from typing import Callable

import numpy as np
import scipy.optimize as optimise

from Mathematics_fundamentals.numbers.numbers import Complex


class Functions:
    @staticmethod
    def line(x, gradient, intercept):
        return gradient * x + intercept

    @staticmethod
    def polynomial(x, *coefficients):
        output = 0
        degree = len(coefficients) - 1
        for index, coefficient in enumerate(coefficients):
            output += coefficient * x ** (degree - index)
        return output

    @staticmethod
    def reciprocal(x, horizontal_shift=0, vertical_shift=0):
        if x.any() == 0:
            return 0
        return 1 / (x - horizontal_shift) + vertical_shift

    @staticmethod
    def step(x, horizontal_shift=0):
        if x.any() == horizontal_shift:
            return 1
        else:
            return 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sinh(x):
        return (Functions.exp(x) - Functions.exp(x * -1)) * 0.5

    @staticmethod
    def cosh(x):
        return (Functions.exp(x) + Functions.exp(x * -1)) * 0.5

    @staticmethod
    def tanh(x):
        if x.any() == 0:
            return 0
        return Functions.sinh(x) / Functions.cosh(x)

    @staticmethod
    def bell_curve(x, stretch=1, horizontal_shift=0):
        return stretch * np.exp(-((x + horizontal_shift) ** 2))

    @staticmethod
    def sin(x, frequency=1):
        return np.sin(x / (2 * np.pi) * frequency)

    @staticmethod
    def cos(x, frequency=1):
        return np.cos(x / (2 * np.pi) * frequency)

    @staticmethod
    def tan(x, frequency=1):
        return np.tan(x / (2 * np.pi) * frequency)

    @staticmethod
    def exp(x):
        if isinstance(x, Complex):
            re = x.re
            im = x.im
            return Complex(np.exp(re) * np.cos(im), np.exp(re) * np.sin(im))
        return np.exp(x)

    @staticmethod
    def ln(x):
        if x <= 0:
            return 0
        return np.log(x)

    @staticmethod
    def log_base_n(x, n):
        return np.log(x) / np.log(n)

    @staticmethod
    def circle(x, radius, centre_x, centre_y):
        pos_y_co_ordinate = centre_y + np.sqrt(radius**2 - (x - centre_x) ** 2)
        neg_y_co_ordinate = centre_y - np.sqrt(radius**2 - (x - centre_x) ** 2)
        return [neg_y_co_ordinate, pos_y_co_ordinate]

    @staticmethod
    def hyperbola(x, major, minor):
        y_co_ordinate = major * np.sqrt(1 + (x / minor) ** 2)
        return [-y_co_ordinate, y_co_ordinate]

    @staticmethod
    def elipse(x, major, minor):
        y_co_ordinate = major * np.sqrt(1 - (x / minor) ** 2)
        return [-y_co_ordinate, y_co_ordinate]

    @staticmethod
    def get_scalar_inverse(a:float,function:Callable) -> float:
        """Will return the inverse value of a function at the point x,
        so if f(x) = a, then this function will find x.

        Parameters
        ----------
        a : float
        function : Callable

        Returns
        ----------
            float
        """
        minimising_function = lambda x: (function(x) - a) ** 2
        return optimise.minimize_scalar(minimising_function)['x']

