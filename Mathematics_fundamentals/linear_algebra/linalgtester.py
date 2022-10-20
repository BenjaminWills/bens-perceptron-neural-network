import numpy as np

from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.visualisations.visualisations import \
    Visualisation

theta = 5 / 3 * np.pi
TRANSFORMATION_MATRIX = [
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)],
]



