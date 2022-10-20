import numpy as np

from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)


class Layer:
    def __init__(self,nodes_in:int,nodes_out:int,weights_in:Matrix,biases:Vector) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = weights_in
        self.biases = biases
        self.output_weights = self.get_outputs()

    def get_outputs(self,input:Vector) -> Vector:
        weight_vector = self.weights * input + self.biases
        sigmoid_vector = Vector(
            [Functions.sigmoid(component) for component in Vector.unpack_vector(weight_vector)]
        )
        return sigmoid_vector

class Neural_Network:
    def __init__(self,layers:Vector) -> None:
        self.layers = layers

    def calculate_output(self,input:Vector) -> Vector:
        for layer in self.layers:
            input = layer.get_outputs(input)
        return input

    def classify_output(self,input:Vector) -> float:
        outputs = self.calculate_output(input)
        return np.argmax(Vector.unpack_vector(outputs))


if __name__ == "__main__":
    layer = Layer(
        2,
        2,
        Matrix(
            [0.5,0.5],
            [0.5,0.5]
            ),
        Vector(
            1,
            1
        ),
        Vector(
            1,
            1
        )
        )
    layer.weights.show_matrix()