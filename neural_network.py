from typing import List

import numpy as np

from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)


class Layer:
    def __init__(self,nodes_in:int,nodes_out:int,inputs:Vector) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.inputs = inputs
        self.weights = self.get_random_weights()
        self.biases = self.get_random_biases()
        self.output_weights = self.get_outputs()

    def get_random_weights(self):
        random_matrix = [[np.random.uniform(-10,10) for i in range(self.nodes_in)] for j in range(self.nodes_out)]
        return Matrix(*random_matrix)
    def get_random_biases(self):
        random_list = [np.random.uniform(-10,10) for i in range(self.nodes_in)]
        return Vector(*random_list)

    def get_outputs(self) -> Vector:
        weight_vector = self.weights * self.inputs + self.biases
        sigmoid_vector = Vector(
            [Functions.sigmoid(component) for component in Vector.unpack_vector(weight_vector)]
        )
        return sigmoid_vector

class Neural_Network:
    def __init__(self,inputs:Vector,*layer_sizes:List[int]) -> None:
        self.layer_sizes = layer_sizes
        self.inputs = inputs
        self.layers = self.create_network()

    def create_network(self):
        layers = []
        inputs = self.inputs
        for i in range(len(self.layer_sizes)):
            new_layer = Layer(
                        self.layer_sizes[i],
                        self.layer_sizes[i+1],
                        inputs,
                        )
            inputs = new_layer.get_outputs()
            layers.append(new_layer)

    def classify_output(self) -> float:
        outputs = self.layers[self.layer_sizes[-1]].get_outputs()
        return np.argmax(Vector.unpack_vector(outputs))


if __name__ == "__main__":
    network = Neural_Network(Vector(1,1),2,3,2)
    