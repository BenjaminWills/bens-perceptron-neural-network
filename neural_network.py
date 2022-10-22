from typing import List

import numpy as np

from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)


class Layer:
    def __init__(self,nodes_in:int,nodes_out:int) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = self.get_random_weights()
        self.biases = self.get_random_biases()

    def get_random_weights(self):
        random_matrix = [[np.random.uniform(-10,10) for i in range(self.nodes_in)] for j in range(self.nodes_out)]
        return Matrix(*random_matrix)
    def get_random_biases(self):
        r = np.random.uniform(-10,10)
        random_list = [r for i in range(self.nodes_out)]
        return Vector(*random_list)

    def get_outputs(self,inputs:Vector) -> Vector:
        print('=================================WEIGHTS=====================================')
        self.weights.show_matrix()
        print('=================================INPUTS======================================')
        inputs.show_vector()
        print('=================================BIASES======================================')
        self.biases.show_vector()
        print('================================OUTPUTS======================================')
        weight_vector = self.weights * inputs + self.biases
        weight_vector.show_vector()
        print('=============================END OF LOOP=====================================')
        sigmoid_vector = Vector(
            *[Functions.sigmoid(component) for component in Vector.unpack_vector(weight_vector)]
        )
        return sigmoid_vector

class Neural_Network:
    def __init__(self,*layer_sizes:List[int]) -> None:
        self.layer_sizes = layer_sizes
        self.layers = self.create_network()

    def create_network(self) -> List[Layer]:
        layers = []
        for i in range(len(self.layer_sizes)-1):
            new_layer = Layer(
                        self.layer_sizes[i],
                        self.layer_sizes[i+1],
                        )
            layers.append(new_layer)
        return layers

    def get_output(self,input:Vector) -> Vector:
        for layer in self.layers:
            input = layer.get_outputs(input)
        return input

    def classify_output(self,input:Vector) -> float:
        outputs = self.get_output(input)
        return np.argmax(Vector.unpack_vector(outputs))


if __name__ == "__main__":
    network = Neural_Network(2,3,2,3,3,3,3,2)

    network.get_output(Vector(1,1)).show_vector()