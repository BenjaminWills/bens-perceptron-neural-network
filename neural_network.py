from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)


class Layer:
    def __init__(self,nodes_in:int,nodes_out:int,weights_in:Matrix,biases:Vector,input:Vector) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = weights_in
        self.biases = biases
        self.input = input
        self.outputs = self.get_outputs()

    def get_outputs(self):
        weight_vector = self.weights * self.input + self.biases
        sigmoid_vector = Vector(
            [Functions.sigmoid(component) for component in weight_vector]
        )
        return sigmoid_vector

class Neural_Network:
    def __init__(self,layers:Vector) -> None:
        self.layers = layers
