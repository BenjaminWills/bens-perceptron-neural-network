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
        self.output_weights = self.get_outputs()

    def get_outputs(self):
        weight_vector = self.weights * self.input + self.biases
        sigmoid_vector = Vector(
            [Functions.sigmoid(component) for component in Vector.unpack_vector(weight_vector)]
        )
        return sigmoid_vector

class Neural_Network:
    def __init__(self,layers:Vector) -> None:
        self.layers = layers

    def create_network(self):
        pass


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