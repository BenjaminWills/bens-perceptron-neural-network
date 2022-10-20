from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)


class Layer:
    def __init__(self,nodes_in:int,nodes_out:int,weights:Matrix,biases:Vector) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = weights
        self.biases = biases

class Neural_Network:
    def __init__(self,layers:Vector) -> None:
        self.layers = layers
