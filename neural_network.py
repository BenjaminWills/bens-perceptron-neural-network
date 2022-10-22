from typing import List

import numpy as np

from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)


class Layer:
    """
    Will describe a singular layer in a perceptron neural network.
    """
    def __init__(self,nodes_in:int,nodes_out:int) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = self.get_random_weights()
        self.biases = self.get_random_biases()

    def get_random_weights(self) -> Matrix:
        """Will get a random weight matrix with dimensions nodes_in x nodes_out

        Returns
        -------
        Matrix
            A matrix of weights where the (i,j) component is the weight from node i
            in first layer to node j in the second layer.
        """
        random_matrix = [[np.random.uniform(-1,1) for i in range(self.nodes_in)] for j in range(self.nodes_out)]
        return Matrix(*random_matrix)
    def get_random_biases(self) -> Vector:
        """Will generate a vector of a random bias with the dimension of nodes_out

        Returns
        -------
        Vector
            A vector of biases.
        """
        r = np.random.uniform(-1,1)
        random_list = [r for i in range(self.nodes_out)]
        return Vector(*random_list)

    def get_outputs(self,inputs:Vector,verbose:bool = False) -> Vector:
        """Will get the output of a layer of the network, given a suitable input.

        Parameters
        ----------
        inputs : Vector
            A vector with dimensions nodes_in
        verbose : bool, optional
            Will show the weights,inputs,biases and outputs when calculating the output, by default False

        Returns
        -------
        Vector
            The result of sending the arguments through the layer, squished into a sigmoid.
        """
        if verbose:
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
        else:         
            weight_vector = self.weights * inputs + self.biases
        sigmoid_vector = Vector(
            *[Functions.sigmoid(component) for component in Vector.unpack_vector(weight_vector)]
        )
        return sigmoid_vector

class Neural_Network:
    """
    Will describe a set of layers known as a network.
    """
    def __init__(self,*layer_sizes:List[int]) -> None:
        self.layer_sizes = layer_sizes
        self.layers = self.create_network()

    def create_network(self) -> List[Layer]:
        """Will create the network specified by the layer sizes array.

        Returns
        -------
        List[Layer]
            A list of layers, all connected with their respective weight matrices/bias vectors.
        """
        layers = []
        for i in range(len(self.layer_sizes)-1):
            new_layer = Layer(
                        self.layer_sizes[i],
                        self.layer_sizes[i+1],
                        )
            layers.append(new_layer)
        return layers

    def get_output(self,input:Vector,verbose:bool = False) -> Vector:
        """Will calculate the output of the whole network given an input,
        i.e an input will be sent through the layers to the end.

        Parameters
        ----------
        input : Vector
            A vector matching the dimension of the first layer
        verbose : bool, optional
            Will show the weights,inputs,biases and outputs when calculating the output
            for each layer, by default False

        Returns
        -------
        Vector
            The output of the network
        """
        for layer in self.layers:
            input = layer.get_outputs(input,verbose=verbose)
        return input

    def classify_output(self,input:Vector) -> int:
        """Will classify the output, and return the index of the largest output.

        Parameters
        ----------
        input : Vector
            A vector matching the dimension of the first layer

        Returns
        -------
        int
            Index of largest output
        """
        outputs = self.get_output(input)
        return np.argmax(Vector.unpack_vector(outputs))

    def layer_cost(self,layer_output:Vector,expected_output:Vector) -> Vector:
        error = Vector.unpack_vector(layer_output - expected_output)
        squared_error = [e ** 2 for e in error]
        return Vector(*squared_error)

    def cost(self,network_output:Vector,expected_output:Vector) -> float:
        error = self.layer_cost(network_output,expected_output)
        error_list = Vector.unpack_vector(error)
        return sum(error_list)/expected_output.dim




if __name__ == "__main__":
    network = Neural_Network(2,3,2)

    input = Vector(1,1)
    expected_output = Vector(0.5,0.5)

    output = network.get_output(input)
    output.show_vector()
    print(network.cost(output,expected_output))