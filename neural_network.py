from os import stat
from typing import Callable, List

import numpy as np

from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.linear_algebra.linear_algebra import (Matrix,
                                                                    Vector)

H = 10 ** -5

class Layer:
    """
    Will describe a singular layer in a perceptron neural network.
    """
    def __init__(self,nodes_in:int,nodes_out:int, weights:Matrix = None, biases:Vector = None) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = self.get_random_weights() if not weights else weights
        self.biases = self.get_random_biases() if not biases else biases

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
            *[Functions.sigmoid(component) 
            for component in Vector.unpack_vector(weight_vector)]
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
        """Calculates the cost for a given layer

        Parameters
        ----------
        layer_output : Vector
        expected_output : Vector

        Returns
        -------
        Vector
            Will return a vector of cost for each output
        """
        error = Vector.unpack_vector(layer_output - expected_output)
        squared_error = [e ** 2 for e in error]
        return Vector(*squared_error)

    def cost(self,input:Vector,expected_output:Vector) -> float:
        error = self.layer_cost(
            self.get_output(input),
            expected_output)
        error_list = Vector.unpack_vector(error)
        return sum(error_list)/expected_output.dim

    
    def get_weight_derivative(self,layer:Layer,input:Vector,output:Vector) -> Matrix:
        weights = layer.weights

        initial_cost = self.cost(input,output)

        rows = weights.rows
        columns = weights.columns

        derivatives = []

        for i in range(rows):
            derivatives.append([])
            for j in range(columns):
                weight_matrix = weights.matrix
                weight_matrix[i][j] += H
                layer.weights = Matrix(*weight_matrix)
                updated_cost = self.cost(input,output)
                derivative = (updated_cost - initial_cost)/H
                derivatives[i].append(derivative)
                layer.weights = weights
        return Matrix(*derivatives)

    def get_bias_derivative(self,layer:Layer,input:Vector,output:Vector) -> Vector:
        biases = layer.biases
        initial_cost = self.cost(input,output)
        dimension = biases.dim
        offset_vector = Vector(*[H]*dimension)
        new_biases = layer.biases + offset_vector
        layer.biases = new_biases
        new_cost = self.cost(input,output)
        derivative = (new_cost-initial_cost)/H
        return Vector(*[derivative]*dimension)

    def learn(self,training_inputs:Vector,training_outputs:int,learning_rate:float = 0.5) -> Vector:
        weight_derivative_list = []
        bias_derivatives_list = []
        for layer in self.layers:
            weight_derivatives = self.get_weight_derivative(layer,training_inputs,training_outputs)
            bias_derivatives = self.get_bias_derivative(layer,training_inputs,training_outputs)
            weight_derivative_list.append(weight_derivatives)
            bias_derivatives_list.append(bias_derivatives)
        
        for i in range(len(weight_derivative_list)):
            layer = self.layers[i]
            layer.weights -= weight_derivative_list[i] * learning_rate
            layer.biases -= bias_derivatives_list[i] * learning_rate





if __name__ == "__main__":
    network = Neural_Network(1,2,2)
    input,output = Vector(15),Vector(0,1)
    print(network.cost(input,output))
    network.learn(input,output)
