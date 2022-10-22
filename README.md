# Multilayered Perceptron Project

A perceptron neural network is essentially a function of the form: $f(\bold{x}) = \sigma(f_k(\dots(f_1(x_kw_1 + b_1)+b_k)\dots)$, where $\bold{x} \in \R^{n}$ and $n \in \N$ is the number of inputs into the neural network and $k \in \N$ is the number of layers, and $f_i(x)$ are vector valued functions that will calculate the output of the $i$'th layer.

The sigmoid function: $\sigma(x) = \frac{1}{1+ e^{-x} }$ is used to squish outputs of perceptrons to lie within the range $[0,1]$. We will use this to calculate activation values for each node (perceptron).

## Calculating Activation Values

Suppose that we are in layer $i$ of the network that has a depth of $n_i$, layer $i$ had some input vector of $\bold{x}_i \in \R^{n_i}$, to find the output vector of the $i$ th layer i.e the input to the $i+1$ th layer (which I will refer to as $\bold{x}_{i+1}$), say that the $i+1$ th layer has depth $n_{i+1}$ thus $\bold{x}_{i+1} \in \R^{n_{i+1}}$. We can calculate $\bold{x}_{i+1}$ using the weights and biases of each layer as follows:

$$\bold{x}_{i+1} = \sigma(W_i\bold{x}_{i} + \bold{b}_{i+1})$$
Note that we apply $\sigma(x)$ component wise to $W_i\bold{x}_{i} + \bold{b}_{i+1}$, as $\sigma: \R \rightarrow (0,1)$ the above is simply for ease of notation.

Where $W_i$ is the matrix of weights connecting each vertex of layer $i$ to layer $i+1$ - it will look something like:

$$
W_i =
\begin{bmatrix}
w_{1,1} & \dots & w_{n_{i+1},1} \\\\ \vdots & \vdots & \vdots \\\\ w_{1,n_{i}} & \dots & w_{n_{i+1},n_{i}}
\end{bmatrix}
$$

Where $w_{a,b}$ is the weight from node $a$ in layer $i$ to node $b$ in layer $i+1$. Secondly $\bold{b}_{i+1}$ is simply the vector of biases of all of the nodes within the $i+1$ th layer.

The end goal of all of this is to find the largest activation value at the end of the network, in other words $\max_{x\in\bold{x}_k}$

The question you may have in mind is how can this network possibly learn? All of this preamble only discusses this 'static' definition of a network. The answer, as usual, is calculus!

## Making the network learn
