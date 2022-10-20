# Multilayered Perceptron Project.

A perceptron neural network is essentially a function of the form: $f(\bold{x}) = \sigma(f_k(\dots(f_1(x_kw_1 + b_1)+b_k)\dots)$, where $\bold{x} \in \R^{n}$ and $n \in \N$ is the number of inputs into the neural network and $k \in \N$ is the number of layers.

The sigmoid function: $\sigma(x) = \frac{1}{1+ e^{-x} }$ is used to squish outputs of perceptrons to lie within the range $[0,1]$.
