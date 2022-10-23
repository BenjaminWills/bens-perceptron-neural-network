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

To make this network be able to learn I will be using the idea of `backpropogation`. Essentially this allows us to check the performance of our network against expected data, then we can use calculus to adjust our weights and then repeat.

We need a way to quantify how successful our network is. A logical step is to think about the squared error when compared to expected data, i.e if $y$ is an observed data point and $\hat{y}$ is what our neural network estimated, then $ (y - \hat{y})^2 $ is the squared error. If we were to calculate this error for each of the nodes of the output layer, then we would have $\sum_{k=1}^{n}{(y_{k} - \hat{y}_k)^2}$, it is then logical to find the average of these errors, this average square error function is called the `mean squared error` (MSE). I will be using this as the `Cost function`, $C(\bold{y},\bold{\hat{y}})$. $C: \R^{d_{output}} \rightarrow \R$

$$C(\bold{y},\bold{\hat{y}}) = \frac{1}{n}  \sum_{k=1}^{n}{(y_{k} - \hat{y}_k)^2} $$

We need to look at how the `weights` and `biases` effect the value of this cost function, in an effort to minimise it, and in turn minimise the error of the network. This is where the notion of `learning` comes from: repeated iterations of minimisation given training data!

### Backpropogation

We now need to view the cost function as a function of three variables: $\bold{w},\bold{b},\bold{y}$ which are the weights, biases and expected outputs respectively. This makes sense as $\bold{\hat{y}}$ that we saw above, is calculated using $\bold{w}$ and $\bold{b}$ so we can rewrite the cost function.

`New notation alert` We will refer to the weights and biases from layer $L$ linking node $i$ from layer $L-1$ to node $j$ in layer $L$ as: $w^{(L)}_{i,j}$ and $b^{(L)}_{j}$.

$$C(\bold{w},\bold{b},\bold{x},\bold{y}) = \frac{1}{n}  \sum_{k=1}^{n}{(y_{k} - \hat{y}_k(\bold{x},\bold{w},\bold{b}))^2}$$

Our task now is to minimise $C$ with respect to the weights and biases, so that we can adjust them to make the network more accurate. To do this we must consider the gradient of $C$.

$$
\nabla{C} = (
    \frac{\partial C}{\partial w^{{1}}_{1,1}} \ ,
    \  \dots \ ,
    \ \frac{\partial C}{\partial w^{(L)}}_{L,n_{L}} \ ,
    \frac{\partial C}{\partial b^{(1)}_{1}} \ ,
    \ \dots \ ,
    \frac{\partial C}{\partial b^{(L)}_{L}})
$$

Analytically we can minimise $C$ (which is a convex function; thus its minimum is global), by setting $\nabla{C} = 0$, and then solving for the appropriate weights and biases. However this is not simple for a computer to calculate, thus we must resort to numerical methods to minimise C.

#### Interlude into gradient descent

This is a numerical method that allows us to minimise a function consisting of an arbitrary number of input variables.

The method gets its name from the fact that the `gradient` of some function $f(\bold{x})$ - where $f:\R^n \rightarrow \R$ - is the direction of steepest `ascent`, thus its negative is the direction of steepest `descent` hence if we start with some point $\bold{x}_0$, we can logically move from that point to a point closer to a local minimum of $f$ by minusing some multiple of the gradient from $x_0$.

$$\bold{x}_{i+1} = \bold{x}_i - \alpha \nabla{f}(\bold{x_i})$$

Where $\alpha$ is the `learning rate` of the descent process. There are methods of choosing the learning rate to ensure sufficient decrease of the function.

Now you may be wondering: when do we terminate this descent? How can we quantify when the function is at a minimum?

Well, great questions! We define some small number $0 <\epsilon << 1$ such that when the gradient of $f$ falls within the interval $-\epsilon < \nabla f(\bold{x_{i+1}}) < \epsilon$ or equivalently $|\nabla{f(\bold{x_{i+1}})}| < \epsilon$, we terminate the process and return $\bold{x_{i+1}}$

So when applying this algorithm to our use case, we intend to tweak the `weights` and `biases` of our neurons in such a way that the `MSE`, $C$, is minimised. If we split $\bold{x}$ into $\bold{x} = (\bold{w},\bold{b})$, then minimising $C$ becomes a case of running gradient descent on $C$ with this iteration rule:

$$
\begin{align*}
\bold{w}_{i+1} = \bold{w}_{i} - \alpha \frac{\partial C}{\partial \bold{w}}
\\
\bold{b}_{i+1} = \bold{b}_{i} - \alpha \frac{\partial C}{\partial \bold{b}}
\end{align*}
$$

Where $\frac{\partial C}{\partial \bold{x}} = (
    \frac{\partial C}{\partial x_1},
    \
    \dots
    \
    ,
    \frac{\partial C}{\partial x_n}
)$ and $\bold{x} \in\R^n$.
