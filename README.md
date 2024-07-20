# Neural Network From Scratch

Building a modular Deep Neural Network system using only the NumPy and Math python libraries. The system has the capacity to add layers with different activation functions, backpropogate error and plot average cost.

# Architecture & Math

The system built here is meant for Multi-Classification problems with Softmax as the output layer activation function. Therefore, a Cross-Entropy cost function was implemented:

$$Softmax(x_i) = \dfrac{e^{x_i}}{\sum_{k=1}^k e^{x_k}}$$
$$J\theta = CE = \sum_{k=1}^k y_k \times log(\hat{y_k})$$

Based on the Softmax activation and Cross-Entropy loss, the local gradients for output and hidden layer nodes respectively are calculated as follows using the chain rule of differentiation:

$$\delta_{O_k} = 
\dfrac{\partial J\theta}{\partial x_i} = 
\dfrac{\partial J\theta}{\partial S(x_i)} \times \dfrac{\partial S(x_i)}{\partial x_i} = \left(-\dfrac{y}{\hat{y}}\right) \times (\hat{y} \times (1-\hat{y})) = \hat{y}-y$$

$$\delta_{h_i} = \left(\sum_{k=1}^k w_{h_i, o_k} \times \delta_{O_k} \right) \times g'(a_i)$$

Lastly, weights are updated using the following formula:

$$\Delta w_{x_{1}, h_{1}} = \eta\delta_{h_i}y_{x_i}$$
$$new \space w_{x_{1}, h_{1}} = w_{x_{1}, h_{1}}-\Delta w_{x_{1}, h_{1}}$$