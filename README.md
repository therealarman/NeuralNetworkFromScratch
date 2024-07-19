# Neural Network From Scratch
Building a simple multi-classification neural network using only numpy and the python math library


$$J\theta = CE = \sum_{k=1}^k y_k \times log(\hat{y_k})$$

$$S(x_i) = \dfrac{e^{x_i}}{\sum_{k=1}^k e^{x_k}}$$

$$\delta_{O_k} = 
\dfrac{\partial J\theta}{\partial x_i} = 
\dfrac{\partial J\theta}{\partial S(x_i)} \times \dfrac{\partial S(x_i)}{\partial x_i} = \left(-\dfrac{y}{\hat{y}}\right) \times (\hat{y} \times (1-\hat{y})) = \hat{y}-y$$

$$\delta_{h_i} = \left(\sum_{k=1}^k w_{h_i, o_k} \times \delta_{O_k} \right) \times g'(a_i)$$

