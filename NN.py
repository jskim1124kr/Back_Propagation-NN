from layer import *
from helpers import numerical_gradient
from collections import OrderedDict


class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Layer1'] = Layer(self.params['W1'], self.params['b1'])
        self.layers['Layer2'] = Layer(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxLayer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, y):
        pred = self.predict(x)
        return self.lastLayer.forward(pred, y)

    def accuracy(self, x, y):
        pred = self.predict(x)
        pred = np.argmax(pred, axis=1)
        if y.ndim != 1: y = np.argmax(y, axis=1)

        accuracy = np.sum(pred == y) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def propagation(self, x, y):
        self.loss(x, y)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Layer1'].dW, self.layers['Layer1'].db
        grads['W2'], grads['b2'] = self.layers['Layer2'].dW, self.layers['Layer2'].db

        return grads

