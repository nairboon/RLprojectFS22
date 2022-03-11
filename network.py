from collections import Iterable
import numpy as np


def create_weights(N_in, N_out):
    """create weights using Xavier initialization"""
    W = np.random.randn(N_out, N_in)
    W /= np.sqrt(N_in)
    return W


class MLP:
    """Multilayer Perceptron with ReLU activations"""
    def __init__(self, N_in, N_out, N_hs, activation="sigmoid"):
        assert activation in ["relu", "sigmoid"], "activation should be either 'relu' or 'sigmoid'"
        self.activation = activation

        if not isinstance(N_hs, Iterable):
            N_hs = [N_hs]

        # input layer
        self.W = [create_weights(N_in, N_hs[0])]
        self.b = [np.zeros((N_hs[0],))]

        # hidden layers
        for idx in range(len(N_hs) - 1):
            self.W.append(create_weights(N_hs[idx], N_hs[idx + 1]))
            self.b.append(np.zeros((N_hs[idx + 1],)))

        # output layer
        self.W.append(create_weights(N_hs[-1], N_out))
        self.b.append(np.zeros((N_out,)))

        # create cache for backpropagation
        self.zero_grad()

    def __call__(self, x):
        return self.forward(x)

    @property
    def params(self):
        """return all parameters as a 1D vector"""
        _params = []
        for layer in range(len(self.W)):
            _params.append(self.W[layer].flatten())
            _params.append(self.b[layer])
        return np.concatenate(_params)

    @property
    def grads(self):
        """return all parameters as a 1D vector"""
        _grads = []
        for layer in range(len(self.dW)):
            _grads.append(self.dW[layer].flatten())
            _grads.append(self.db[layer])
        return np.concatenate(_grads)

    def zero_grad(self):
        """set all gradients to zero and remove cache"""
        self.x, self.h = [], []
        self.dW, self.db = [], []
        for layer in range(len(self.W)):
            W, b = self.W[layer], self.b[layer]
            self.dW.append(np.zeros_like(W))
            self.db.append(np.zeros_like(b))
            self.x.append(None)
            self.h.append(None)

    def step(self, delta, eta=1.0):
        """perform a gradient update step given delta as a 1D vector"""
        idx = 0
        for layer in range(len(self.W)):
            W = self.W[layer]
            diff = np.prod(W.shape)
            dW = delta[idx:idx + diff].reshape(W.shape)
            self.W[layer] -= eta * dW
            idx += diff

            b = self.b[layer]
            diff = b.shape[0]
            db = delta[idx:idx + diff].reshape(b.shape)
            self.b[layer] -= eta * db
            idx += diff

    def forward(self, x):
        out = x.copy()
        for layer in range(len(self.W)):
            self.x[layer] = out.copy()

            W, b = self.W[layer], self.b[layer]
            h = out @ W.T + b

            if self.activation == 'relu':
                out = h * (h > 0).astype(float)
            elif self.activation == 'sigmoid':
                out = 1 / (1 + np.exp(-h))
            else:
                raise NotImplementedError

            self.h[layer] = h.copy()

        return out

    def backward(self, dx):
        assert not any([x is None for x in self.x]), "must run a forward pass first"

        dout = dx.copy()
        for layer in reversed(range(len(self.W))):
            W = self.W[layer]
            x, h = self.x[layer], self.h[layer]

            if self.activation == 'relu':
                dh = dout * (h > 0).astype(float)
            elif self.activation == 'sigmoid':
                sig = 1 / (1 + np.exp(-h))
                dh = dout * sig * (1 - sig)

            self.dW[layer] = np.outer(dh, x) / x.shape[0]
            self.db[layer] = dh.mean(axis=0)
            dout = dh @ W
