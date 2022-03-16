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
        assert activation in ["relu", "sigmoid", None], "activation should be either 'relu' or 'sigmoid' or None"
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
        try:
            return np.concatenate(_grads)
        except:
            import pdb; pdb.set_trace()

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
        for layer in range(len(self.W) - 1):
            self.x[layer] = out.copy()

            W, b = self.W[layer], self.b[layer]
            h = out @ W.T + b

            if self.activation == 'relu':
                out = np.clip(h, a_min=0.0, a_max=np.inf)
            elif self.activation == 'sigmoid':
                out = 1 / (1 + np.exp(-h))
            elif self.activation is None:
                out = h
            else:
                raise NotImplementedError

            self.h[layer] = h.copy()
        
        self.x[-1] = out.copy()
        W, b = self.W[-1], self.b[-1]
        h = out @ W.T + b
        self.h[-1] = h.copy()

        return h

    def backward(self, dx):
        assert not any([x is None for x in self.x]), "must run a forward pass first"

        W = self.W[-1]
        x, h = self.x[-1], self.h[-1]
        dh = dx.copy()

        self.dW[-1] += (dh[..., np.newaxis] * x[..., np.newaxis, :]).mean(axis=0)
        self.db[-1] += dh.mean(axis=0)
        dout = dh @ W

        for layer in reversed(range(len(self.W) - 1)):
            W = self.W[layer]
            x, h = self.x[layer], self.h[layer]

            if self.activation == 'relu':
                dh = dout * (h > 0).astype(float)
            elif self.activation == 'sigmoid':
                sig = 1 / (1 + np.exp(-h))
                dh = dout * sig * (1 - sig)
            elif self.activation is None:
                dh = dout

            self.dW[layer] += (dh[..., np.newaxis] * x[..., np.newaxis, :]).mean(axis=0)
            self.db[layer] += dh.mean(axis=0)
            dout = dh @ W
