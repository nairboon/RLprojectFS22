import random
import numpy as np

from rl.agent.Base import BaseAgent
from rl.network import MLP


class QLearnerAgent(BaseAgent):

    def __init__(self, **kwargs):
        name = kwargs.get("name", __class__.__name__)
        super().__init__(name)
        # configuration
        self.eps_0 = kwargs.get("epsilon_0", 0.2)
        self.beta = kwargs.get("beta", 0.05)
        self.gamma = kwargs.get("gamma", 0.85)
        self.eta = kwargs.get("eta", 1e-3)
        self.N_h = kwargs.get("N_h", None)
        self.activation = kwargs.get("activation", "relu")
        self.mlp_initialization = kwargs.get("initialization", "xavier")

        # learning method
        self.method = kwargs.get("method", "q-learning")
        assert self.method in ['q-learning', 'sarsa']

        # optimizer
        self.optimizer = kwargs.get("optimizer", "sgd")
        assert self.optimizer in ['sgd', 'rmsprop']

        if self.optimizer == 'rmsprop':
            self.eps_rms = kwargs.get("epsilon_rmsprop", 1e-8)
            self.alpha_rms = kwargs.get("alpha_rmsprop", 0.99)
            self.mom_rms = kwargs.get("momentum_rmsprop", 0.0)
            self._v_rms = 0.0
            self._b_rms = 0.0

    def init(self, n_episodes, shape_input, shape_output):
        self.QNet = MLP(shape_input, shape_output, self.N_h, self.activation, self.mlp_initialization)
        self.eps = self.eps_0

        if self.optimizer == 'rmsprop':
            self._v_rms = 0.0
            self._b_rms = 0.0

    def feedback(self, R, prev_X, X, A, allowed_A, it, episode_is_over=False):
        # adjust epsilon
        self.eps = self.eps_0 / (1 + self.beta * it)

        # temporal difference
        if not episode_is_over:
            next_Q = self.QNet(X[np.newaxis, ...])[0]
            next_Q[allowed_A.flatten() == 0] = -np.inf

            if self.method == 'q-learning':
                next_Q = np.max(next_Q, axis=-1)
            elif self.method == 'sarsa':
                # epsilon-greedy
                if random.random() < self.eps:
                    a, _ = np.where(allowed_A == 1)
                    a_agent = np.random.permutation(a)[0]
                    next_Q = next_Q[a_agent]
                else:
                    next_Q = np.max(next_Q, axis=-1)
            else:
                raise NotImplementedError

        # create bellman target
        self.QNet.zero_grad()
        Q = self.QNet(prev_X[np.newaxis, ...])[0]

        delta = np.zeros_like(Q)
        delta[A] = Q[A] - R
        if not episode_is_over:
            delta[A] -= self.gamma * next_Q
        delta = delta[np.newaxis, ...]

        # gradient descent
        self.QNet.backward(delta)
        grads = self.QNet.grads
        if self.optimizer == 'sgd':
            self.QNet.step(grads, eta=self.eta)
        elif self.optimizer == 'rmsprop':
            self._v_rms *= self.alpha_rms
            self._v_rms += (1 - self.alpha_rms) * grads ** 2
            self._b_rms *= self.mom_rms
            self._b_rms += grads / (np.sqrt(self._v_rms) + self.eps_rms)
            self.QNet.step(self._b_rms, eta=self.eta)
        else:
            raise NotImplementedError

        return Q

    def action(self, S, X, A):
        # get Q values - assuming non-batch
        Q = self.QNet(X[np.newaxis, ...])[0]

        # epsilon-greedy
        if random.random() < self.eps:
            a, _ = np.where(A == 1)
            a_agent = np.random.permutation(a)[0]
        else:
            Q[A.flatten() == 0] = -np.inf  # mask actions
            a_agent = np.argmax(Q, axis=-1)

        return a_agent
