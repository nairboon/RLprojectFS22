import random
import numpy as np

from rl.agent.Base import BaseAgent
from rl.network import MLP


class QLearnerAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(__class__.__name__)
        # configuration
        self.eps_0 = kwargs.get("epsilon_0", 0.2)
        self.beta = kwargs.get("beta", 0.00005)
        self.gamma = kwargs.get("gamma", 0.85)
        self.eta = kwargs.get("eta", 0.0035)
        self.N_h = kwargs.get("N_h", 200)
        self.activation = kwargs.get("activation", "sigmoid")

    def init(self, n_episodes, shape_input, shape_output):
        self.QNet = MLP(shape_input, shape_output, self.N_h, self.activation)
        self.eps = self.eps_0
        self.prev_Q = None

    def feedback(self, R, prev_X, X, A, allowed_A, it, episode_is_over=False):
        # adjust epsilon
        self.eps = self.eps_0 / (1 + self.beta * it)

        # temporal difference
        if not episode_is_over:
            next_Q = self.QNet(X[np.newaxis, ...])[0]
            next_Q[allowed_A.flatten() == 0] = -np.inf
            next_Q = np.max(next_Q, axis=-1)

        self.QNet.zero_grad()
        Q = self.QNet(prev_X[np.newaxis, ...])[0]

        target = np.zeros_like(Q)
        target[A] = R - Q[A]
        if not episode_is_over:
            target[A] += self.gamma * next_Q
        delta = Q - target

        self.QNet.backward(delta)
        grads = self.QNet.grads
        self.QNet.step(grads, eta=self.eta)

    def action(self, S, X, A):
        # get Q values - assuming non-batch
        Q = self.QNet(X[np.newaxis, ...])[0]
        self.prev_Q = Q.copy()

        # epsilon-greedy
        if random.random() < self.eps:
            a, _ = np.where(A == 1)
            a_agent = np.random.permutation(a)[0]
        else:
            Q[A.flatten() == 0] = -np.inf  # mask actions
            a_agent = np.argmax(Q, axis=-1)

        return a_agent
