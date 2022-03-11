

class BaseAgent:

    def __init__(self, name):
        self.name = name

    def reset(self):
        raise NotImplementedError

    def feedback(self, R, X):
        raise NotImplementedError

    def action(self,S,X,A):
        raise NotImplementedError