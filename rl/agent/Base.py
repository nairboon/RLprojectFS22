

class BaseAgent:

    def __init__(self, name):
        self.name = name

    def action(self,S,X,A):
        raise NotImplementedError