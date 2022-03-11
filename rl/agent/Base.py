

class BaseAgent:

    def __init__(self, name):
        self.name = name

    def init(self, n_episodes, shape_input, shape_output):
        pass


    def feedback(self, R, prev_X, X, A, allowed_A, it, episode_is_over=False):
        pass

    def action(self,S,X,A):
        raise NotImplementedError