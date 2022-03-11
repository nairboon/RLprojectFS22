import numpy as np

from rl.agent.Base import BaseAgent


class QLearnerAgent(BaseAgent):

    def __init__(self):
        super().__init__(__class__.__name__)


    def action(self,S,X,A):
        a,_=np.where(A==1)
        a_agent=np.random.permutation(a)[0]
        return a_agent