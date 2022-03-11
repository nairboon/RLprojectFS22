from typing import List
from dataclasses import dataclass

import numpy as np

from rl.agent.Base import BaseAgent
from rl.env.Chess_env import Chess_Env


@dataclass
class RunMetrics:
    avg_reward: float
    avg_moves: float

class Arena:


    def __init__(self, agents : List[BaseAgent]):
        self.agents = agents
        self.params = {"board_size" : 4}

    def sample(self, episodes):

        results = {}
        for agent in self.agents:
            res = self.run_agent(agent, episodes=episodes)
            results[agent.name] = res

        return results

    def run_agent(self, agent, episodes, runs=1):
        env = Chess_Env(self.params["board_size"])

        rewards = np.ndarray((runs,episodes))
        moves = np.ndarray((runs, episodes))

        for k in range(runs):
            for i in range(episodes):
                S, X, allowed_a = env.Initialise_game()
                Done = 0
                action_cnt = 0
                R = 0
                while Done==0:

                    selected_action = agent.action(S,X,allowed_a)
                    S, X, allowed_a, R, Done = env.OneStep(selected_action)
                    agent.feedback(R, X)
                    action_cnt += 1


                    if Done:
                        rewards[k,i] = R
                        moves[k,i] = action_cnt
                        break




        return RunMetrics(rewards.mean(axis=0), moves.mean(axis=0))

