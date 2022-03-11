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

    def run_agent(self, agent, episodes):
        env = Chess_Env(self.params["board_size"])
        rewards = []
        moves = []
        for i in range(episodes):
            S, X, allowed_a = env.Initialise_game()
            Done = 0
            action_cnt = 0
            R = 0
            while Done==0:

                selected_action = agent.action(S,X,allowed_a)
                S, X, allowed_a, R, Done = env.OneStep(selected_action)
                action_cnt += 1

                if Done:
                    break

            rewards.append(R)
            moves.append(action_cnt)

        return RunMetrics(np.mean(rewards), np.mean(moves))

