from typing import List
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from rl.agent.Base import BaseAgent
from rl.env.Chess_env import Chess_Env

def ma(arr, N):
    res = np.empty(len(arr))
    res[0] = arr[0]

    for i in range(N):
        res[i] = np.sum(arr[:i+1])/(i+1)

    for i in range(N, len(arr)):
        res[i] = np.sum(arr[i-N+1:i+1])/N

    return res


@dataclass
class RunMetrics:
    avg_reward: float
    avg_moves: float
    max_norms: float
    max_qvalues: float


class Arena:

    def __init__(self, agents : List[BaseAgent], **kwargs):
        self.agents = agents
        self.params = {"board_size" : 4}
        self.ma_length = kwargs.get("ma_length", 16)

    def sample(self, episodes, runs=1):

        results = {}
        for agent in self.agents:
            res = self.run_agent(agent, episodes=episodes,runs=runs)
            results[agent.name] = res

        return results

    def run_agent(self, agent, episodes, runs):
        env = Chess_Env(self.params["board_size"])

        # per step
        rewards = np.ndarray((runs, episodes))
        moves = np.ndarray((runs, episodes))
        max_norms = np.ndarray((runs, episodes))
        max_qvalues = np.ndarray((runs, episodes))

        with tqdm(total=runs * episodes) as pbar:
            for k in range(runs):
                S, X, allowed_a = env.Initialise_game()
                agent.init(n_episodes=episodes, shape_input=X.shape[0], shape_output=allowed_a.shape[0])
                for i in range(episodes):
                    S, X, allowed_a = env.Initialise_game()
                    Done = 0
                    action_cnt = 0
                    R = 0
                    max_norm = 0
                    max_qvalue = 0
                    while Done==0:
                        prev_X = X.copy()
                        selected_action = agent.action(S,X,allowed_a)
                        S, X, allowed_a, R, Done = env.OneStep(selected_action)
                        Q = agent.feedback(R, prev_X, X, selected_action, allowed_a, i,Done==1)
                        action_cnt += 1

                        if not np.isfinite(Q).all():
                            print(f"Exp: {k}, {i}, Qvalues fucked")

                        max_norm = max(max_norm, np.linalg.norm(agent.QNet.grads))
                        # if max_norm > 20:
                        #     print(f"Exp: {k}, {i}, {max_norm}")

                        max_qvalue = max(max_qvalue, np.linalg.norm(Q))
                        # if max_qvalue > 20:
                        #     print(f"Q Exp: {k}, {i}, {max_qvalue}")


                    rewards[k,i] = R
                    moves[k,i] = action_cnt
                    max_norms[k,i] = max_norm
                    max_qvalues[k, i] = max_qvalue
                    pbar.update()


        return RunMetrics(ma(rewards.mean(axis=0),self.ma_length), ma(moves.mean(axis=0),self.ma_length), max_norms.mean(axis=0),max_qvalues.mean(axis=0))
