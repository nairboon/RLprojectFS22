from typing import List
from dataclasses import dataclass, field

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
    avg_reward: List[float]
    avg_moves: List[float]
    avg_wins: List[float]
    max_norms: float
    max_qvalues: float
    early_stopped_episodes: List[int] = field(default_factory=list)


class Arena:

    def __init__(self, agents : List[BaseAgent], **kwargs):
        self.agents = agents
        self.params = {"board_size" : 4}
        self.ma_length = kwargs.get("ma_length", 10)
        self.early_stop = kwargs.get("early_stop", False)
        self.early_stop_q_treshold = kwargs.get("early_stop_q_treshold", 1e3)
        self.arena_args = kwargs

    def sample(self, episodes, runs=1):

        results = {}
        for agent in self.agents:
            res = self.run_agent(agent, episodes=episodes,runs=runs)
            results[agent.name] = res

        return results

    def run_agent(self, agent, episodes, runs):
        print(f"run_agent {agent.name} with {self.arena_args}")
        env = Chess_Env(self.params["board_size"], **self.arena_args)

        early_stop_ids = []

        # per step
        wins = np.ndarray((runs, episodes))
        rewards = np.ndarray((runs, episodes))
        moves = np.ndarray((runs, episodes))
        max_norms = np.ndarray((runs, episodes))
        max_qvalues = np.ndarray((runs, episodes))

        with tqdm(total=runs * episodes) as pbar:
            for k in range(runs):
                early_stopped = False
                S, X, allowed_a = env.Initialise_game()
                agent.init(n_episodes=episodes, shape_input=X.shape[0], shape_output=allowed_a.shape[0])
                for i in range(episodes):
                    agent.reset()
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
                        if self.early_stop and max_qvalue > self.early_stop_q_treshold:
                            print(f"Q Early stop reached: {k}, {i}, {max_qvalue}")
                            early_stopped = True
                            early_stop_ids.append(i)
                            break
                        # if max_qvalue > 20:
                        #     print(f"Q Exp: {k}, {i}, {max_qvalue}")

                    if R >= env.reward_win:
                        wins[k,i] = 1
                    else:
                        wins[k,i] = 0

                    rewards[k,i] = R
                    moves[k,i] = action_cnt
                    max_norms[k,i] = max_norm
                    max_qvalues[k, i] = max_qvalue
                    pbar.update()

                    if early_stopped:
                        break

        metrics = RunMetrics(ma(rewards.mean(axis=0),self.ma_length), ma(moves.mean(axis=0),self.ma_length),
                             ma(wins.mean(axis=0), self.ma_length),
                             max_norms.mean(axis=0),max_qvalues.mean(axis=0))
        if self.early_stop:
            metrics.early_stopped_episodes = early_stop_ids

        return metrics
