import numpy as np
from matplotlib import rcParams

from rl.Arena import Arena
from rl.agent.QLearner import QLearnerAgent
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt


np.random.seed(42)



gamma=0.85

eta = 0.0035
epsilon_0=0.2
activation = "sigmoid"
activation = None

N_h = [200]
initialization = "uniform"


ma_length = 25


exp_name = "t3_std_semigrads"

agent_1_lbl = "QLearning (Linear)"
agent_2_lbl = "QLearning (Deep)"
agent_3_lbl = "SARSA"

qlearnerAgent = QLearnerAgent(optimizer='sgd', method="q-learning", activation=activation,initialization=initialization,gamma=gamma,
                              eta=eta, epsilon_0=epsilon_0, beta=0.00005, N_h=N_h, name=agent_1_lbl)

deepqlearnerAgent = QLearnerAgent(optimizer='sgd', method="q-learning", activation="relu",initialization="uniform",gamma=gamma,
                              eta=eta, epsilon_0=epsilon_0, beta=0.00005, N_h=[200], name=agent_2_lbl)


sarsaAgent = QLearnerAgent(optimizer='sgd', method="sarsa", activation=activation,initialization=initialization,gamma=gamma,
                              eta=eta, epsilon_0=epsilon_0, beta=0.00005, N_h=N_h, name=agent_3_lbl)



arena = Arena(agents=[sarsaAgent, deepqlearnerAgent, qlearnerAgent],ma_length=ma_length)


n_runs = 25

results = arena.sample(episodes=30000, runs=n_runs)

res_1 = results[agent_1_lbl]
res_2 = results[agent_2_lbl]
res_3 = results[agent_3_lbl]

params = {
   'font.family': 'serif',
   'xtick.labelsize': 'x-small',
   'ytick.labelsize': 'x-small',
   'figure.figsize': [4.5, 4.5]
   }
rcParams.update(params)

fig, ax = plt.subplots()
#fig.suptitle(f"Reward per game")
ax.set_xlabel('Episodes')
ax.set_ylim([0,1])
ax.set_ylabel('Average Reward per game')

ax.plot(res_1.avg_reward, label=f"{agent_1_lbl}")
ax.plot(res_2.avg_reward, label=f"{agent_2_lbl}")
ax.plot(res_3.avg_reward, label=f"{agent_3_lbl}")
ax.legend()
fig.savefig(f'plots/{exp_name}_reward.png', dpi=300)


fig, ax = plt.subplots()
#fig.suptitle(f"Number of moves per game")
ax.set_xlabel('Episodes')
ax.set_ylabel('Average number of moves per game')
ax.plot(res_1.avg_moves, label=f"{agent_1_lbl}")
ax.plot(res_2.avg_moves, label=f"{agent_2_lbl}")
ax.plot(res_3.avg_moves, label=f"{agent_3_lbl}")
ax.legend()
fig.savefig(f'plots/{exp_name}_moves.png', dpi=300)


# Debug plots
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4)
fig.suptitle(f"Averaged over {n_runs} runs")
ax1.set_xlabel('training time')
ax1.set_ylabel('Avg Reward per game')

ax1.plot(res_1.avg_reward, label=f"{agent_1_lbl} Agent")
ax1.plot(res_2.avg_reward, label=f"{agent_2_lbl} Agent")
ax1.legend()

ax2.set_xlabel('training time')
ax2.set_ylabel('Number of moves per game')

ax2.plot(res_1.avg_moves, label=f"{agent_1_lbl} Agent")
ax2.plot(res_2.avg_moves, label=f"{agent_2_lbl} Agent")
ax2.legend()

ax3.set_xlabel('training time')
ax3.set_ylabel('Gradient norm (L2)')

ax3.plot(res_1.max_norms, label=f"{agent_1_lbl} Agent")
ax3.plot(res_2.max_norms, label=f"{agent_2_lbl} Agent")
ax3.legend()

ax4.set_xlabel('training time')
ax4.set_ylabel('QValues norm (L2)')

ax4.plot(res_1.max_qvalues,  label=f"{agent_1_lbl} Agent")
ax4.plot(res_2.max_qvalues,  label=f"{agent_2_lbl} Agent")
ax4.legend()

plt.show()

