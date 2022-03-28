import numpy as np
from matplotlib import rcParams

from rl.Arena import Arena, RunMetrics
from rl.agent.QLearner import QLearnerAgent
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt


np.random.seed(42)



gamma=0.85
eta = 0.0035
epsilon_0=0.2

activation = None
N_h = None
initialization = "uniform"

exp_name = "t6_B_check"


n_runs = 10
n_episodes = 10000
ma_length = 50

agent_1_lbl = "QLearning"
agent_2_lbl = "Deep QLearning"
agent_3_lbl = "SARSA"

agents_lbl = [agent_1_lbl,agent_2_lbl, agent_3_lbl]

qlearnerAgent = QLearnerAgent(optimizer='sgd', method="q-learning", activation=activation,initialization=initialization,gamma=gamma,
                              eta=eta, epsilon_0=epsilon_0, beta=0.00005, N_h=N_h, name=agent_1_lbl)

deepqlearnerAgent = QLearnerAgent(optimizer='sgd', method="q-learning", activation="sigmoid",initialization=initialization,gamma=gamma,
                              eta=eta, epsilon_0=epsilon_0, beta=0.00005, N_h=[64], name=agent_2_lbl)

sarsaAgent = QLearnerAgent(optimizer='sgd', method="sarsa", activation=activation,initialization=initialization,gamma=gamma,
                              eta=eta, epsilon_0=epsilon_0, beta=0.00005, N_h=N_h, name=agent_3_lbl)




agents = [qlearnerAgent, deepqlearnerAgent, sarsaAgent]


arena_baseline = Arena(agents=agents, ce_reward_check=False, ma_length=ma_length)
results_baseline = arena_baseline.sample(episodes=n_episodes, runs=n_runs)


arena_state = Arena(agents=agents, ce_reward_check=True, ma_length=ma_length)
results_state = arena_state.sample(episodes=n_episodes, runs=n_runs)

rel_results = {}

for agent in agents_lbl:
    br= results_baseline[agent]
    cr = results_state[agent]

    rel_reward = cr.avg_reward - br.avg_reward
    rel_moves = cr.avg_moves - br.avg_moves
    rel_wins = cr.avg_wins - br.avg_wins

    #rel_results[agent] = {"avg_reward": rel_reward, "avg_moves": rel_moves, "avg_wins": rel_wins}
    rel_results[agent] = RunMetrics(rel_reward, rel_moves, rel_wins, [], [])

    print(f"{agent} Baseline: Reward: {br.avg_reward[-1]} Wins: {br.avg_wins[-1]}")
    print(f"{agent} Modified: Reward: {cr.avg_reward[-1]} Wins: {cr.avg_wins[-1]}")
    print(f"{agent} Modified Improvement Reward: {rel_reward[-1]} Wins: {rel_wins[-1]}")


res_1 = rel_results[agent_1_lbl]
#res_2 = rel_results[agent_2_lbl]

params = {
   'font.family': 'serif',
   'xtick.labelsize': 'x-small',
   'ytick.labelsize': 'x-small',
   'figure.figsize': [4.5, 4.5]
   }
rcParams.update(params)


def plot_abs(metric, label):
    fig, ax = plt.subplots()
    # fig.suptitle(f"Reward per game")
    ax.set_xlabel('Steps')
    # ax.set_ylim([0,1])
    ax.set_ylabel(label)


    for agent in agents_lbl:
        res = results_baseline[agent]
        ax.plot(getattr(res,metric), label=f"{agent} Baseline")


    for agent in agents_lbl:
        res = results_state[agent]
        ax.plot(getattr(res,metric), label=f"{agent} Mod")

    ax.legend()
    fig.savefig(f'plots/{exp_name}_abs_{metric}.png', dpi=300)
    plt.show()


def plot_rel(metric, label):
    fig, ax = plt.subplots()
    # fig.suptitle(f"Reward per game")
    ax.set_xlabel('Steps')
    # ax.set_ylim([0,1])
    ax.set_ylabel(label)

    for agent in agents_lbl:
        res = rel_results[agent]
        ax.plot(getattr(res,metric), label=f"{agent} Modified")

    ax.legend()
    fig.savefig(f'plots/{exp_name}_rel_{metric}.png', dpi=300)
    plt.show()


plot_abs("avg_wins", f'Average win rate per {ma_length} games')
plot_abs("avg_moves", f'Average number of moves per game')


plot_rel("avg_wins", f'Average win rate per {ma_length} games')
plot_rel("avg_moves", f'Average number of moves per game')


#
# fig, ax = plt.subplots()
# #fig.suptitle(f"Reward per game")
# ax.set_xlabel('Steps')
# #ax.set_ylim([0,1])
# ax.set_ylabel('Average Reward per game')
#
# ax.plot(res_1.avg_reward, label=f"{agent_1_lbl}")
# #ax.plot(res_2["avg_reward"], label=f"{agent_2_lbl}")
# ax.legend()
# fig.savefig(f'plots/{exp_name}_reward.png', dpi=300)
# plt.show()

#
# fig, ax = plt.subplots()
# #fig.suptitle(f"Reward per game")
# ax.set_xlabel('Steps')
# #ax.set_ylim([0,1])
# ax.set_ylabel('Average Winrate per game')
#
# ax.plot(res_1.avg_wins, label=f"{agent_1_lbl}")
# #ax.plot(res_2["avg_reward"], label=f"{agent_2_lbl}")
# ax.legend()
# fig.savefig(f'plots/{exp_name}_wins.png', dpi=300)
# plt.show()

#
# fig, ax = plt.subplots()
# #fig.suptitle(f"Number of moves per game")
# ax.set_xlabel('Steps')
# ax.set_ylabel('Average number of moves per game')
# ax.plot(res_1.avg_moves, label=f"{agent_1_lbl}")
# #ax.plot(res_2["avg_moves"], label=f"{agent_2_lbl}")
# ax.legend()
# fig.savefig(f'plots/{exp_name}_moves.png', dpi=300)


#Debug plots
fig, (ax1,ax2,ax3, ax4,ax5) = plt.subplots(5)
fig.suptitle(f"Averaged over {n_runs} runs")
ax1.set_xlabel('training time')
ax1.set_ylabel('Avg Reward per game')

ax1.plot(res_1.avg_reward, label=f"{agent_1_lbl} Agent")
#ax1.plot(res_2.avg_reward, label=f"{agent_2_lbl} Agent")
ax1.legend()

ax2.set_xlabel('training time')
ax2.set_ylabel('Number of moves per game')

ax2.plot(res_1.avg_moves, label=f"{agent_1_lbl} Agent")
#ax2.plot(res_2.avg_moves, label=f"{agent_2_lbl} Agent")
ax2.legend()

# ax3.set_xlabel('training time')
# ax3.set_ylabel('Gradient norm (L2)')
#
# ax3.plot(res_1.max_norms, label=f"{agent_1_lbl} Agent")
# #ax3.plot(res_2.max_norms, label=f"{agent_2_lbl} Agent")
# ax3.legend()
#
# ax4.set_xlabel('training time')
# ax4.set_ylabel('QValues norm (L2)')
#
# ax4.plot(res_1.max_qvalues,  label=f"{agent_1_lbl} Agent")
# #ax4.plot(res_2.max_qvalues,  label=f"{agent_2_lbl} Agent")
# ax4.legend()


ax5.set_xlabel('training time')
ax5.set_ylabel('Avg Wins per N games')
ax5.plot(res_1.avg_wins, label=f"{agent_1_lbl} Agent")
#a51.plot(res_2.avg_reward, label=f"{agent_2_lbl} Agent")
ax5.legend()

plt.show()

