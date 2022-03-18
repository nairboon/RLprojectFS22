import numpy as np
from datasets import tqdm
from matplotlib import rcParams

from rl.Arena import Arena
from rl.agent.QLearner import QLearnerAgent
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt
import matplotlib.tri as tri

np.random.seed(42)





eta = 0.25
epsilon_0=0.2
#activation = "sigmoid"
activation = None

N_h = None
#N_h = [36]
initialization = "uniform"

exp_name = "t7_unstable_single"

agent_1_lbl = "Q(none)"
agent_2_lbl = "Q(relu)"
agent_3_lbl = "Q(sigmoid)"

agents = [agent_1_lbl]

N_episodes = 5000
N_runs = 10

def eval_algos(gamma, eta, beta=0.00005):
    print(gamma,eta)
    qlearnerAgent = QLearnerAgent(optimizer='sgd', method="q-learning", activation=activation,
                                  initialization=initialization, gamma=gamma,
                                  eta=eta, epsilon_0=epsilon_0, beta=beta, N_h=N_h, name=agent_1_lbl)

    arena = Arena(agents=[qlearnerAgent], early_stop=True)
    results = arena.sample(episodes=N_episodes, runs=N_runs)
    res_1 = results[agent_1_lbl]

    return results

#res = eval_algos(0.98)


def calc_survival(N_runs, N_episodes, early):
    surviving = np.zeros(N_episodes)
    deadnt = N_runs - len(early)
    surviving[:] += deadnt
    for i in early:
        surviving[:i] += 1

    S = surviving / N_runs
    return S



from sklearn.model_selection import ParameterGrid
param_grid = {'gamma': [0.90, 0.99], 'eta': [0.25, 0.3, 0.4]}
pg = ParameterGrid(param_grid)


curves = []
labels = []
for param in pg:
    print(param)
    res = eval_algos(**param)
    for agent in agents:
        early = res[agent].early_stopped_episodes
        S = calc_survival(N_runs, N_episodes, early)
        curves.append(S)
        labels.append(f"{agent} ($\gamma={param['gamma']}$, $\eta={param['eta']}$)")


print(labels)

#early = res[agent_1_lbl].early_stopped_episodes

# surviving = np.zeros(N_episodes)
# deadnt = N_runs - len(early)
# surviving[:] += deadnt
# for i in early:
#     surviving[:i] += 1
#
# S = surviving/N_runs
#plt.plot(S[:150])
#plt.show()

for agent in agents:
    fig, ax = plt.subplots()
    #fig.suptitle(f"Reward per game")
    ax.set_xlabel('episodes t')
    ax.set_ylim([0,1.01])
    ax.set_ylabel('numerical survival $S(t)$')

    for x, label in zip(curves, labels):
        if agent in label:
            ax.plot(x, label=label)
    ax.legend()
    plt.show()
    fig.savefig(f'plots/{exp_name}_survival_{agent}.png', dpi=300)


# N_gamma = 5
# N_beta = 5
#
# X = np.linspace(0,1,N_gamma)# gamma
# Y = np.logspace(-3,-1,N_beta) # beta in log
#
# Z = np.zeros((X.shape[0], Y.shape[0]))
#
# for i,x in enumerate(X):
#     for j,y in enumerate(Y):
#         Z[i,j] = eval(x, y)
#
#
#
#
# params = {
#    'font.family': 'serif',
#    'figure.figsize': [4.5, 4.5]
#    }
# #rcParams.update(params)
#
# fig, ax = plt.subplots()
# #fig.suptitle(f"Reward per game")
# ax.set_xlabel(r"$\gamma$")
# ax.set_ylabel(r"$\beta$")
# im = ax.imshow(Z,extent=[0,1,0,1])
# fig.colorbar(im, ax=ax)
# plt.show()
# fig.savefig(f'plots/{exp_name}_imshow.png', dpi=300)

#
#
#
# fig, ax = plt.subplots()
#
# ngridx = 5
# ngridy = 5
#
# xi = np.linspace(0, 1, ngridx)
# yi = np.linspace(-1, 1, ngridy)
#
# X, Y = np.meshgrid(X,Y)
#
# # Perform linear interpolation of the data (x,y)
# # on a grid defined by (xi,yi)
# triang = tri.Triangulation(X, Y)
# interpolator = tri.LinearTriInterpolator(triang, Z)
# Xi, Yi = np.meshgrid(xi, yi)
# zi = interpolator(Xi, Yi)
#
# from scipy.interpolate import griddata
# zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear')
#
#
#
#
#
# fig, ax2 = plt.subplots()
#
# #ax2.tricontour(X, Y, Z, levels=14, linewidths=0.5, colors='k')
# #cntr2 = ax2.tricontourf(X, Y, Z, levels=14, cmap="RdBu_r")
#
# #fig.colorbar(cntr2, ax=ax2)
#
# ax2.plot(X1, X2, 'ko', ms=3)
# #ax2.set(xlim=(-2, 2), ylim=(-2, 2))
# ax2.set_title('tricontour (%d points)')
#
# plt.subplots_adjust(hspace=0.5)
# plt.show()