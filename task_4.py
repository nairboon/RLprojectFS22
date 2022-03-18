import numpy as np
from datasets import tqdm
from matplotlib import rcParams

from rl.Arena import Arena
from rl.agent.QLearner import QLearnerAgent
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt
import matplotlib.tri as tri

np.random.seed(42)




eta = 0.0035
epsilon_0=0.2
#activation = "sigmoid"
activation = None

N_h = None
initialization = "uniform"

exp_name = "t4_grid"

agent_1_lbl = "QLearning"
agent_2_lbl = "SARSA"


N_episodes = 500
N_runs = 3
last_n = 25
def eval(gamma, beta):
    qlearnerAgent = QLearnerAgent(optimizer='sgd', method="q-learning", activation=activation,
                                  initialization=initialization, gamma=gamma,
                                  eta=eta, epsilon_0=epsilon_0, beta=beta, N_h=N_h, name=agent_1_lbl)
    arena = Arena(agents=[qlearnerAgent])
    results = arena.sample(episodes=N_episodes, runs=N_runs)
    res_1 = results[agent_1_lbl]
    rewards = res_1.avg_reward[-last_n:]
    avg_last_r = np.mean(rewards)
    return avg_last_r


N_gamma = 5
N_beta = 5

X = np.linspace(0,1,N_gamma)# gamma
Y = np.logspace(-3,-1,N_beta) # beta in log

Z = np.zeros((X.shape[0], Y.shape[0]))

for i,x in enumerate(X):
    for j,y in enumerate(Y):
        Z[i,j] = eval(x, y)




params = {
   'font.family': 'serif',
   'figure.figsize': [4.5, 4.5]
   }
#rcParams.update(params)

fig, ax = plt.subplots()
#fig.suptitle(f"Reward per game")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\beta$")
im = ax.imshow(Z,extent=[0,1,0,1])
fig.colorbar(im, ax=ax)
plt.show()
fig.savefig(f'plots/{exp_name}_imshow.png', dpi=300)

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