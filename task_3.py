from rl.Arena import Arena
from rl.agent.QLearner import QLearnerAgent
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt



randomAgent = RandomAgent()
qlearnerAgent = QLearnerAgent(optimizer='sgd')

# arena = Arena(agents=[randomAgent, qlearnerAgent])
arena = Arena(agents=[qlearnerAgent])

n_runs = 10

results = arena.sample(episodes=1000, runs=n_runs)

print(results)

fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(f"Averaged over {n_runs} runs")
ax1.set_xlabel('training time')
ax1.set_ylabel('Avg Reward per game')
# ax1.plot(results['RandomAgent'].avg_reward, label="Random Agent")
ax1.plot(results['QLearnerAgent'].avg_reward, label="QLearning Agent")
ax1.legend()

ax2.set_xlabel('training time')
ax2.set_ylabel('Number of moves per game')
# ax2.plot(results['RandomAgent'].avg_moves, label="Random Agent")
ax2.plot(results['QLearnerAgent'].avg_moves, label="QLearning Agent")
ax2.legend()

ax3.set_xlabel('training time')
ax3.set_ylabel('Gradient norm (L2)')
# ax2.plot(results['RandomAgent'].avg_moves, label="Random Agent")
ax3.plot(results['QLearnerAgent'].max_norms, label="QLearning Agent")
ax3.legend()

plt.show()

