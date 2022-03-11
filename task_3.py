from rl.Arena import Arena
from rl.agent.QLearner import QLearnerAgent
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt



randomAgent = RandomAgent()
qlearnerAgent = QLearnerAgent()

arena = Arena(agents=[randomAgent, qlearnerAgent])

results = arena.sample(episodes=15)

print(results)

