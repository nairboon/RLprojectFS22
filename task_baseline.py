from rl.Arena import Arena
from rl.agent.Random import RandomAgent
import matplotlib.pyplot as plt



randomAgent = RandomAgent()

arena = Arena(agents=[randomAgent])

results = arena.sample(episodes=15)

print(results)

