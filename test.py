import gymnasium as gym
import torch
import numpy as np
from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=0)

weights = torch.load('final_network.pth', weights_only=True)
agent.qnetwork_local.load_state_dict(weights)

agent.qnetwork_local.eval()

num_episodes = 100
scores = []

env = gym.make('LunarLander-v3')

for i in range(1, num_episodes + 1):
    
    if i == num_episodes:
        env.close() 
        env = gym.make('LunarLander-v3', render_mode='human')

    state, _ = env.reset()
    score = 0
    done = False
    
    while not done:
        action = agent.act(state, is_eval=True)
        
        state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
        
    scores.append(score)
    
    if i % 10 == 0 and i != num_episodes:
        print(f"{i}/{num_episodes} episode. Mean score {np.mean(scores):.2f}")

env.close()

print(f"Mean reward: {np.mean(scores):.2f}")
print(f"max reward {np.max(scores):.2f}")
print(f"Min reward {np.min(scores):.2f}")
