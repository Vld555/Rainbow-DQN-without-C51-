import gymnasium as gym
import torch
import numpy as np
from dqn_agent import Agent
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, ResizeObservation

N_ATOMS=51

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class CropObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shape[0] - 12, shape[1], shape[2]),
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation[:-12, :, :]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make('CarRacing-v3', continuous=False, render_mode='human')
env = SkipFrame(env)
env = CropObservation(env)
env = GrayscaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4)
state_size = env.observation_space.shape
action_size = env.action_space.n

agent = Agent(state_size=state_size, action_size=action_size,
              n_atoms=N_ATOMS, seed=1)

weights = torch.load('/Users/vladharcenko/Desktop/ML/РСК/task5/rainbow_weights',
                     map_location=device, weights_only=True)
agent.qnetwork_local.load_state_dict(weights)

agent.qnetwork_local.eval()

num_episodes = 5
scores = []


for i in range(1, num_episodes + 1):
    state, _ = env.reset()
    score = 0
    done = False

    while not done:
        action = agent.act(state, is_eval=True)

        state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated

    scores.append(score)
    print(f"{i}/{num_episodes} episode. Score {score:.2f}")

env.close()

print(f"Mean reward: {np.mean(scores):.2f}")
print(f"max reward {np.max(scores):.2f}")
print(f"Min reward {np.min(scores):.2f}")
