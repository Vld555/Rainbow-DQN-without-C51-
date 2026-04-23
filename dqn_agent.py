import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork, DuelingQNetwork, DuelingNoisyQNetwork
from tree import SumTree
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    def __init__(self, state_size, action_size, seed, n_step=3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        # Q-Network
        self.qnetwork_local = DuelingNoisyQNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingNoisyQNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = PrioritizingExperienceBuffer(
            self.state_size, self.action_size, BUFFER_SIZE)
        self.t_step = 0

    def _get_n_step(self):
        state, action = self.n_step_buffer[0][:2]
        reward, next_state, done = self.n_step_buffer[-1][2:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, s_next, d = transition[2:]
            reward = r + GAMMA*reward*(1-d)
            if d:
                next_state, done = s_next, d

        return state, action, reward, next_state, done

    def step(self, state, action, reward, next_state, done):
        # multi-step block
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            s, a, r, s_next, d = self._get_n_step()
            self.memory.add(s, a, r, s_next, d)

        if done:
            while len(self.n_step_buffer) > 0:
                s, a, r, s_next, d = self._get_n_step()
                self.memory.add(s, a, r, s_next, d)
                self.n_step_buffer.popleft()

        # end multi step block
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if self.memory.real_size > BATCH_SIZE:
                experiences, weights, tree_idxs = self.memory.sample(
                    BATCH_SIZE)
                self.learn(experiences, weights, tree_idxs, GAMMA)

    def act(self, state, is_eval=False):  # not using e-greedy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if is_eval:
            self.qnetwork_local.eval()
        else:
            self.qnetwork_local.train()
            self.qnetwork_local.reset_noise()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, weights, tree_idxs, gamma):
        '''
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones = experiences
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        '''
        рассчитать TD-ошибку (для обновления дерева)
        применить веса Importance Sampling (IS) к Loss-функции
        обновить приоритеты.
        '''
        # classic dqn
        # q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # double dqn
        q_local_best_actions = self.qnetwork_local(next_states).detach().max(
            1)[1].unsqueeze(1)  # выбираем лучшее действие по мнению local
        q_targets_next = self.qnetwork_target(next_states).detach().gather(
            1, q_local_best_actions)  # оцениваем качество выбранных действй

        # multi-step gamma
        gamma_n = gamma ** self.n_step
        q_target = rewards + gamma_n * q_targets_next * (1-dones)

        q_expected = self.qnetwork_local(states).gather(1, actions)

        td_error = torch.abs(q_expected - q_target).detach()

        loss = torch.mean(((q_expected-q_target)**2) * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(
            tree_idxs, td_error.cpu().numpy().squeeze())
        # TODO

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''
        θ_target = τ*θ_local + (1-τ)*θ_target
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizingExperienceBuffer:
    def __init__(self, state_size, action_size, buffer_size, eps=0.01, alpha=0.1, beta=0.1, beta_incr=0.001):
        self.tree = SumTree(size=buffer_size)

        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps
        self.beta_incr = beta_incr

        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, 1, dtype=torch.long)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(
            buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.float)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, self.count)

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size
        self.beta = min(1.0, self.beta+self.beta_incr)
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].to(device),
            self.reward[sample_idxs].unsqueeze(1),
            self.next_state[sample_idxs],
            self.done[sample_idxs].unsqueeze(1)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple('Experience', field_names=[
                                     'state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
