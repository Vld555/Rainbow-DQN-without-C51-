import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import DuelingNoisyQNetwork
from tree import SumTree

BUFFER_SIZE = int(5e5)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 5e-3
LR = 1e-4
UPDATE_EVERY = 4
N_ATOMS = 51

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    def __init__(self, state_size, action_size, n_atoms, seed, n_step=3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.n_atoms = n_atoms
        self.vmin = -100
        self.vmax = 1000
        self.dz = (self.vmax-self.vmin) / (self.n_atoms - 1)
        self.supports = torch.linspace(self.vmin, self.vmax, self.n_atoms, device=device)

        # Q-Network
        self.qnetwork_local = DuelingNoisyQNetwork(
            state_size, action_size, seed, n_atoms=n_atoms).to(device)
        self.qnetwork_target = DuelingNoisyQNetwork(
            state_size, action_size, seed, n_atoms=n_atoms).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=50, min_lr=1e-6)
        self.memory = PrioritizingExperienceBuffer(
            self.state_size, self.action_size, BUFFER_SIZE, beta=0.4)
        self.t_step = 0

    def _get_n_step(self):
        state, action = self.n_step_buffer[0][:2]
        reward, next_state, term = self.n_step_buffer[-1][2:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, s_next, t = transition[2:]
            reward = r + GAMMA*reward*(1-t)
            if t:
                next_state, term = s_next, t

        return state, action, reward, next_state, term

    def step(self, state, action, reward, next_state, done, terminated):
        # multi-step block
        self.n_step_buffer.append((state, action, reward, next_state, terminated))
        if len(self.n_step_buffer) == self.n_step and not done:
            s, a, r, s_next, term = self._get_n_step()
            self.memory.add(s, a, r, s_next, term)

        if done:
            while len(self.n_step_buffer) > 0:
                s, a, r, s_next, term = self._get_n_step()
                self.memory.add(s, a, r, s_next, term)
                self.n_step_buffer.popleft()

        # end multi step block
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if self.memory.real_size > 10000:
                experiences, weights, tree_idxs = self.memory.sample(
                    BATCH_SIZE)
                loss=self.learn(experiences, weights, tree_idxs, GAMMA)
                return loss
        return None

    def act(self, state, is_eval=False):  # not using e-greedy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if is_eval:
            self.qnetwork_local.eval()
        else:
            self.qnetwork_local.train()


        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            probs = F.softmax(action_values, dim=-1)
        self.qnetwork_local.train()
        # return np.argmax(action_values.cpu().data.numpy())
        q_values = (probs * self.supports).sum(dim=2) # (1, action_size)
        return q_values.squeeze(0).argmax().item()

    def learn(self, experiences, weights, tree_idxs, gamma):
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        
        states, actions, rewards, next_states, dones = experiences
        dz = (self.vmax - self.vmin) / (self.n_atoms - 1)
        batch_size = states.size(0)
        
        with torch.no_grad():
            next_logits_local = self.qnetwork_local(next_states)
            next_probs_local = F.softmax(next_logits_local, dim=-1)
            next_action_values = (next_probs_local * self.supports).sum(dim=2)
            
            best_actions = next_action_values.argmax(dim=1).unsqueeze(1).unsqueeze(2)
            best_actions = best_actions.expand(-1, -1, self.n_atoms)
            
            next_logits_target = self.qnetwork_target(next_states)
            next_probs_target = F.softmax(next_logits_target, dim=-1)
            p_next = next_probs_target.gather(1, best_actions).squeeze(1)

            gamma_n = gamma ** self.n_step
            Tz = rewards + gamma_n * self.supports * (1 - dones)
            Tz = Tz.clamp(self.vmin, self.vmax)
            b = (Tz - self.vmin) / dz
            l = b.floor().long()
            u = b.ceil().long()

            dl = u.float() - b
            du = b - l.float()
            dl[(l == u)] = 1.0
            du[(l == u)] = 0.0

            m = torch.zeros(batch_size, self.n_atoms).to(device)

            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.n_atoms).to(device)
            m.view(-1).index_add_(0, (l + offset).view(-1), (p_next * dl).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (p_next * du).view(-1))

        logits_local = self.qnetwork_local(states)
        log_probs = F.log_softmax(logits_local, dim=-1)
        log_p = log_probs.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)
        
        loss = -(m * log_p).sum(dim=1)
        
        elementwise_loss = loss
        loss = (loss * weights.squeeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        
        self.optimizer.step()
        
        self.memory.update_priorities(tree_idxs, elementwise_loss.detach().cpu().numpy())
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        '''
        θ_target = τ*θ_local + (1-τ)*θ_target
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizingExperienceBuffer:
    def __init__(self, state_shape, action_size, buffer_size, eps=0.01, alpha=0.4, beta=0.1, beta_incr=1e-5):
        self.tree = SumTree(size=buffer_size)

        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps
        self.beta_incr = beta_incr

        self.state = torch.empty(buffer_size, *state_shape, dtype=torch.uint8)
        self.action = torch.empty(buffer_size, 1, dtype=torch.long)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(
            buffer_size, *state_shape, dtype=torch.uint8)
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
            while True:
                cumsum = random.uniform(a, b)
                tree_idx, priority, sample_idx = self.tree.get(cumsum)
                if sample_idx is not None: break
            priorities[i] = float(priority)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = (weights / weights.max()).to(device)

        batch = (
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].to(device),
            self.reward[sample_idxs].unsqueeze(1).to(device),
            self.next_state[sample_idxs].to(device),
            self.done[sample_idxs].unsqueeze(1).to(device)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

# class ReplayBuffer:
#     def __init__(self, action_size, buffer_size, batch_size, seed):
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.seed = random.seed(seed)
#         self.experience = namedtuple('Experience', field_names=[
#                                      'state', 'action', 'reward', 'next_state', 'done'])

#     def add(self, state, action, reward, next_state, done):
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)

#     def sample(self):
#         experiences = random.sample(self.memory, k=self.batch_size)
#         states = torch.from_numpy(
#             np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(
#             np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(
#             np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack(
#             [e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(
#             np.vstack([e.done for e in experiences if e is not None])).float().to(device)

#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         return len(self.memory)
