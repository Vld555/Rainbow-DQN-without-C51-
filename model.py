import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NoisyLinear(nn.Module):
    # y=(mu_w + sigma_w * epsolon_w)x + mu_b + sigma_b * epsilon_b
    def __init__(self, in_features, out_features, std_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        mu_range = 1 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(std_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(std_init / math.sqrt(out_features))

        self.reset_noise()
    
    def reset_noise(self,):
        # f(x) = sign(x) * sqrt(|x|)
        f = lambda x: x.sign().mul_(x.abs().sqrt())
        eps_in = f(torch.randn(self.in_features,device=device))
        eps_out = f(torch.randn(self.out_features, device=device))

        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)
        
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma*self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma*self.bias_epsilon
        else: 
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)



class QNetwork(nn.Module):
    def __init__(self, state_size,  action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

        self.act = nn.ReLU()
        # code here

    def forward(self, state):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(state)))))


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU()
        )
        self.values = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantages = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        # code here

    def forward(self, state):
        y = self.feature_extractor(state)
        v = self.values(y)
        a = self.advantages(y)
        Q = v + (a - a.mean(dim=1, keepdim=True))
        return Q
    
class DuelingNoisyQNetwork(nn.Module):
    def __init__(self, state_shape, action_size, seed, n_atoms):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.n_atoms = n_atoms
        self.action_size = action_size
        channel_in, h,w = state_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.values = nn.Sequential(
            NoisyLinear(3136, 256),
            nn.ReLU(),
            NoisyLinear(256, n_atoms)
        )

        self.advantages = nn.Sequential(
            NoisyLinear(3136, 256), 
            nn.ReLU(),
            NoisyLinear(256, action_size * n_atoms)
        )
        
    def forward(self, state):
        state = state.float()/255.0
        y = self.encoder(state)
        v = self.values(y).reshape(-1, 1, self.n_atoms) # (batch_size, 1, 51)
        a = self.advantages(y).reshape(-1, self.action_size, self.n_atoms) # (batch_size, action_size, 51)
        Q = v + (a - a.mean(dim=1, keepdim=True)) # (batch_size, action_size, 51) - логиты распределения для каждого действия
        return Q

    def reset_noise(self):
        self.values[0].reset_noise()
        self.values[2].reset_noise()
        self.advantages[0].reset_noise()
        self.advantages[2].reset_noise()