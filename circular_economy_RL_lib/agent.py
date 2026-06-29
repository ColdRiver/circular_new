import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from config import stages

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    """
    Stabilizes initial PPO policy gradients using Orthogonal initialization.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=128, min_val=0.01, max_val=100.0):
        super(Actor, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.layer1 = nn.Linear(n_observations, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, n_actions)
        
        self.apply(orthogonal_init)
        orthogonal_init(self.layer3, gain=0.01)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Unbounded output allowing standard continuous Gaussian policy optimization
        return self.layer3(x)

class Critic(nn.Module):
    def __init__(self, n_observations, hidden_dims=128):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, 1)
        self.apply(orthogonal_init)
        orthogonal_init(self.layer3, gain=0.01)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class OptimalFollowerValueEstimator(nn.Module):
    """
    Auxiliary Value Network tracking optimal follower returns V*(phi, s_lower)
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(OptimalFollowerValueEstimator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.apply(orthogonal_init)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Unbounded linear output allowing exact regression to the scaled follower returns
        return self.layer3(x)

    def update(self, state_phi, target_returns):
        self.optimizer.zero_grad()
        predictions = self.forward(state_phi).squeeze()
        
        # Squeeze and detach targets to isolate graph execution
        targets = target_returns.clone().detach().squeeze() if isinstance(target_returns, torch.Tensor) else torch.tensor(target_returns, dtype=torch.float32).squeeze()
        
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
        return loss.item()

class PPOAgent:
    def __init__(self, n_observations, n_actions, chkpt_dir, hidden_dims=128, lr=0.01, min_val=0.01, max_val=100.0):
        self.actor = Actor(n_observations, n_actions, hidden_dims, min_val, max_val)
        self.critic = Critic(n_observations, hidden_dims)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Learnable standard deviation parameter allowing variance decay as confidence rises
        self.log_std = nn.Parameter(torch.zeros(n_actions) - 0.5)
        self.actor_optim.add_param_group({'params': self.log_std, 'lr': lr})
        
        self.chkpt_dir = chkpt_dir
        self.clip = 0.2
        self.max_grad_norm = 0.5

    def get_action(self, obs):
        mean = self.actor(obs)
        std = torch.clamp(torch.exp(self.log_std), min=1e-3, max=10.0)
        cov_mat = torch.diag(std ** 2)
        
        dist = MultivariateNormal(mean, cov_mat)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action)
        
        # Apply Sigmoid bounding transformation to the sampled raw action right before execution
        action = torch.sigmoid(raw_action) * (self.actor.max_val - self.actor.min_val) + self.actor.min_val
        
        return action.detach().numpy(), raw_action.detach().numpy(), log_prob.detach().numpy()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        std = torch.clamp(torch.exp(self.log_std), min=1e-3, max=10.0)
        cov_mat = torch.diag(std ** 2)
        
        dist = MultivariateNormal(mean, cov_mat)
        # Evaluates log-probabilities of raw actions in the unconstrained space
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()
        return V, log_probs, entropy

    def learn(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, n_itr, entropy_coef=0.01):
        V, _, _ = self.evaluate(batch_obs, batch_acts)
        
        # Column-wise (agent-specific) Advantage Normalization preventing gradient suppression
        A_k = batch_rtgs - V.detach()
        if A_k.dim() > 1:
            mean = A_k.mean(dim=0, keepdim=True)
            std = A_k.std(dim=0, keepdim=True)
            A_k = (A_k - mean) / (std + 1e-10)
        else:
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        a_loss, c_loss = 0.0, 0.0
        for _ in range(n_itr):
            bs = batch_obs.shape[0]
            indices = torch.randperm(bs)
            
            b_obs = batch_obs[indices]
            b_acts = batch_acts[indices]
            b_log_probs = batch_log_probs[indices]
            
            b_rtgs = batch_rtgs[indices]
            if b_rtgs.dim() > 1 and b_rtgs.shape[-1] == 1:
                b_rtgs = b_rtgs.squeeze(-1)
                
            b_A_k = A_k[indices]

            V, curr_log_probs, entropy = self.evaluate(b_obs, b_acts)
            ratios = torch.exp(curr_log_probs - b_log_probs)
            surr1 = ratios * b_A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * b_A_k
            
            # Entropy bonus preventing premature local minima convergence
            actor_loss = (-torch.min(surr1, surr2)).mean() - entropy_coef * entropy.mean()
            critic_loss = nn.MSELoss()(V, b_rtgs)
            
            self.actor_optim.zero_grad()
            actor_loss.backward()  # Normal backward pass releasing activation graphs
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()

            a_loss += actor_loss.item()
            c_loss += critic_loss.item()
        
        return a_loss / float(n_itr), c_loss / float(n_itr)
