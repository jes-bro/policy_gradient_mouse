'''
'''
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torchdiffeq import odeint
from env import Env

# Max number of episode steps
T_MAX = 100

'''
'''
class MouseAgent(nn.Module):
    '''
    '''
    def __init__(self, input_dim, hidden_dim, env, device):
        super().__init__() 
        self.env = env
        self.device = device
        self.accel_dim = 2
        covariance_dim = 3
        self.linear1 = nn.Linear(input_dim, hidden_dim, device=self.device).float() 
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, device=self.device).float() 
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, device=self.device).float() 
        # Layer that outputs the mean (2D) of a multivariate distribution
        self.action_mean = nn.Linear(hidden_dim, self.accel_dim, device=self.device).float() 
        # Layer that outputs the covariance matrix describing a multivariate distribution
        self.covariance_matrix = nn.Linear(hidden_dim, covariance_dim, device=self.device).float() 

        dt = 0.1  # Time step

        # A defines how the state evolves when there are no controls
        self.A = torch.tensor([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

        # B specifies how the state evolves in response to a control input
        self.B = torch.tensor([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]], dtype=torch.float32, device=self.device)
        self.state = torch.tensor([0, 0, 0, 0], device=self.device).float() 

    '''
    '''
    def forward(self):
        x = F.relu(self.linear1(self.state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean_accel_x, mean_accel_y = self.action_mean(x)
        raw_cov_params = self.covariance_matrix(x)
        L = torch.zeros((2, 2), device=self.device)
        L[0, 0] = torch.exp(raw_cov_params[0])
        L[1, 0] = raw_cov_params[1]    
        L[1, 1] = torch.exp(raw_cov_params[2])
        
        cov_matrix = L @ L.T
        mean = torch.tensor([mean_accel_x, mean_accel_y], requires_grad=True).to(self.device).float() 
        return mean, cov_matrix

    '''
    '''
    def sample(self, distribution_mean, covariance_matrix):
        multivariate_distribution = MultivariateNormal(distribution_mean, covariance_matrix)
        breakpoint()
        accel = multivariate_distribution.sample()
        breakpoint()
        return torch.tensor([accel[0], accel[1]]).to(self.device)
    
    '''
    Compute the loss of the episode. 

    Given an episode of states, actions, rewards, and action probabilities, 
    '''
    def get_loss(self, episode):
        log_probs = []
        disc_cum_sum_rewards = []
        episode_len = len(episode)
        # for each t step in episode
        for t in range(0, episode_len):
            state, action, reward, log_prob_action = episode[t]
            log_probs.append(log_prob_action)
            #breakpoint()
            disc_cum_sum_rewards.append(self.calc_disc_sum(episode[t:])) # only the current timestep to the end of an ep
        terms = []
        # calculate loss
        for log_prob, cum_sum in zip(log_probs, disc_cum_sum_rewards):
            terms.append(-log_prob * cum_sum)
        terms_tensor = torch.stack(terms).to(self.device)
        #breakpoint()
        return torch.sum(terms_tensor).to(self.device)

    '''
    '''
    def update_state(self, state, action):
        # x dot = Ax + Bu
        x_dot = torch.matmul(self.A,state) + torch.matmul(self.B,action)
        return x_dot
    
    '''
    '''
    def integrate_dynamics(self, current_action):
        func = lambda time, state: self.update_state(state, current_action)
        state = odeint(func, self.state, torch.linspace(0, 0.1, 100).to(self.device))
        self.state = state[-1]
        max_state_value = 20.0  
        self.state = torch.clamp(self.state, -max_state_value, max_state_value)
        
    def calc_disc_sum(self, ep):
       #breakpoint()_, reward = self.env.get_reward(self.state[0], self.state[1], threshold=20)
        sum = 0
        gamma = 0.9
        index = 1
        for t_step, (_, _, reward, _) in enumerate(ep):
            sum = sum + (gamma ** t_step)*reward# reward
            index = index + 1
        return sum

    def collect_episode(self):
        self.state = torch.tensor([0, 0, 0, 0], device=self.device).float()
        t_step = 0
        episode = []
        state_type = None
        while state_type != "Cookie" and t_step < T_MAX:
            distribution_mean, covariance_matrix = self.forward()

            covariance_matrix = torch.clamp(covariance_matrix, min=1e-6)
            dist = MultivariateNormal(distribution_mean, covariance_matrix)
            accel = dist.sample()
            action = torch.tensor([accel[0], accel[1]]).to(self.device)
            adjusted_action = self.adjust_action_if_oob(self.state, action)

            log_prob_action = dist.log_prob(adjusted_action)
            
            self.integrate_dynamics(adjusted_action)
            _, reward = self.env.get_reward(self.state[0], self.state[1], threshold=5)
            
            episode.append((self.state, adjusted_action, reward, log_prob_action))
            
            t_step += 1
            state_type, _ = self.env.get_reward(self.state[0], self.state[1], threshold=5)

        self.state = torch.tensor([0, 0, 0, 0], device=self.device).float()
        return episode


    def adjust_action_if_oob(self, current_state, proposed_action):
        dt = 0.1
        max_position = 10.0 
        max_velocity = 5.0 
        next_position = current_state[:2] + current_state[2:] * dt + 0.5 * proposed_action * (dt ** 2)
        next_velocity = current_state[2:] + proposed_action * dt
        for i in range(2):
            if abs(next_position[i]) > max_position or abs(next_velocity[i]) > max_velocity:
                proposed_action[i] *= -1 

        return proposed_action
