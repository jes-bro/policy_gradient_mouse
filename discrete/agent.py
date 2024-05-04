'''
Defines the mouse agent class.
'''
import torch
import wandb
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Agent(nn.Module):
    '''
    Define a mouse agent class that can advance its state and encode a policy that
    tells it where to go next based on its current state.
    '''
    def __init__(self, input_dim, device, env, state_map):       
        '''
        Initialize the agent and the policy network (layer).
        
        Params:
            input_dim: An integer representing the state dimension (2)
            device: The device to do the NN computations on (cuda:0)
            env: A torch tensor representing the grid world and the
            types of its states encoded in integers.
            state_map: A dictionary that maps the state encodings (ints)
            to their types (Strings)
        '''
        super().__init__() 
        self.env = env
        self.state_map = state_map
        self.device = device
        self.linear1 = nn.Linear(input_dim, 4, device=self.device).float() 
        self.state = torch.tensor([0, 0], device=self.device).int()

    def update_state(self, action):
        '''
        Update the agent's state

        Given an action that will update the agent's state along a particular
        dimension, advance the state of the agent in the grid world.

        Actions shouldn't cause the state to go out of bounds, but as an extra
        security measure, only update the state if the state is within bounds/
        check whether performing the action will cause the agent's state to
        go out of bounds. If it doesn't update the self.state attribute.

        Params:
            action: An integer encoding the direction in which the agent should move.
        '''
        new_row, new_col = self.state[0].item(), self.state[1].item()
        if action == 0 and new_row < self.env.shape[0] - 1: # move down 
            new_row += 1
        elif action == 1 and new_row > 0: # move up
            new_row -= 1
        elif action == 2 and new_col < self.env.shape[1] - 1:
            new_col += 1 # move right
        elif action == 3 and new_col > 0:
            new_col -= 1 # move left

        self.state[0], self.state[1] = new_row, new_col

    def forward(self):
        '''
        Perform a forward pass through the network.
        Using the agent's state, retrieve a discrete probability distribution
        over actions to take.

        Returns:
            action_probs: A torch tensor of discrete action probabilities (1 by 4)
            (there should be a probability associated with each potential action,
            up, down, left, or right)
        '''
        action_probs = F.softmax(self.linear1(self.state.float()))
        return action_probs

    def sample(self, action_probs):  
        '''
        Sample an action from a discrete probability distribution.

        Given a probability distribution over potential actions, 
        select an action A mousefrom the distribution. Higher probability actions
        are more likely to be selected, but the stochastic nature of the sampling
        means that exploration of the state space is still encouraged.

        Also, mask the action probabilities so that only legal actions can be outputted
        by the policy. As in, any actions that put the agent in a state outside of the grid
        world are prevented, by making the probability of those actions being selected 0.

        Params:
            action_probs: A torch tensor of discrete action probabilities (1 by 4)
                (there should be a probability associated with each potential action,
                up, down, left, or right)

        Returns:
            action: A torch tensor representing the selected action (1D scalar).
        '''
        row, col = self.state[0].item(), self.state[1].item()
        mask = torch.ones_like(action_probs)
        if row == 0:
            mask[1] = 0
        if row == self.env.shape[0] - 1:
            mask[0] = 0
        if col == 0:
            mask[3] = 0 
        if col == self.env.shape[1] - 1:
            mask[2] = 0 
        masked_probs = action_probs * mask
        masked_probs /= masked_probs.sum()
        action = torch.multinomial(masked_probs, 1)
        return action
    
    def get_reward(self):
        '''
        Figure out the type of state the agent is in and return 
        the reward assoicated with that state.

        Returns:
            An integer representing the reward for being in the agent's state.
        '''
        state_type = self.get_state_type()
        if state_type == "Cheese":
            return 100
        elif state_type == "Salad":
            return -10000
        elif state_type == "Cookie":
            return 100000
        else:
            return 0

    def get_state_type(self):  
        '''
        Determine the state type of the state the agent is currently located in.
        
        Determine the value of the state at agent's location in the environment, 
        then use that value (key) to access its corresponding label/type in the state map.

        Returns:
            A string representing the type of the state.
        '''
        return self.state_map[self.env[self.state[0], self.state[1]].item()]
        
    def collect_episode(self):
        '''
        Collect an episode. 

        Collect an episode, where each step in the episode is a tuple consisting of
        states, actions, rewards and action probabilities. For each step in the episode,
        run a forward pass through the network to obtain a new discrete distribution
        over actions. Then, sample from that distribution to collect the action. Then,
        log the state, action, action distribution (for calculcating the log probabilities),
        and reward for that time step as a tuple in the episode list. Then,
        evolve the state forward, and use that state as the input to the network to get the next
        action out.

        Returns:
            A list of state, action, reward, action distribution tuples for each step in the episode.
        '''
        self.state = torch.tensor([1, 1], device=self.device).int() # Reset the state at the start
        # of each episode
        episode = [] # Reset episode []
        step = 0
        max_steps = 10 # Only allow 10 steps before cutting off trajectory/episode
        done = False # Not done collecting trajectory yet
        while not done and step < max_steps:
            state = self.state.clone() # So as to not put the state directly into the episode list but
            # just its value
            action_probs = self.forward() # Get action probs
            action = self.sample(action_probs) # Sample action from distribution
            self.update_state(action) # Update the state
            reward = self.get_reward() # Obtain the reward (really high for cookie)
            done = (self.get_state_type() == "Cookie") # If you found the cookie, be done, else, continue
            episode.append((state.clone(), action_probs, action, reward, done)) # Add step to episode
            if done:
                break
            step += 1 # Increment step so as to not exceed max number of steps
        return episode
    
    def disc_rewards(self, ep):
        '''
        Calculate the discounted cumulative sum of rewards over the course of an episode
        from some start step. Because Policy Gradients use a Monte-Carlo style reward
        accumulation, we only calculcate the sum of rewards from some inital step to
        the end of the episode, (for all steps in the episode, but this functuion only
        does one steps' discounted sum of rewards calculation).

        Params:
            ep: The episode/ tuple of states, actions, rewards, and action distributions. This
            is actually a fraction of the episode, starting at a particular step in the episode.

        Returns:
            The discounted sum of rewards from the beginning of this slice of the episode to the
            end.
        '''
        sum = 0
        gamma = 0.9 # Discount factor
        for t_step, (_, _, _, reward, _) in enumerate(ep):
            sum = sum + (gamma ** t_step)*reward # gamma to the power of the t_step index, times the reward
        return sum


        



