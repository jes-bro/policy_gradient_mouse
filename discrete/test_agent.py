'''
A class for testing the policy gradient agents' methods. 
'''
import unittest
import torch
from agent import Agent

class TestAgent(unittest.TestCase):
    '''
    Test the deterministic methods of the Policy Gradient agent class. 

    I'm not going to test probablistic functions because 
    they are inherently stochastic and will not have a deterministic
    answer to test.
    '''
    def test_get_reward(self):
        '''
        Test the get_reward method of the Policy Gradient agent class. 
        Test that for different state types it returns the corresponding reward.
        '''
        device = torch.device('cuda:0')

        # Discrete 4x4 grid world where each state type is encoded with an integer value
        grid_world = torch.tensor([[1, 0, 0, 2], [2, 0, 1, 0],[3, 0, 0, 0], [0, 1, 0, 0]], device=device)

        # Mapping from code (for torch tensors which can't have strings) to strings representing state type
        state_dict = {0:"Empty", 1:"Cheese", 2:"Salad", 3:"Cookie"}

        # Initialize mouse agent
        agent = Agent(2, device, grid_world, state_dict)
        agent.state = torch.tensor([0, 3], device=agent.device).int() # The upper right hand corner should be
        # a salad, therefore the reward should be -10000
        reward = agent.get_reward()
        self.assertEqual(-10000,reward, "The agent's reward should match that of it's state")

        agent.state = torch.tensor([3, 3], device=agent.device).int() # The the lower hand corner should be
        # empty, therefore the reward should be 0
        reward = agent.get_reward()
        self.assertEqual(0,reward, "The agent's reward should match that of it's state")

        agent.state = torch.tensor([1, 2], device=agent.device).int() # The 2, 1 spot should be
        # a cheese, therefore the reward should be 100
        reward = agent.get_reward()
        self.assertEqual(100,reward, "The agent's reward should match that of it's state")

    def test_update_state(self):
        '''
        Test the update_state method of the Policy Gradient Agent class. 
        Test that action updates the agents' state correctly.
        '''
        device = torch.device('cuda:0')

        # Discrete 4x4 grid world where each state type is encoded with an integer value
        grid_world = torch.tensor([[1, 0, 0, 2], [2, 0, 1, 0],[3, 0, 0, 0], [0, 1, 0, 0]], device=device)

        # Mapping from code (for torch tensors which can't have strings) to strings representing state type
        state_dict = {0:"Empty", 1:"Cheese", 2:"Salad", 3:"Cookie"}

        # Initialize mouse agent
        agent = Agent(2, device, grid_world, state_dict)
        agent.state = torch.tensor([0, 3], device=agent.device).int()
        # Moving down from 0, 3 should bring the state to 1,3
        # The encoding for down is 0
        agent.update_state(0)
        agent_state = (agent.state.cpu()[0], agent.state.cpu()[1])
        self.assertEqual(agent_state, (1,3))
        # Moving up from 1, 3 should bring the state to 0,3
        # The encoding for down is 0
        agent.update_state(1)
        agent_state = (agent.state.cpu()[0], agent.state.cpu()[1])
        self.assertEqual(agent_state, (0,3))
        # Moving up from 0, 3 should keep the state where it is
        # to prevent the agent from going out of bounds
        agent.update_state(1)
        agent_state = (agent.state.cpu()[0], agent.state.cpu()[1])
        self.assertEqual(agent_state, (0,3))
        # Moving to the right from 0,3 should keep the state where it is
        agent.update_state(2)
        agent_state = (agent.state.cpu()[0], agent.state.cpu()[1])
        self.assertEqual(agent_state, (0,3))
        # Moving to the left from 0,3 should move the agent to 0,2
        agent.update_state(3)
        agent_state = (agent.state.cpu()[0], agent.state.cpu()[1])
        self.assertEqual(agent_state, (0,2))

    def test_get_state_type(self):
        '''
        Test the get_state_type method of the Policy Gradient mouse agent class.

        Test that the correct state type is returned based on the state's position/encoding in the grid world.
        '''
        device = torch.device('cuda:0')
        # Discrete 4x4 grid world where each state type is encoded with an integer value
        grid_world = torch.tensor([[1, 0, 0, 2], [2, 0, 1, 0],[3, 0, 0, 0], [0, 1, 0, 0]], device=device)

        # Mapping from code (for torch tensors which can't have strings) to strings representing state type
        state_dict = {0:"Empty", 1:"Cheese", 2:"Salad", 3:"Cookie"}

        # Initialize mouse agent
        agent = Agent(2, device, grid_world, state_dict)
        agent.state = torch.tensor([0, 3], device=agent.device).int() # The upper right hand corner should be
        # a salad
        state = agent.get_state_type()
        self.assertEqual("Salad", state, "The agent's state should match that of the grid world")

        agent.state = torch.tensor([3, 3], device=agent.device).int() # The the lower hand corner should be
        # empty
        state = agent.get_state_type()
        self.assertEqual("Empty",state, "The agent's state should match that of the grid world")

        agent.state = torch.tensor([1, 2], device=agent.device).int() # The 2, 1 spot should be
        # a cheese
        state = agent.get_state_type()
        self.assertEqual("Cheese",state, "The agent's state should match that of the grid world")

    def test_disc_rewards(self):
        '''
        Test that disc_rewards method calculcates the discounted sum of rewards correctly for a dummy
        episode's worth of data.
        '''
        device = torch.device('cuda:0')
        # Discrete 4x4 grid world where each state type is encoded with an integer value
        grid_world = torch.tensor([[1, 0, 0, 2], [2, 0, 1, 0],[3, 0, 0, 0], [0, 1, 0, 0]], device=device)

        # Mapping from code (for torch tensors which can't have strings) to strings representing state type
        state_dict = {0:"Empty", 1:"Cheese", 2:"Salad", 3:"Cookie"}

        # Initialize mouse agent
        agent = Agent(2, device, grid_world, state_dict)
        ep = [(0, 0, 0, 1, 10), (0, 0, 0, 2, 0), (0, 0, 0, 1, 1), (1, 0, 0, 2, 10)]
        # Fake scenario where the episode is length 4, and the rewards are 1, 2, 1, 2
        # For the first step in the episode, the discounted sum of rewards should be 5.068
        self.assertEqual(agent.disc_rewards(ep), 5.068)
        # For the second step in the episode, the discounted sum of rewards should be 4.52
        self.assertEqual(agent.disc_rewards(ep[1:]), 4.52)
        # For the third step, the sum should be 2.8
        self.assertEqual(agent.disc_rewards(ep[2:]), 2.8)
        # The last step should just be the reward itself- because the discount factor will be zeroed out
        self.assertEqual(agent.disc_rewards(ep[3:]), 2)
        

if __name__ == '__main__':
    unittest.main()
