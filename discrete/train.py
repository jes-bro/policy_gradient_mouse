'''
Train a mouse agent to encode a policy that will help it navigate to cookies and
cheese, while avoiding the salads.
'''
import torch
import wandb 
from agent import Agent

# GPU
device = torch.device('cuda:0')

# Discrete 4x4 grid world where each state type is encoded with an integer value
grid_world = torch.tensor([[1, 0, 0, 2], [2, 0, 1, 0],[3, 0, 0, 0], [0, 1, 0, 0]], device=device)

# Mapping from code (for torch tensors which can't have strings) to strings representing state type
state_dict = {0:"Empty", 1:"Cheese", 2:"Salad", 3:"Cookie"}

# Initalize wandb run
run = wandb.init(
    project="MouseAgent",
    config={
        "learning_rate": 1e-3,
        "epochs": 20000,
    },
)

# Initialize mouse agent
agent = Agent(2, device, grid_world, state_dict)

# Adam optimizer, chosen because its stable and the Internet recommended it
# Learning rate is 1e-2 because it seems like a reasonable choice
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

def train():
    '''
    Train a mouse agent to learn a policy that will take it to a cookie 
    while avoiding salads.

    Using pytorch, train a single layer NN to encode a policy that maps discrete
    grid world states (encoded as x, y coordinates) to actions (which are up, down,
    left, right). The actions update the state directly, so there is no need to evolve
    the state forward through an ODE integrator or anything.

    '''
    for epoch in range(run.config.epochs):
        total_reward = 0
        episode = agent.collect_episode()
        log_probs = []
        disc_rewards = []
        for step_index, step in enumerate(episode):
            state, action_probs, action, reward, done = step
            log_probs.append(torch.log(action_probs.squeeze(0)[action]))
            disc_rewards.append(torch.tensor(agent.disc_rewards(episode[step_index:]), device=device))
            total_reward += reward
        #breakpoint()
        loss = torch.sum(-torch.stack(log_probs) * torch.stack(disc_rewards))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch}, loss: {loss}, total reward: {total_reward}")
        wandb.log({"epoch":epoch, "loss": loss, "total_reward": total_reward})
    return episode

def print_policy(agent):
    '''
    Print the policy as a grid world where each state corresponds to the action the policy recommend
    the agent take if it were in that state.

    Params:
        agent: The mouse agent that learns the policy to navigate the grid world
    '''
    action_labels = {0: 'Down', 1: 'Up', 2: 'Right', 3: 'Left'}
    state_emojis = {0: "🔲", 1: "🧀", 2: "🥗", 3: "🍪"}
    policy_map = torch.zeros(grid_world.shape, dtype=torch.int)
    for i in range(grid_world.shape[0]):
        for j in range(grid_world.shape[1]):
            agent.state = torch.tensor([i, j], device=device).int()
            with torch.no_grad():
                action_probs = agent.forward()
                recommended_action = torch.argmax(action_probs).item()
                policy_map[i, j] = recommended_action
    policy_map = policy_map.cpu().numpy()
    print("Grid World:\t\t\t\tPolicy:")
    for i in range(grid_world.shape[0]):
        grid_row = ""
        policy_row = ""
        for j in range(grid_world.shape[1]):
            state_type = grid_world[i, j].item()
            grid_row += f"{state_emojis[state_type]}\t"
            action = action_labels[policy_map[i, j]]
            policy_row += f"{action}\t"
        print(f"{grid_row}\t\t{policy_row}")

# Train the policy 
episode = train()

# Display the policy next to the grid world
print_policy(agent)
wandb.finish()
