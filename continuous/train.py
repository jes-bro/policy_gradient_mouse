import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import no_grad
from torch.distributions import MultivariateNormal
from pg_agent import MouseAgent
from env import Env

WORLD_DIM = 20

env = Env(WORLD_DIM, WORLD_DIM, 3, 5, 5, 3)
device = torch.device('cuda:0')
agent = MouseAgent(4, 5, env, device)
agent = agent.to(device)
from matplotlib.animation import FuncAnimation

run = wandb.init(
    project="MouseAgent",
    config={
        "learning_rate": 1e-1,
        "epochs": 200,
    },
)
def train(num_epochs, lr):
    opt = torch.optim.Adam(agent.parameters(),lr=lr)
    agent.train()
    for epoch in range(num_epochs):
        ep = agent.collect_episode()
        #print(len(ep))
        #for stuff in ep:
            #rint(stuff[0])
        loss = agent.get_loss(ep)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch={epoch}, loss={loss}")
        wandb.log({"loss": loss})

train(run.config["epochs"], run.config["learning_rate"])
wandb.finish()
