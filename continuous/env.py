import torch
import math 
import random
import numpy as np
import matplotlib.pyplot as plt

class Env():
    def __init__(self, world_dim_x, world_dim_y, num_cheeses, cookie_x, cookie_y, num_salads):
        self.num_cheeses = num_cheeses
        self.num_salads = num_salads
        self.x_dim = world_dim_x
        self.y_dim = world_dim_y
        self.cookie_x = cookie_x
        self.cookie_y = cookie_y
        self.build_world()
        #breakpoint()

    def build_world(self):
        self.important_states = {}
        self.important_states[(self.cookie_x, self.cookie_y)] = ("Cookie", 900000000)
        for _ in range(1, self.num_cheeses):
            x, y = self.generate_point()
            self.important_states[(x, y)] = ("Cheese", 10000)
        for _ in range(1, self.num_salads):
            x, y = self.generate_point()
            self.important_states[(x, y)] = ("Salad", -2000000)
            
    
    def check_distance_between_important_states(self, x, y, threshold):
        for (imp_state_x, imp_state_y), _ in self.important_states.items():
            if (math.dist(np.array([x, y]), np.array([imp_state_x, imp_state_y])) < threshold):
                return False
        return True
    
    def get_reward(self, x, y, threshold):
        for (imp_state_x, imp_state_y), (label, reward) in self.important_states.items():
            if (math.dist(np.array([x.cpu().detach().numpy(), y.cpu().detach().numpy()]), np.array([imp_state_x, imp_state_y])) < threshold):
                x.to(torch.device('cuda:0'))
                x.to(torch.device('cuda:0'))
                return label, reward
        return None, 1
    
    def generate_point(self):
        while True:
            x = random.randint(-self.x_dim, self.x_dim)
            y = random.randint(-self.y_dim, self.y_dim)
            if self.check_distance_between_important_states(x, y, threshold=5):
                return x, y
            
    def step(self, action, current_state):
        x, y = current_state
        if action == 0:  # up
            y += 1
        elif action == 1:  # down
            y -= 1
        elif action == 2:  # left
            x -= 1
        elif action == 3:  # right
            x += 1
        x = max(min(x, self.x_dim), -self.x_dim)
        y = max(min(y, self.y_dim), -self.y_dim)

        label, reward = self.get_reward(x, y, 3)
        done = label == "Cookie"  # End episode if Cookie is found
        next_state = (x, y)
        return next_state, reward, done
    
    def plot_important_states_with_emojis(self):
            x_coords = []
            y_coords = []
            colors = []
            emojis = []
            
            for (x, y), (label, _) in self.important_states.items():
                x_coords.append(x)
                y_coords.append(y)
                if label == "Cookie":
                    colors.append('gold')
                    emojis.append('Cookie')
                elif label == "Cheese":
                    colors.append('yellowgreen')
                    emojis.append('Cheese') 
                elif label == "Salad":
                    colors.append('red')
                    emojis.append('Salad')

            fig, ax = plt.subplots(figsize=(10, 10))
            scatter = ax.scatter(x_coords, y_coords, c=colors, s=200) 
            for i, emoji in enumerate(emojis):
                ax.annotate(emoji, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
            ax.grid(True)
            ax.set_xlim([-self.x_dim, self.x_dim])
            ax.set_ylim([-self.y_dim, self.y_dim])
            ax.set_title('Important States in the Environment')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            plt.show()
