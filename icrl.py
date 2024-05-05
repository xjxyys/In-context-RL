import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

####################
# Environment Class
####################
class Environment:
    def __init__(self, num_actions, context_dim, std_variance=1.5):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.std_variance = std_variance  # Standard deviation of the Gaussian noise
        self.w_star = np.random.uniform(0, 1, context_dim)  # True model parameters
        self.action_set = np.random.uniform(-1, 1, (num_actions, context_dim))  # Fixed action set
    
    def get_action_set(self):
        return self.action_set

    def get_best_action_index(self):
        # Return the index of the best action based on the true model parameters
        return np.argmax(np.dot(self.action_set, self.w_star))

    def step(self, action_index):
        action = self.action_set[action_index]
        reward = np.dot(action, self.w_star) + np.random.normal(0, self.std_variance)
        return reward, action
    
    def reset(self):
        # reset w_star and action_set
        self.w_star = np.random.uniform(0, 1, self.context_dim)
        self.action_set = np.random.uniform(-1, 1, (self.num_actions, self.context_dim))
        
####################
# LinUCB Class
####################

class LinUCB:
    def __init__(self, num_actions, context_dim, alpha=2, lambda_reg=1):
        self.alpha = alpha  # Set alpha to the fixed value of 2
        self.lambda_reg = lambda_reg
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.A = np.eye(context_dim) * lambda_reg  # Initialize A as a diagonal matrix with lambda_reg on the diagonal: X.T @ X + lambda_reg * I
        self.b = np.zeros((context_dim, 1)) # Initialize b as a zero vector, b represents the sum of rewards for each action: X.T @ y
        # self.historical_rewards = np.zeros(1)  # Initialize the historical rewards vector
        self.t = 0  # Time step counter

    def estimate_w_ridge(self):
        """
        Estimates the ridge regression parameter w_ridge for action k
        with normalization by 2t.
        """
        A_inv = np.linalg.inv(self.A) # (X.T @ X + lambda_reg * I)^-1
        return A_inv @ self.b # A_inv @ X.T @ y


    def select_action(self, action_set):
        p = np.zeros(self.num_actions)
        w_ridge = self.estimate_w_ridge().flatten()
        A_inv = np.linalg.inv(self.A)
        # print('A_inv', A_inv.shape)
        # print('w_ridge', w_ridge.shape)
        
        for k in range(self.num_actions):
            chosen_action = action_set[k]
            # print('chosen_action', chosen_action.shape)
            p[k] = (chosen_action.T @ w_ridge +
                    self.alpha * np.sqrt(chosen_action.T @ A_inv @ chosen_action))
        # Select the action with the highest UCB
        return np.argmax(p)

    def update(self, reward, action):
        self.t += 1  # Increment the time step
        # Update A and b matrices with the chosen action and received reward
        self.A += np.outer(action, action)
        self.b += reward * action.reshape(-1, 1)  # Ensure 'action' is a one-dimensional array

    def reset(self):
        # Reset the A and b matrices, and the time step counter
        self.A = np.eye(self.context_dim) * self.lambda_reg  # Initialize A as a diagonal matrix with lambda_reg on the diagonal: X.T @ X + lambda_reg * I
        self.b = np.zeros((self.context_dim, 1)) # Initialize b as a zero vector, b represents the sum of rewards for each action: X.T @ y
        self.t = 0
####################
# Thompson Sampling Class
####################
class ThompsonSampling:
    def __init__(self, num_actions, context_dim, std_dev=1.5, lambda_param=1):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.std_dev = std_dev
        self.lambda_param = lambda_param
        self.A = np.eye(context_dim) * lambda_param
        self.b = np.zeros((context_dim, 1))

    def select_action(self, action_set):
        A_inv = np.linalg.inv(self.A)
        mu_t = A_inv @ self.b
        Sigma_t = self.lambda_param * self.std_dev * A_inv
        sampled_theta = np.random.multivariate_normal(mu_t.flatten(), Sigma_t)
        
        # Compute the value for each action
        values = action_set @ sampled_theta
        return np.argmax(values)

    def update(self, reward, action):
        # Update A and b matrices with the chosen action and received reward
        self.A += np.outer(action, action)
        self.b += (reward * action).reshape(-1, 1)

    def reset(self):
        # Reset the A and b matrices
        self.A = np.eye(self.context_dim) * self.lambda_param
        self.b = np.zeros((self.context_dim, 1))


####################
# Dataset
####################

class TrajectoryDataset:
    def __init__(self, traj_data, time_step=200, num_actions=10, context_dim=5):
        self.traj_data = traj_data
        self.time_step = time_step
        self.num_actions = num_actions
        self.context_dim = context_dim

    def __len__(self):
        return len(self.traj_data)
    
    def embed_odd(self, state, t):
        h1a = np.zeros(self.context_dim+1)  # h^a_{2t-1}
        h1b = state[t].reshape(self.num_actions*self.context_dim)  # h^b_{2t-1}
        h1c = np.zeros(self.num_actions)  # h^c_{2t-1}
        h1d = np.zeros(1)
        pos_1 = np.array([2*t-1, (2*t-1)**2, 1])
        h1 = np.concatenate([h1a, h1b, h1c, h1d, pos_1])
        return h1

    def embed_even(self, action, reward, t):
        h2a = action[t]
        h2a = np.concatenate([h2a, np.array([reward[t]])])  # Add reward to the action embedding
        h2b = np.zeros(self.num_actions*self.context_dim)
        h2c = np.zeros(self.num_actions)
        h2d = np.zeros(1)
        pos_2 = np.array([2*t, (2*t)**2, 1])
        h2 = np.concatenate([h2a, h2b, h2c, h2d, pos_2])
        return h2

    def tokenize(self, traj):
        states, actions, rewards, action_indexs = traj
        action_set = states[0]
        tokens = []
        for t in range(self.time_step):  # fixed range issue here
            h1 = self.embed_odd(states, t)
            h2 = self.embed_even(actions, rewards, t)
            tokens.extend([h1, h2])
        # to torch tensor
        tokens = torch.tensor(tokens, dtype=torch.float32)  
        # action_set: [num_actions, context_dim]
        # find the action_index for each action in actions
        action_labels = torch.tensor(action_indexs, dtype=torch.long)

        return tokens, action_labels
    
    def __getitem__(self, idx):
        return self.tokenize(self.traj_data[idx])

