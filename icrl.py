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
# Dataset
####################

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        # trajectories的结构应该是一个列表，其中包含形如([s1, s2, ..., sT], [a1, a2, ..., aT], [r1, r2, ..., rT])的元组
        self.trajectories = trajectories

    def __len__(self):
        return sum(len(traj[0]) - 1 for traj in self.trajectories)  # 每个轨迹减去初始状态

    def __getitem__(self, idx):
        # 根据idx找到对应的轨迹和时间步
        for trajectory in self.trajectories:
            states, actions, rewards = trajectory
            if idx < len(states) - 1:
                break
            idx -= len(states) - 1
        
        # 构建输入，前面所有的状态和动作和奖励
        state_action = torch.cat((states[idx].flatten(), actions[idx].reshape(-1)), dim=0)
        # 下一个动作作为目标
        next_action = actions[idx + 1].reshape(-1)

        # 目标是预测下一个动作的概率分布
        return state_action, next_action

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, history_len=200):
        self.sequences = []
        for trajectory in trajectories:
            # 创建历史序列，最大长度为history_len
            history = []
            for t in range(1, len(trajectory)):
                state, action, reward = trajectory[t-1]
                # 将状态、动作和奖励添加到历史中
                history.extend(state.flatten().tolist())
                history.extend(action.flatten().tolist())
                history.append(reward)  # 奖励是个标量，直接添加
                
                # 当前状态和历史一起形成模型的输入
                model_input = history + trajectory[t][0].flatten().tolist()

                # 下一个动作是目标
                model_target = trajectory[t][1].flatten().tolist()

                self.sequences.append((model_input, model_target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_sequence, target_action = self.sequences[idx]
        return torch.tensor(input_sequence, dtype=torch.float), torch.tensor(target_action, dtype=torch.float)


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
# Transformer Model Class
####################

# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, model_dim, output_dim, num_heads, num_layers):
#         super(TransformerModel, self).__init__()
        
#         # Embedding layer that expands the input features to the model dimension
#         self.input_embedding = nn.Linear(input_dim, model_dim)
        
#         # Transformer encoder layer
#         encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
#         # Output layer that predicts the action logits
#         self.output_layer = nn.Linear(model_dim, output_dim)
        
#     def forward(self, x):
#         x = self.input_embedding(x)
#         x = self.transformer_encoder(x)
#         x = self.output_layer(x)
#         return x

#     def predict(self, x):
#         with torch.no_grad():
#             return self(x)

#     def compute_loss(self, predicted, target):
#         criterion = nn.CrossEntropyLoss()
#         return criterion(predicted, target)

#     def optimize(self, loss):
#         optimizer = optim.Adam(self.parameters(), lr=0.001)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


# ####################
# # Experiment Class
# ####################

# class Experiment:
#     def __init__(self, env, linucb):
#         self.env = env
#         self.linucb = linucb

#     def run(self, n_rounds):
#         for _ in range(n_rounds):
#             context = self.env.reset()
#             action = self.linucb.select_action(context)
#             reward = self.env.step(action)
#             self.linucb.update(context, action, reward)
#             best_action_index = self.env.get_best_action_index()

#         print(f"Experiment completed for {n_rounds} rounds.")



# class Experiment:
#     def __init__(self, env, model, alg, config):
#         self.env = env
#         self.model = model
#         self.alg = alg
#         self.config = config
    
#     def run(self):
#         data = self.collect_data()
#         self.train_transformer(data)
    
#     def collect_data(self):
#         data = []
#         for episode in range(self.config['num_episodes']):
#             state = self.env.reset()
#             for t in range(self.config['max_steps']):
#                 context = self.extract_context(state)
#                 action = self.alg.select_action(context)
#                 next_state, reward, done = self.env.step(action)
                
#                 data.append((context, action, reward))
#                 self.alg.update(action, context, reward)
                
#                 if done:
#                     break
#                 state = next_state
#         return data
    
#     def train_transformer(self, data):
#         preprocessed_data = self.preprocess_data(data)
        
#         for epoch in range(self.config['num_epochs']):
#             for batch in self.data_loader(preprocessed_data, self.config['batch_size']):
#                 contexts, actions, rewards = batch
#                 predicted_actions = self.model.predict(contexts)
#                 loss = self.model.compute_loss(predicted_actions, actions)
                
#                 self.model.optimize(loss)
    
#     @staticmethod
#     def extract_context(state):
#         # Context extraction logic goes here...
#         pass
    
#     @staticmethod
#     def preprocess_data(data):
#         # Data preprocessing logic goes here...
#         pass
    
#     @staticmethod
#     def data_loader(data, batch_size):
#         # Batching logic goes here...
#         pass

# # Example usage:
# env = Environment() # Your environment class
# model = TransformerModel() # Your transformer model
# alg = LinUCB(alpha=1.0, d=feature_dimension) # Your LinUCB instance
# config = {
#     'num_episodes': 100,
#     'max_steps': 200,
#     'num_epochs': 10,
#     'batch_size': 32,
# }

# experiment = Experiment(env, model, alg, config)
# experiment.run()
