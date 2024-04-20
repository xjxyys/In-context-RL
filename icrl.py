import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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

####################
# LinUCB Class
####################
class LinUCB:
    def __init__(self, num_actions, context_dim, alpha=1.0, lambda_reg=1.0):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.A = np.array([np.eye(context_dim) * lambda_reg for _ in range(num_actions)])
        self.b = np.zeros((context_dim, num_actions))
        self.context_dim = context_dim
        self.num_actions = num_actions

    def select_action(self, actions):
        p = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[:, a]
            p[a] = theta_a.T @ actions[a] + self.alpha * np.sqrt(actions[a].T @ A_inv @ actions[a])
        return np.argmax(p)

    # def update(self, action_index, reward, action):
    #     self.A[action_index] += np.outer(action, action)
    #     self.b[:, action_index] += reward * action.reshape(-1)  # Ensures 'action' is one-dimensional
    def update(self, action_index, reward, action):
        # Ensure 'action' is a one-dimensional array
        action = np.array(action).flatten()
        # Update A matrix for the chosen action
        self.A[action_index] += np.outer(action, action)
        # Update b vector for the chosen action
        # print(self.b[:, action_index].shape)
        # print(reward * action.shape)
        self.b[:, action_index] += reward * action
    
####################
# Usefull Functions
####################
# Function to run a single trajectory and compute regret
def run_trajectory_with_regret(env, linucb, rounds=200):
    total_regret = 0
    regrets = []
    best_action_index = env.get_best_action_index()  # Best action doesn't change in this setup
    best_action_reward = np.dot(env.action_set[best_action_index], env.w_star)
    
    for _ in range(rounds):
        action_index = linucb.select_action(env.action_set)
        reward = env.step(action_index)
        linucb.update(action_index, reward, env.action_set[action_index])
        
        # Calculate regret for this round and add to total
        round_regret = best_action_reward - reward
        total_regret += round_regret
        regrets.append(total_regret)
    
    return regrets

# Function to average the regrets over multiple trajectories
def average_regrets(num_trajectories, rounds=200):
    all_regrets = np.zeros((num_trajectories, rounds))
    for i in range(num_trajectories):
        env = Environment(num_actions=10, context_dim=5)
        linucb = LinUCB(num_actions=10, context_dim=5)
        regrets = run_trajectory_with_regret(env, linucb, rounds)
        all_regrets[i] = regrets
    # Average over all trajectories
    return np.mean(all_regrets, axis=0)




####################
# Thompson Sampling Class
####################
class ThompsonSampling:
    def __init__(self, env, lambda_reg):
        self.env = env
        self.lambda_reg = lambda_reg
        self.A = np.array([np.eye(env.context_dim) * lambda_reg for _ in range(env.num_actions)])
        self.b = np.zeros((env.context_dim, env.num_actions))

    def select_action(self, context):
        # Sample theta from the posterior distribution
        theta_samples = np.array([np.random.multivariate_normal(np.linalg.inv(self.A[a]) @ self.b[:, a], np.linalg.inv(self.A[a])) for a in range(self.env.num_actions)])
        # Select action based on the sampled theta
        p = np.array([theta.T @ context for theta in theta_samples])
        return np.argmax(p)

    def update(self, context, action, reward):
        # Update the estimates of A and b based on the observed reward
        self.A[action] += np.outer(context, context)
        self.b[:, action] += reward * context

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
