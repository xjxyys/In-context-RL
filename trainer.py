import numpy as np
import torch
import wandb
import time

# activate wandb with key
wandb.login(key= 'da9f2bbc59300d7434138182a83a9b35c0cea793')
wandb.init(project='decision_transformer')


class SequenceTrainer:
       def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        def train_iter(self, num_steps, iter_num=0):
            """
            iter_num: int. The current iteration number."""
            train_losses = []
            logs = dict()

            train_set = time.time()

            self.model.train()

            for _ in range(num_steps):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
                

                logs['training_time'] = time.time() - train_set
                
                eval_time = time.time() 

                self.model.eval()

                for eval_fn in self.eval_fns:
                    outputs = eval_fn(self.model)
                    for key, value in outputs.items():
                        logs[key] = value
                
                logs['eval_time'] = time.time() - eval_time
                logs['training_loss_mean'] = np.mean(train_losses)
                logs['training_loss_std'] = np.std(train_losses)
            
                wandb.log(logs)
        
        def train_step(self):
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
            action_target = torch.clone(actions)

            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(
                None, action_preds, None,
                None, action_target, None,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()

            with torch.no_grad():
                self.diagnostics['training_action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

            return loss.detach().cpu().item()
        
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
        states, actions, rewards = traj
        tokens = []
        for t in range(self.time_step):  # fixed range issue here
            h1 = self.embed_odd(states, t)
            h2 = self.embed_even(actions, rewards, t)
            tokens.extend([h1, h2])
        # to torch tensor
        tokens = torch.tensor(tokens, dtype=torch.float32)
        return tokens
    
    def __getitem__(self, idx):
        return self.tokenize(self.traj_data[idx])

