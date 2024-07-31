from icrl import *
import tqdm
from model import Transformer
import pandas as pd
import argparse

def convert_to_numpy(array_str):
    # Remove all brackets and split the string based on spaces
    array_str = array_str.replace('[', '').replace(']', '').replace('\n', '').strip()
    # Convert string to float and form a numpy array
    return np.array(list(map(float, array_str.split())))

def expiriment(env, algo, num_trajectories, T=200):
    # algo : linucb/Tomp
    trajectories = [] # Store all trajectories
    # In this setting, s_t = \mathbb{A} = action_set!
    all_regrets = np.zeros((num_trajectories, T))

    # use tqdm to show progress bar

    for i in tqdm.tqdm(range(num_trajectories)):

        total_regret = 0
        regrets = []
        best_action_index = env.get_best_action_index()  # Best action doesn't change in this setup
        best_action_reward = np.dot(env.action_set[best_action_index], env.w_star)
        states, actions, rewards, action_indexs = [], [], [], []
        
        for _ in range(T):
            action_index = algo.select_action(env.action_set)
            reward, action = env.step(action_index)
            # find action
            algo.update(reward, action)
            # Calculate regret for this round and add to total
            expected_reward = np.dot(env.action_set[action_index], env.w_star)
            
            round_regret = best_action_reward - expected_reward
            total_regret += round_regret
            # print(round_regret)
            # Store state, action, reward for this round
            states.append(env.get_action_set()) 
            actions.append(action)
            rewards.append(reward)
            action_indexs.append(action_index)
            regrets.append(total_regret)

        all_regrets[i] = regrets # Store regrets for this trajectory
        trajectories.append((states, actions, rewards, action_indexs)) # Store trajectory
        # Reset env and LinUCB for next trajectory
        env.reset()
        algo.reset()

    return trajectories, all_regrets

def validate_with_training_data(args):
    if args.source == 'linucb':
        df_states_and_best_actions = pd.read_csv('data/linucb_states_and_best_actions.csv') # states, best_action_index, w_star
    else:
        df_states_and_best_actions = pd.read_csv('data/random_states_and_best_actions.csv')

    df_states_and_best_actions['state'] = df_states_and_best_actions['state'].apply(lambda x: convert_to_numpy(x).reshape(-1, 5))
    df_states_and_best_actions['w_star'] = df_states_and_best_actions['w_star'].apply(convert_to_numpy)

    config = {
        'horizon': args.horizon,
        'dim': args.dim,
        'act_num': args.action_num,
        'state_dim': args.state_dim,
        'dropout': args.dropout,
        'action_dim': args.action_num,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'shuffle': True,
        'activation': args.activation,
        'pred_q': args.Q,
        'test': True
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config, device)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    

    states = np.array(df_states_and_best_actions['state'].tolist())
    best_action_indices = df_states_and_best_actions['best_action_index'].to_numpy()
    w_star = np.array(df_states_and_best_actions['w_star'].tolist())

    num_trajectories = args.num_trajectories
    assert num_trajectories <= len(states), 'Number of trajectories should be less than the number of training data'

    all_regrets = np.zeros((num_trajectories, args.horizon))
    batchsize = 32
    trajectories = []
    initialize_prob = 1

    for i in tqdm.tqdm(range(num_trajectories//batchsize)):
        total_regrets = np.zeros(batchsize)
        regrets = [[] for _ in range(batchsize)]

        best_action_indexs = best_action_indices[i*batchsize:(i+1)*batchsize]
        best_action_rewards = [np.dot(states[i][best_action_index], w_star[i]) for i, best_action_index in enumerate(best_action_indexs)]
        states, actions, rewards, action_indices = [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)]
        action_sets = torch(states[i*batchsize:(i+1)*batchsize]).reshape(batchsize, -1).to(device)
        
        for t in range(1, args.horizon+1):
            if t == 1:
                context_actions = torch.empty((batchsize, 0, args.action_num), dtype=torch.float32).to(device)
                context_rewards = torch.empty((batchsize, 0, 1), dtype=torch.float32).to(device)
                x = {
                    'action_set': action_sets,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards
                }
            else:
                x = {
                    'action_set': action_sets,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards
                }
            random_number = np.random.rand()
            if random_number < initialize_prob/np.sqrt(t):
                action_indices = torch.randint(0, args.action_num, (batchsize, 1)).to(device)
            else:
                last_timestep_outputs = model(x) 
                action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)

            rewards_ = [np.dot(states[i][action_index], w_star[i]) for i, action_index in enumerate(action_indices)]
            actions_ = [states[i][action_index] for i, action_index in enumerate(action_indices)]
            actions_one_hot = torch.zeros(batchsize, 1, args.action_num).to(device)
            actions_one_hot.scatter_(2, action_indices.unsqueeze(1), 1)

            reward_tensor = torch.tensor(rewards_, dtype=torch.float32).to(device).reshape(batchsize, 1, 1)
            
            context_actions = torch.cat([context_actions, actions_one_hot], dim=1)
            context_rewards = torch.cat([context_rewards, reward_tensor], dim=1)

            expected_rewards = [np.dot(states[i][action_index], w_star[i]) for i, action_index in enumerate(action_indices)]
            round_regrets = [best_action_reward - expected_reward for best_action_reward, expected_reward in zip(best_action_rewards, expected_rewards)]
            total_regrets += round_regrets

            for j in range(batchsize):
                regrets[j].append(total_regrets[j])
                states[j].append(states[j])
                actions[j].append(actions_[j])
                rewards[j].append(rewards_[j])
                action_indices[j].append(action_indices[j].item())

        all_regrets[i*batchsize:(i+1)*batchsize] = regrets
        trajectories.append((states, actions, rewards, action_indices)) # Store trajectory

    df_regrets = pd.DataFrame(all_regrets)
    df_regrets.to_csv(args.save_path, index=False)

def validate(args):
    config = {
        'horizon': args.horizon,
        'dim': args.dim,
        'act_num': args.action_num,
        'state_dim': args.state_dim,
        'dropout': args.dropout,
        'action_dim': args.action_num,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'shuffle': True,
        'activation': args.activation,
        'pred_q': args.Q,
        'test': True
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    num_trajectories = args.num_trajectories
    batchsize = 32
    trajectories = []
    all_regrets = np.zeros((num_trajectories, args.horizon))
    initilize_prob = 1

    for i in tqdm.tqdm(range(num_trajectories//batchsize)):

        envs = [Environment(args.action_num, args.dim) for _ in range(batchsize)]
        total_regret = np.zeros(batchsize)
        # regrets = np.zeros((batchsize, T))
        regrets = [[] for _ in range(batchsize)]

        best_action_indexs = [env.get_best_action_index() for env in envs]  # Best action doesn't change in this setup
        best_action_rewards = [np.dot(env.action_set[best_action_index], env.w_star) for env, best_action_index in zip(envs, best_action_indexs)]
        states, actions, rewards, action_indexs = [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)]
        action_sets = [torch.tensor(env.get_action_set(), dtype=torch.float32).to(device).reshape(-1) for env in envs]
        # action_sets shape: [batchsize, num_actions*context_dim]
        action_sets = torch.stack(action_sets).reshape(batchsize, -1) # [batchsize, num_actions*context_dim]

        for t in range(1, args.horizon+1):
            if t == 1:
                context_actions = torch.empty((batchsize, 0, args.action_num), dtype=torch.float32).to(device)
                context_rewards = torch.empty((batchsize, 0, 1), dtype=torch.float32).to(device)
                x = {
                    'action_set': action_sets,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards
                }
            else:
                x = {
                    'action_set': action_sets,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards
                }
            random_number = np.random.rand()
            if random_number < initilize_prob/np.sqrt(t):
            # if t <= 30:
                action_indices = torch.randint(0, args.action_num, (batchsize, 1)).to(device)
                # choose t%num_actions
                # action_indices = torch.tensor([t%num_actions]*batchsize).to(device).unsqueeze(1)
            else:
                last_timestep_outputs = model(x) 
                # last_timestep_outputs shape: [batchsize, num_actions]
                # action_indices = torch.multinomial(F.softmax(last_timestep_outputs, dim=-1), 1)
                action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)
                # 接一个softmax
                # tao = initial_tao/np.sqrt(t)
                # action_indices = torch.multinomial(F.softmax(last_timestep_outputs/tao, dim=-1), 1) #dimension [batchsize, 1]
                
            # last_timestep_outputs = model(x) 
            # [2*t-1].argmax().item()

            rewards_ = [env.step(action_index)[0] for env, action_index in zip(envs, action_indices)]
            actions_ = [env.step(action_index)[1] for env, action_index in zip(envs, action_indices)]
            # print(len(actions_))
            # find action
            actions_one_hot = torch.zeros(batchsize, 1, args.action_num).to(device)
            actions_one_hot.scatter_(2, action_indices.unsqueeze(1), 1)

            reward_tensor = torch.tensor(rewards_, dtype=torch.float32).to(device).reshape(batchsize, 1, 1)
            
            context_actions = torch.cat([context_actions, actions_one_hot], dim=1)
            context_rewards = torch.cat([context_rewards, reward_tensor], dim=1)
            # print(context_rewards.shape)
            # print(context_actions.shape)
            
            expected_rewards = [np.dot(env.action_set[action_index], env.w_star) for env, action_index in zip(envs, action_indices)]
            
            # round_regret = best_action_reward - expected_reward
            round_regrets = [best_action_reward - expected_reward for best_action_reward, expected_reward in zip(best_action_rewards, expected_rewards)]
            # print(round_regret)
            # total_regret += round_regret
            total_regret += round_regrets
            for j in range(batchsize):
                regrets[j].append(total_regret[j])
                states[j].append(envs[j].get_action_set())
                actions[j].append(actions_[j])
                rewards[j].append(rewards_[j])
                action_indexs[j].append(action_indices[j].item())

        # all_regrets[i] = regrets # Store regrets for this trajectory
        all_regrets[i*batchsize:(i+1)*batchsize] = regrets
        trajectories.append((states, actions, rewards, action_indexs)) # Store trajectory
    
    df_regrets = pd.DataFrame(all_regrets)
    df_regrets.to_csv(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain the model')
    parser.add_argument('--source', type=str, default='linucb', help='source of the data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--state_dim', type=int, default=50, help='state dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--action_num', type=int, default=10, help='number of actions')
    parser.add_argument('--horizon', type=int, default=200, help='horizon')
    parser.add_argument('--dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--n_layer', type=int, default=8, help='number of layers')
    parser.add_argument('--n_embd', type=int, default=32, help='embedding dimension')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads')
    parser.add_argument('--save_path', type=str, default='data/_regrets.csv', help='path to save the regrets')
    parser.add_argument('--traning_data', action='store_true', help='use training data')
    args = parser.parse_args()
    if args.traning_data:
        validate_with_training_data(args)
    else:
        validate(args)