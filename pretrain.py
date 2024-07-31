import argparse
from icrl import *
from model import *
import pickle
import wandb
import tqdm
# from sklearn.model_selection import train_test_split

def load_data(args):
    assert args.source in ['linucb', 'random'] and args.data_size <= 100000
    if args.source == 'random':
        traj_0 = pickle.load(open('data/RandomChoose_trajectories_part0.pkl', 'rb'))
        traj_1 = pickle.load(open('data/RandomChoose_trajectories_part1.pkl', 'rb'))
        traj_2 = pickle.load(open('data/RandomChoose_trajectories_part2.pkl', 'rb'))
        traj_3 = pickle.load(open('data/RandomChoose_trajectories_part3.pkl', 'rb'))
        traj_4 = pickle.load(open('data/RandomChoose_trajectories_part4.pkl', 'rb'))
    else:
        traj_0 = pickle.load(open('data/linucb_trajectories_part0.pkl', 'rb'))
        traj_1 = pickle.load(open('data/linucb_trajectories_part1.pkl', 'rb'))
        traj_2 = pickle.load(open('data/linucb_trajectories_part2.pkl', 'rb'))
        traj_3 = pickle.load(open('data/linucb_trajectories_part3.pkl', 'rb'))
        traj_4 = pickle.load(open('data/linucb_trajectories_part4.pkl', 'rb'))
    
    # integrate all trajectories
    traj = []
    traj.extend(traj_0)
    traj.extend(traj_1)
    traj.extend(traj_2)
    traj.extend(traj_3)
    traj.extend(traj_4)
    return traj[:args.data_size + 1]


def trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, loss_fn, use_wandb=True, num_epochs=10):
    # DataLoader can handle batching and shuffling
    model.train()  # Set the model to training mode

    best_model_state = model.state_dict()
    best_loss = float('inf')

    for epoch in range(1, 1+num_epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_dataloader):
            # Unpack the data
            # tokens, action_labels = batch
            # print(batch['context_actions'].size())
            pred_actions = model(batch) # dimension: (batch_size, seq_len, action_dim)
            # print(pred_actions.shape)
            true_actions = batch['true_actions'] # dimension: (batch_size, seq_len)
            pred_actions_flat =  pred_actions.view(-1, action_dim)
            true_actions_flat = true_actions.view(-1)
            loss = loss_fn(pred_actions_flat, true_actions_flat)
            # loss.backward()
            # print(action_labels)
            # Reset the gradients in the optimizer
            optimizer.zero_grad()
            # # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        # Compute the test loss
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader):
                pred_actions = model(batch)
                true_actions = batch['true_actions']
                pred_actions_flat =  pred_actions.view(-1, action_dim)
                true_actions_flat = true_actions.view(-1)
                loss = loss_fn(pred_actions_flat, true_actions_flat)    
                test_loss += loss.item()
        model.train()
        test_loss /= len(test_dataloader)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
        
        if use_wandb:
            # wandb logging
            wandb.log({"train_loss": epoch_loss, "test_loss": test_loss})
            print(f"Epoch {epoch} | Train Loss {epoch_loss} | Test Loss {test_loss}")
    best_model = Transformer(config)
    best_model.load_state_dict(best_model_state)
    return best_model

def Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, loss_fn, use_wandb=True, num_epochs=10, gamma = 0.99, double=False):
    # record the loss before training
    train_loss = 0.0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            Qvalues = model(batch) # dimension: (batch_size, seq_len, A)
            # pred_Qvalues = pred_Qvalues.gather(2, batch['context_actions'].unsqueeze(-1).long()) # get the Q value of the context actions
            if args.soft_max:
                next_Qvalues = F.softmax(Qvalues, dim=-1) * Qvalues # dimension: (batch_size, seq_len, A)
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
            else:
                next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                
            next_Qvalues = torch.cat([next_Qvalues[:, 1:,:], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
            pred_Qvalues =  Qvalues * batch['context_actions']
            pred_Qvalues =  pred_Qvalues.sum(dim=-1, keepdim=True)
            
            TD_target = batch['context_rewards'] + gamma * next_Qvalues
            loss = loss_fn(pred_Qvalues, TD_target)
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        for batch in tqdm.tqdm(test_dataloader):
            # pred_Qvalues = model(batch).gather(2, batch['context_actions'].unsqueeze(-1).long())
            Qvalues = model(batch)
            # hardmax
            if args.soft_max:
                next_Qvalues = F.softmax(Qvalues, dim=-1) * Qvalues
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
            else:
                next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]

            next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
            pred_Qvalues =  Qvalues * batch['context_actions']
            pred_Qvalues =  pred_Qvalues.sum(dim=-1, keepdim=True)
            
            TD_target = batch['context_rewards'] + gamma * next_Qvalues
            loss = loss_fn(pred_Qvalues, TD_target)
            test_loss += loss.item()
        test_loss /= len(test_dataloader)
        print(f"Train Loss {train_loss} | Test Loss {test_loss}")
        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
    # DataLoader can handle batching and shuffling
    model.train()  # Set the model to training mode

    if double:
        target_model = Transformer(config)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        target_model.to(device)

    best_model_state = model.state_dict()
    best_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        counter = 0
        for batch in tqdm.tqdm(train_dataloader):
            Qvalues = model(batch) # dimension: (batch_size, seq_len, A)
            pred_Qvalues = Qvalues * batch['context_actions']
            pred_Qvalues =  pred_Qvalues.sum(dim=-1, keepdim=True)
            
            with torch.no_grad():
                if args.soft_max:
                    next_actions = F.softmax(model(batch), dim=-1) # dimension: (batch_size, seq_len, A)
                else:
                    next_actions = Qvalues.max(dim=-1, keepdim=True)[1] # dimension: (batch_size, seq_len, 1)
                    # one-hot encoding
                    next_actions = F.one_hot(next_actions, num_classes=config['action_dim']).float().squeeze(-2)

                if double:
                    # use target model to get the next Q values
                    # next_Qvalues = target_model(batch).gather(2, next_actions.unsqueeze(-1).long())
                    # next_Qvalues = target_model(batch).gather(2, next_actions.long())
                    next_Qvalues = target_model(batch) * next_actions
                    
                else:
                    # next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                    next_Qvalues = Qvalues * next_actions

                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
                next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
            # Note: batch['context_rewards'] should be in the shape (batch_size, seq_len, 1) to match pred_Qvalues
            TD_target = batch['context_rewards'] + gamma * next_Qvalues
            # Calculate the loss
            loss = loss_fn(pred_Qvalues, TD_target, reduction='mean')
            # Reset the gradients in the optimizer
            optimizer.zero_grad()
            # # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            counter += 1
            # if double:
            #     for param, target_param in zip(model.parameters(), target_model.parameters()):
            #         target_param.data = target_param.data * (1 - 0.01) + param.data * 0.01
            if double:
                if counter % 100 == 0:
                    target_model.load_state_dict(model.state_dict())
            
        epoch_loss /= len(train_dataloader)
        # Compute the test loss
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader):
                # pred_Qvalues = model(batch).gather(2, batch['context_actions'].unsqueeze(-1).long())
                Qvalues = model(batch)
                pred_Qvalues = Qvalues * batch['context_actions']
                pred_Qvalues =  pred_Qvalues.sum(dim=-1, keepdim=True)

                if args.soft_max:
                    next_actions = F.softmax(model(batch), dim=-1)
                else:
                    next_actions = Qvalues.max(dim=-1, keepdim=True)[1]

                
                if double:
                    
                    # next_Qvalues = target_model(batch).gather(2, next_actions.long())
                    next_Qvalues = target_model(batch) * next_actions
                else:
                    # next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                    next_Qvalues = Qvalues * next_actions
                    
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
                next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
                # shifted_pred_Qvalues = torch.cat([pred_Qvalues[:, 1:, :], torch.zeros((pred_Qvalues.shape[0], 1, 1), device=pred_Qvalues.device)], dim=1)
                
                TD_target = batch['context_rewards'] + gamma * next_Qvalues

                loss = loss_fn(pred_Qvalues, TD_target)
                test_loss += loss.item()
        model.train()
        test_loss /= len(test_dataloader)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()

        # wandb logging
        if use_wandb:
            wandb.log({"train_loss": epoch_loss, "test_loss": test_loss})
            print(f"Epoch {epoch} | Train Loss {epoch_loss} | Test Loss {test_loss}")


    best_model = Transformer(config)
    best_model.load_state_dict(best_model_state)
    return best_model

def pretrain(args):

    # parse arguments
    batch_size = args.batch_size
    action_num = args.action_num

    # load data
    traj = load_data(args)
    # pretrain
    # train_traj, test_traj = train_test_split(traj, test_size=0.2)
    # define the train and test size
    train_size = int(0.8 * len(traj))
    test_size = len(traj) - train_size
    train_traj, test_traj = torch.utils.data.random_split(traj, [train_size, test_size])
    # create dataset
    train_dataset = TrajectoryDataset(train_traj, action_num)
    test_dataset = TrajectoryDataset(test_traj, action_num)

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # set up model
    # model = TransformerModel(embed_dim=70, num_heads=5)
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
            'test': False
        }
    action_dim = config['action_dim']
    model = Transformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.use_wandb:
        wandb.login(key = 'da9f2bbc59300d7434138182a83a9b35c0cea793') # login to wandb
        wandb.init(project="icrl_pretrain", reinit=True)

    if args.Q:
        loss_fn = F.mse_loss
        best_model = Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, loss_fn, use_wandb=args.use_wandb, num_epochs=args.num_epochs, gamma=args.gamma, double=args.double)
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        best_model = trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, loss_fn, use_wandb=args.use_wandb, num_epochs=args.num_epochs)

    # save the model
    # Q = 'Q' if args.Q else ''
    # save_path = 'models/' + Q + '_pretrain_' + args.source + '_lr_' + str(args.lr) + '_batch_size_' + str(args.batch_size) + '_n_layer_' + str(args.n_layer) + '_n_embd_' + str(args.n_embd) + '_n_head_' + str(args.n_head) + '_gamma_' + str(args.gamma) + '_data_size_' + str(args.data_size) + '.pth'
    # save_path = 'models/model_'
    # for key, value in config.items():
    #     save_path += f"{key}_{value}_"
    torch.save(best_model.state_dict(), args.model_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain the model')
    parser.add_argument('--source', type=str, default='linucb', help='source of the data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--state_dim', type=int, default=50, help='state dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--action_num', type=int, default=10, help='number of actions')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--horizon', type=int, default=200, help='horizon')
    parser.add_argument('--dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--n_layer', type=int, default=8, help='number of layers')
    parser.add_argument('--n_embd', type=int, default=32, help='embedding dimension')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads')
    parser.add_argument('--gamma', type=float, default=0.5, help='discount factor')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epoches')
    parser.add_argument('--activation', choices=['relu', 'softmax'], default='relu', help='activation function')
    parser.add_argument('--double', action='store_true', help='use double DQN')
    parser.add_argument('--Q', action='store_true', help='train Q function')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--data_size', type=int, default=100000, help='size of training dataset')
    parser.add_argument('--soft_max', action='store_true', help='use softmax')
    parser.add_argument('--model_path', type=str, default='models/model.pth', help='path to save the model')
    parser.add_argument('--max_horizon', type=int, default=500, help='max horizon')
    args = parser.parse_args()
    pretrain(args)