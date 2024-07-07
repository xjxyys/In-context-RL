import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
import numpy as np
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError
class DecoderTransformerBackbone(nn.Module):
    def __init__(self, config, activation="relu", normalize_attn=True, mlp=True, layernorm=True, positional_embedding=True):
        super(DecoderTransformerBackbone, self).__init__()
        self.n_positions = config.n_positions
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp
        self.positional_embedding = positional_embedding

        # positional embeddings
        self.wpe = nn.Embedding(self.n_positions, self.n_embd) # dimension (n_positions, n_embd)
        self.wpe.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # layers
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(self.n_layer):
            self._queries.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._keys.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._values.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd]))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(self.n_embd, self.n_embd),
                    nn.ReLU(),
                    nn.Linear(self.n_embd, self.n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))
        
        # pre-compute decoder attention mask
        with torch.no_grad():
            self.mask = torch.zeros(1, self.n_positions, self.n_positions)
            for i in range(self.n_positions):
                if self.normalize_attn:
                    self.mask[0, i, :(i+1)].fill_(1./(i+1)) #normalize the attention mask
                else:
                    self.mask[0, i, :(i+1)].fill_(1.)
    
    def forward(self, inputs_embeds, position_ids=None, return_hidden_states=False):
        # assert inputs_embeds is not None
        # inputs_embeds: (batch_size, seq_len, embed_dim)
        hidden_states = []
        N = inputs_embeds.shape[1]
        H = inputs_embeds

        if self.positional_embedding:
            # Add positional embeddings
            if position_ids is None:
                input_shape = H.size()[:-1] # (batch_size, seq_len), remove the last dimension
                position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=H.device) # (seq_len)
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]) # (batch_size, seq_len)
            position_embeds = self.wpe(position_ids) # (batch_size, seq_len, embed_dim)
            H = H + position_embeds
        hidden_states.append(H)

        for (q, k, v, mlp, ln1, ln2) in zip(self._queries, self._keys, self._values, self._mlps, self._lns_1, self._lns_2):
            # q, k, v: (batch_size, seq_len, embed_dim)
            # ln1, ln2: (batch_size, seq_len, embed_dim)
            # mlp: (batch_size, seq_len, embed_dim)
            # Apply linear transformations
            query = q(H)
            key = k(H)
            value = v(H)

            # Calculate attention scores
            attn_weight = self.activation(torch.einsum('bid,bjd->bij', query, key)) * self.mask[:, :N, :N].to(H.device)
            # attn_weight: (batch_size, seq_len, seq_len)
            H = H + torch.einsum('bij,bjd->bid', attn_weight, value)
            if self.layernorm:
                H = ln1(H)
            if self.mlp:
                # Apply MLP
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)
            hidden_states.append(H)

        if return_hidden_states:
                return H, hidden_states
        return H

class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.test = config['test']
        self.horizon = config['horizon']
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']
        
        self.act_num= self.config['act_num']
        self.dim= self.config['dim']
        self.act_type=self.config['act_type']
        self.pred_q = config['pred_q']

        Model_config = GPT2Config(
            n_positions = 4 * (1+self.horizon),
            n_embd = self.n_embd,
            n_layer = self.n_layer,
            n_head = self.n_head,
            resid_pdrop = self.dropout,
            embd_pdrop = self.dropout,
            attn_pdrop = self.dropout,
            use_cache = False,
        )
        if self.act_type == 'relu':
            self.transformer = DecoderTransformerBackbone(Model_config, activation="relu")
        else:
            self.transformer = GPT2Model(Model_config)

        
        self.embed_transition = nn.Linear(
            3 + self.dim*self.act_num+self.action_dim, self.n_embd #action_dim means the number of actions
        )

        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.pred_qvalues = nn.Linear(self.n_embd, self.action_dim) # for predicting q-values
    
    def forward(self, x):
        # query_states = x['query_states'][:, None, :]
        # if self.test:
        #     batch_size = 1
        #     batch_size = x['action_set'].shape[0]
        # else:
        #     batch_size = x['action_set'].shape[0]
        batch_size = x['action_set'].shape[0]

        current_horizon = x['context_rewards'].size(dim=1) # dimension (batch_size, seq_len)
        action_set_seq = x['action_set']

        # if self.pred_q and self.test:
        #     query_action = x['context_actions'][:, -1, :] # dimension (batch_size, A)
        #     x['context_actions'] = x['context_actions'][:, :-1, :] # dimension (batch_size, H, A) # remove the last action

        # if self.test:
            # action_set_seq = torch.tensor(np.asarray(action_set_seq)).float().to(device)
       
        action_set_seq_res = action_set_seq.reshape(batch_size, 1, -1) # dimension (batch_size, 1, A*D)
        zero_action_set = torch.zeros_like(action_set_seq_res) # dimension (batch_size, 1, A*D)
        action_set_seq = torch.cat([action_set_seq_res, zero_action_set], dim=1) # dimension (batch_size, 2, A*D)
        action_set_seq = action_set_seq.repeat(1, current_horizon, 1) # dimension (batch_size, 2*H, A*D)  # h_{b}
        action_seq = torch.zeros((batch_size, 2*current_horizon, self.action_dim+1), device=device) # dimension (batch_size, 2*H, A+1) # h_{a}

        # if self.pred_q:
        #     # odd numbers for action set and action, even numbers for rewards
        #     action_seq[:, 0::2, :-1] = x['context_actions']  # make the odd layers hold the action set 
        #     action_seq[:, 1::2, -1] = x['context_rewards'].squeeze(-1) # make the even layers hold the rewards
        #     # print(action_seq.size())
        # else:       
        #     action_seq[:, 1::2, :] = torch.cat([x['context_actions'],x['context_rewards']],dim=2)  # make the even layers hold the action actions & rewards
        one_seq=torch.ones((batch_size,2*current_horizon,1),device=device) # dimension (batch_size, 2*H, 1) h_{c}
        pos_seq=torch.arange(1,2*current_horizon+1,dtype=torch.float32, device=device) # dimension (2*H) 
        pos_seq=pos_seq.reshape(1,-1,1).repeat(batch_size,1,1) # dimension (batch_size, 2*H, 1) h_{d}
            # print('action_set_seq size:',action_set_seq.size())
            # print('action_seq size:',action_seq.size())
            # print('one_seq size:',one_seq.size())
            # print('pos_seq size:',pos_seq.size())
        seq=torch.cat([action_set_seq,action_seq,one_seq,pos_seq],dim=2) 

        if self.test:
            action_set_seq_test=torch.zeros((seq.size(dim=0),1,seq.size(dim=2)),device=device) # dimension (batch_size, 1, A*D)

            action_set_seq_test[:,:,:action_set_seq_res.size(dim=2)]=torch.clone(action_set_seq_res)       ## not run when doing dpt test exp
            # if self.pred_q:
            #     action_set_seq_test[:,:,:-2] = query_action.unsqueeze(1) # add the last action to the sequence for test

            action_set_seq_test[:,:,-2]=torch.ones_like(action_set_seq_test[:,:,-2])
            action_set_seq_test[:,:,-1]=(1+2*current_horizon)*torch.ones_like(action_set_seq_test[:,:,-1])
            seq=torch.cat([seq,action_set_seq_test],axis=1) # add H_{2T-1} to the sequence for test
            
        
        # print('seq size:',seq.size())
        stacked_inputs = self.embed_transition(seq)   
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)

        ## output action
        if self.act_type=='relu':
            # if self.pred_q:
            #     preds = self.pred_qvalues(transformer_outputs)
            # else:
                preds = self.pred_actions(transformer_outputs)    ## for relu  
        else:
            # if self.pred_q:
            #     preds = self.pred_qvalues(transformer_outputs['last_hidden_state'])
            # else:
                preds = self.pred_actions(transformer_outputs['last_hidden_state'])   ##for gpt2
        # print('preds size:',preds.size())
        if self.test:
            return preds[:, -1, :]   
        return preds[:, 0::2, :]   ##get the odd layers # dimension (batch_size, H, A)

# trainer 
def trainer(model, train_dataloader, test_dataloader, optimizer, num_epochs=10):
    # DataLoader can handle batching and shuffling
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_dataloader):
            pred_actions = model(batch) # dimension: (batch_size, seq_len, action_dim)
            # print(pred_actions.shape)
            action_dim = pred_actions.size(-1)
            true_actions = batch['true_actions'] # dimension: (batch_size, seq_len)
            pred_actions_flat =  pred_actions.view(-1, action_dim)
            true_actions_flat = true_actions.view(-1)
            loss = loss_fn(pred_actions_flat, true_actions_flat)
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
        # wandb logging
        wandb.log({"train_loss": epoch_loss, "test_loss": test_loss})
        print(f"Epoch {epoch} | Train Loss {epoch_loss} | Test Loss {test_loss}")

# Q_trainer
# Trainer
def Q_trainer(model, train_dataloader, test_dataloader, optimizer, num_epochs=10, gamma = 0.99, double=False):
    # record the loss before training
    train_loss = 0.0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            Qvalues = model(batch) # dimension: (batch_size, seq_len, A)
            # pred_Qvalues = pred_Qvalues.gather(2, batch['context_actions'].unsqueeze(-1).long()) # get the Q value of the context actions
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
        global config
        target_model = Transformer(config)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        target_model.to(device)

    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        counter = 0
        for batch in tqdm.tqdm(train_dataloader):
            Qvalues = model(batch) # dimension: (batch_size, seq_len, A)
            pred_Qvalues = Qvalues * batch['context_actions']
            pred_Qvalues =  pred_Qvalues.sum(dim=-1, keepdim=True)
            
            with torch.no_grad():
                next_actions = model(batch).max(dim=-1, keepdim=True)[1]

                if double:
                    # use target model to get the next Q values
                    # next_Qvalues = target_model(batch).gather(2, next_actions.unsqueeze(-1).long())
                    next_Qvalues = target_model(batch).gather(2, next_actions.long())
                else:
                    next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
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
                next_actions = model(batch).max(dim=-1, keepdim=True)[1]
                if double:
    
                    next_Qvalues = target_model(batch).gather(2, next_actions.long())
                else:
                    next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                    
                next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
                # shifted_pred_Qvalues = torch.cat([pred_Qvalues[:, 1:, :], torch.zeros((pred_Qvalues.shape[0], 1, 1), device=pred_Qvalues.device)], dim=1)
                
                TD_target = batch['context_rewards'] + gamma * next_Qvalues

                loss = loss_fn(pred_Qvalues, TD_target)
                test_loss += loss.item()
        model.train()
        test_loss /= len(test_dataloader)
        # wandb logging
        wandb.log({"train_loss": epoch_loss, "test_loss": test_loss})
        print(f"Epoch {epoch} | Train Loss {epoch_loss} | Test Loss {test_loss}")


###################################################################################
## Previous implementation of the Transformer model
###################################################################################
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
    
    def forward(self, x, attn_mask):
        # print('start transformer!!!')
        # Assuming attn_mask is correctly sized but check its permutation if needed
        # attn_mask = attn_mask.permute(1, 0, 2) 
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_output
        x = self.ln1(x)
        
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
    
    def forward(self, x, attn_mask):
        # x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

class TransformerModel(nn.Module):
    def __init__(self, num_layers=8, embed_dim=70, num_heads=4, num_actions=10, context_dim=5):
        super(TransformerModel, self).__init__()
        self.transformer_decoder = TransformerDecoder(num_layers, embed_dim, num_heads)
        self.num_actions = num_actions
        self.context_dim = context_dim
    
    def generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones((sz, sz), dtype=torch.float), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        # mask = mask.repeat(batch_size, 1, 1)
        return mask
    
    def forward(self, x):
        # Generate causal mask
        # input_ids: (batch_size, seq_len, embed_dim)
        # print(input_ids.size())
        # print(x.device)
        attn_mask = self.generate_causal_mask(x.size(1)) # (batch_size, seq_len, seq_len)
        attn_mask = attn_mask.to(x.device)
        # print(attn_mask.device)
        # if attn_mask is not None and attn_mask.dim() == 2:
        #     print('attn_mask is not None and attn_mask.dim() == 2')
        #     attn_mask = attn_mask.unsqueeze(0)  # Adding batch dimension if needed
        # # print(attn_mask.size())
        # input_ids = input_ids.long()
        # Pass through the transformer decoder
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        post = self.transformer_decoder(x, attn_mask)
        post = post.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)

        # print('finish transformer!!!')
        # Calculate the start index for slicing logits
        start_idx = self.context_dim + 1 + self.context_dim * self.num_actions
        end_idx = start_idx + self.num_actions

        # print(start_idx, end_idx)
        # Extract the logits for h^c_{2t-1}
        # Assuming post is of shape (batch_size, seq_len, total_dim)
        # and we want logits corresponding to every second element in the sequence starting from the first
        logits = post[:, 0::2, start_idx:end_idx]
        return logits




# Loss function based on negative log likelihood
def loss_fn(logits, actions):
    # logits: (batch_size, seq_len, num_actions)
    # actions: (batch_size, seq_len) where each entry is an integer representing the action index
    # print(logits.shape, actions.shape)
    # Calculate log probabilities of all actions
    log_probs = F.log_softmax(logits, dim=-1)  # Apply log_softmax on the last dimension to get log probabilities

    # Gather the log probabilities for the actions taken
    # actions.unsqueeze(-1) adds an extra dimension, making it (batch_size, seq_len, 1)
    # so we can gather along the num_actions dimension
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # Remove the last dimension after gather

    # Calculate the negative sum of these log probabilities
    # Summing over all sequence lengths and batches
    loss = -action_log_probs.sum()
    return loss

# Create a model instance
# model = TransformerModel()

# Trainer
def trainer(model, dataset, optimizer, num_epochs=10):
    # DataLoader can handle batching and shuffling
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for tokens, action_labels in dataloader:
            # Reset the gradients in the optimizer
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(tokens)  # Your forward method should handle the causal mask internally
            
            # Compute loss
            loss = loss_fn(logits.view(-1, model.num_actions), torch.tensor(action_labels).view(-1))
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')