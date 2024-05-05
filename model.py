import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    # Calculate the log probabilities
    # logits: (batch_size, seq_len, num_actions)
    # actions: (batch_size, seq_len)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    
    # Gather the log probabilities for the taken actions
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len), squeeze to remove the last dimension
    
    # The loss is the negative of the sum of action log probabilities
    loss = -action_log_probs.mean()
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