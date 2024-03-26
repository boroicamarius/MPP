import torch.nn as nn
from .attention import Attention

class Block(nn.Module):
    def __init__(self, hidden_dim, context_window, n_heads):
        super(Block, self).__init__()

        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.n_head = n_heads
        self.layers_size = 4 * context_window * hidden_dim

        self.attention_layer = Attention(hidden_dim, context_window, n_heads)
        self.norm_mid_layer = nn.LayerNorm(self.layers_size, self.layers_size)
        self.forward_layer = nn.Linear(self.layers_size, self.layers_size)
        self.norm_end_layer = nn.LayerNorm(self.layers_size, self.layers_size)

    def forward(self, embeds):
        processed_embeds = self.attention_layer(embeds) + embeds
        processed_embeds = self.norm_mid_layer(processed_embeds.view(-1))
        forward_embeds = self.forward_layer(processed_embeds)
        end_embeds = self.norm_end_layer(forward_embeds + processed_embeds)
        return end_embeds.view(-1, self.hidden_dim)
