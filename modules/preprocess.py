import torch
import torch.nn as nn


class Preprocess(nn.Module):
    def __init__(self,
                 context_window,
                 hidden_dim,
                 n_heads,
                 reward_dim,
                 map_dim,
                 path_dim,
                 action_dim):
        super(Preprocess, self).__init__()
        self.context_window = context_window * 4
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.reward_dim = reward_dim
        self.map_dim = map_dim
        self.path_dim = path_dim
        self.action_dim = action_dim

        self.reward_embed = nn.Linear(reward_dim, hidden_dim)
        self.map_embed = nn.Linear(map_dim, hidden_dim)
        self.path_embed = nn.Linear(4 * path_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.timestep_embed = nn.Embedding(context_window, hidden_dim)

    def forward(self, rewards: torch.Tensor, maps: torch.Tensor, paths: torch.Tensor, actions: torch.Tensor):
        reward_tokens = self.reward_embed(rewards)
        map_tokens = self.map_embed(maps)
        path_tokens = self.path_embed(paths)
        action_tokens = self.action_embed(actions)

        embeds = torch.zeros((self.context_window, self.hidden_dim), device='cuda')
        embeds[:len(reward_tokens) * 4:4] = reward_tokens
        embeds[1:len(map_tokens) * 4:4] = map_tokens
        embeds[2:len(path_tokens) * 4:4] = path_tokens
        embeds[3:len(action_tokens) * 4:4] = action_tokens

        timesteps = torch.arange(0, self.context_window / 4, dtype=torch.long, device='cuda')
        timesteps = timesteps.unsqueeze(0).repeat(1, 4).flatten()
        timesteps = self.timestep_embed(timesteps)

        embeds += timesteps
        return embeds
