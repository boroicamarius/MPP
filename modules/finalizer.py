import torch
import torch.nn as nn

class Finalizer(nn.Module):
    def __init__(self,
                 context_window,
                 hidden_dim):
        super(Finalizer, self).__init__()
        self.actions_layer = nn.Linear(4 * context_window * hidden_dim, 5)
        self.actions_sigmoid = nn.Sigmoid()
        self.actions_relu = nn.ReLU()

        self.reward_layer = nn.Linear(4 * context_window * hidden_dim, 1)
        self.reward_relu = nn.ReLU()

    def forward(self, embeddings: torch.Tensor):
        flattened_embeddings = embeddings.view(-1)
        actions = self.actions_layer(flattened_embeddings)
        actions = self.actions_sigmoid(actions)
        actions = self.actions_relu(actions)*4

        reward = self.reward_layer(flattened_embeddings)
        reward = self.reward_relu(reward)

        return torch.cat((actions, reward))
