import torch.nn as nn
from modules import Block, Finalizer, Preprocess

class DecisionTransformer(nn.Module):
    def __init__(self,
                 context_window,
                 hidden_dim,
                 n_heads,
                 n_decoders,
                 reward_dim,
                 map_dim,
                 path_dim,
                 action_dim
                 ):
        """
        :param context_window: size of window
        :param hidden_dim: Dimension of token after translation (HIDDEN_DIM,)
        :param n_heads: Number of attention heads
        :param reward_dim: Dimension of reward, shape of reward (CONTEXT_WINDOW, REWARD_DIM,)
        :param map_dim: Dimension of map (CONTEXT_WINDOW, MAP_DIM,), map_dim is a number = width*height
        :param path_dim: Dimension of path (CONTEXT_WINDOW, PATH_DIM, 4,)
        :param action_dim: Dimension of action (CONTEXT_WINDOW, 1,)
        """
        if hidden_dim % n_heads != 0:
            raise Exception("Hidden dim cant be divided by number of attention heads")

        super(DecisionTransformer, self).__init__()
        self.preprocess = Preprocess(context_window, hidden_dim, n_heads,
                                     reward_dim, map_dim, path_dim, action_dim)
        self.decoders = nn.Sequential(*[Block(hidden_dim, context_window, n_heads) for _ in range(n_decoders)])
        self.finalizers = Finalizer(context_window, hidden_dim)

    def forward(self, rewards, maps, paths, actions):
        embeds = self.preprocess(rewards, maps, paths, actions)
        values = self.decoders(embeds)
        outputs = self.finalizers(values)
        return outputs
