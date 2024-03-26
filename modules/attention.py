import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,
                 embedding_size,
                 query_size,
                 n_heads, has_mask=False):
        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        self.d_k = query_size
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor):
        dims = x.shape
        head_dims = list(dims)
        head_dims[-1] //= self.n_heads

        repeat_dims = (3, *[1] * (len(dims)))
        qkv = x.view(1, *dims).repeat(*repeat_dims)
        qkv = qkv.view(3, self.n_heads, *head_dims)

        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1))

        p_att = F.softmax(scores / (self.d_k ** 0.5), -1)

        attention = torch.matmul(p_att, v)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view((-1, self.embedding_size,))

        return attention
