import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        assert (
            self.head_dim * heads == dim
        ), "Embedding dimension must be divisible by heads"

        self.values = nn.Linear(dim, dim, bias=False)
        self.keys = nn.Linear(dim, dim, bias=False)
        self.queries = nn.Linear(dim, dim, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Split into heads
        values = values.view(N, seq_length, self.heads, self.head_dim)
        keys = keys.view(N, seq_length, self.heads, self.head_dim)
        queries = queries.view(N, seq_length, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)  # (N, heads, seq_length, head_dim)
        keys = keys.permute(0, 2, 1, 3)      # (N, heads, seq_length, head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (N, heads, seq_length, head_dim)

        energy = torch.einsum("nqhd,nkhd->nqk", [queries, keys])  # (N, heads, seq_length, seq_length)
        attention = F.softmax(energy / (self.dim ** (1 / 2)), dim=2)

        out = torch.einsum("nqk,nkhd->nqhd", [attention, values]).reshape(
            N, seq_length, self.heads * self.head_dim
        )

        return self.fc_out(out)


class TransformerModel(nn.Module):
    def __init__(self, dim=192, vocab_size=22, heads=8, num_layers=3):
        super(TransformerModel, self).__init__()
        self.attention_layers = nn.ModuleList(
            [SelfAttention(dim, heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        for layer in self.attention_layers:
            x = layer(x)
        return self.fc_out(x)
