
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TinyTransformerBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        # 1. Self Attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])

        weights = F.softmax(scores, dim=-1)

        attention = weights @ v

        # 2. Residual Connection + LayerNorm
        x = self.norm1(x + attention)

        # 3. Feed Forward Network
        ff_out = F.relu(self.ff1(x))
        ff_out = self.ff2(ff_out)

        # 4. Residual Connection + LayerNorm
        out = self.norm2(x + ff_out)

        return out, weights


# Input
x = torch.tensor([
    [2., 3., 8., 9.],
    [0., 2., 1., 1.],
    [3., 4., 1., 2.],
    [1., 0., 1., 0.]
])

# Model
model = TinyTransformerBlock(d_model=4, d_ff=8)

# Forward pass
output, attention_weights = model(x)

print("Input shape:", x.shape)
print("Attention weights shape:", attention_weights.shape)
print("Attention weights:\n", attention_weights)

print("Output shape:", output.shape)
print("Output:\n", output)

# Parameters
params = sum(p.numel() for p in model.parameters())

print("Trainable parameters:", params)
