import torch
import torch.nn as nn
import torch.nn.functional as F

class RootMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, ptr=None):
        if ptr is not None:
            root_idx = ptr[:-1]
        else:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root_idx = torch.cat([batch.new_zeros(1, dtype=torch.long), root + 1], dim=0)
        x_root = x[root_idx]
        h = F.relu(self.fc1(x_root))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)

