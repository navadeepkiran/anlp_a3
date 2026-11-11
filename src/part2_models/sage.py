import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv as PyG_SAGEConv
    from torch_geometric.nn import global_max_pool as pyg_global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

def native_global_max_pool(x, batch, size=None):
    num_graphs = int(batch.max().item() + 1) if size is None else size
    out = torch.full((num_graphs, x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=False)
    return torch.nan_to_num(out, neginf=0.0)

class NativeSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp_(min=1)

        msg = x[src] / deg[dst].unsqueeze(-1)
        aggr = torch.zeros_like(x)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        out = self.lin_l(aggr) + self.lin_r(x)
        return out

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, dropout=0.5):
        super().__init__()
        if HAS_PYG:
            self.conv1 = PyG_SAGEConv(in_channels, hidden_dim)
            self.conv2 = PyG_SAGEConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = NativeSAGEConv(in_channels, hidden_dim)
            self.conv2 = NativeSAGEConv(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, ptr=None):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)

        if HAS_PYG:
            h_pool = pyg_global_max_pool(h, batch)
        else:
            h_pool = native_global_max_pool(h, batch)

        return self.fc(h_pool).squeeze(-1)
