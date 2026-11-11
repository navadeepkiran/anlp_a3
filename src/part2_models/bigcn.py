import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv as PyG_GCNConv
    from torch_geometric.nn import global_max_pool as pyg_global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

def native_global_max_pool(x, batch, size=None):
    num_graphs = int(batch.max().item() + 1) if size is None else size
    out = torch.full((num_graphs, x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=False)
    return torch.nan_to_num(out, neginf=0.0)

class NativeGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        loop = torch.arange(0, num_nodes, device=x.device, dtype=edge_index.dtype)
        edge_with_self = torch.cat([edge_index, torch.stack([loop, loop], dim=0)], dim=1)

        src, dst = edge_with_self[0], edge_with_self[1]
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        x_mapped = self.linear(x)

        msg = x_mapped[src] * norm.unsqueeze(-1)
        out = torch.zeros_like(x_mapped)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out

class BiGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, dropout=0.5):
        super().__init__()
        if HAS_PYG:
            self.td_conv1 = PyG_GCNConv(in_channels, hidden_dim)
            self.td_conv2 = PyG_GCNConv(hidden_dim, hidden_dim)
            self.bu_conv1 = PyG_GCNConv(in_channels, hidden_dim)
            self.bu_conv2 = PyG_GCNConv(hidden_dim, hidden_dim)
        else:
            self.td_conv1 = NativeGCNConv(in_channels, hidden_dim)
            self.td_conv2 = NativeGCNConv(hidden_dim, hidden_dim)
            self.bu_conv1 = NativeGCNConv(in_channels, hidden_dim)
            self.bu_conv2 = NativeGCNConv(hidden_dim, hidden_dim)

        self.root_fc = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, ptr=None):
        rev_edge_index = edge_index[[1, 0]]

        h_td = F.relu(self.td_conv1(x, edge_index))
        h_td = self.dropout(h_td)
        h_td = F.relu(self.td_conv2(h_td, edge_index))
        h_td = self.dropout(h_td)

        h_bu = F.relu(self.bu_conv1(x, rev_edge_index))
        h_bu = self.dropout(h_bu)
        h_bu = F.relu(self.bu_conv2(h_bu, rev_edge_index))
        h_bu = self.dropout(h_bu)

        if HAS_PYG:
            p_td = pyg_global_max_pool(h_td, batch)
            p_bu = pyg_global_max_pool(h_bu, batch)
        else:
            p_td = native_global_max_pool(h_td, batch)
            p_bu = native_global_max_pool(h_bu, batch)

        if ptr is not None:
            root_idx = ptr[:-1]
        else:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root_idx = torch.cat([batch.new_zeros(1, dtype=torch.long), root + 1], dim=0)
        x_root = F.relu(self.root_fc(x[root_idx]))
        x_root = self.dropout(x_root)

        combined = torch.cat([p_td, p_bu, x_root], dim=-1)
        return self.fc(combined).squeeze(-1)
