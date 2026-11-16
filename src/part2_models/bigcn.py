import torch
import torch.nn as nn
import torch.nn.functional as F

def native_global_mean_pool(x, batch, size=None):
    num_graphs = int(batch.max().item() + 1) if size is None else size
    out = torch.zeros((num_graphs, x.shape[1]), device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    count = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
    count.scatter_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / count.clamp(min=1).unsqueeze(-1)

def drop_edges(edge_index, p, training):
    if not training or p <= 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= p
    return edge_index[:, keep]

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

class DirectionalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = NativeGCNConv(in_channels, hidden_dim)
        self.conv2 = NativeGCNConv(hidden_dim + in_channels, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, root_idx):
        h1 = self.conv1(x, edge_index)
        h1_root = h1[root_idx][batch]
        h = torch.cat([h1, x[root_idx][batch]], dim=-1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h2 = F.relu(self.conv2(h, edge_index))
        h2 = torch.cat([h2, h1_root], dim=-1)
        return native_global_mean_pool(h2, batch)

class BiGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, dropout=0.5, edge_drop=0.2):
        super().__init__()
        self.td = DirectionalGCN(in_channels, hidden_dim, hidden_dim, dropout)
        self.bu = DirectionalGCN(in_channels, hidden_dim, hidden_dim, dropout)
        self.edge_drop = edge_drop
        self.fc = nn.Linear(hidden_dim * 4, out_channels)

    def forward(self, x, edge_index, batch, ptr=None):
        if ptr is not None:
            root_idx = ptr[:-1]
        else:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root_idx = torch.cat([batch.new_zeros(1, dtype=torch.long), root + 1], dim=0)
        td_edges = drop_edges(edge_index, self.edge_drop, self.training)
        bu_edges = drop_edges(edge_index, self.edge_drop, self.training)[[1, 0]]
        s_td = self.td(x, td_edges, batch, root_idx)
        s_bu = self.bu(x, bu_edges, batch, root_idx)
        return self.fc(torch.cat([s_td, s_bu], dim=-1)).squeeze(-1)
