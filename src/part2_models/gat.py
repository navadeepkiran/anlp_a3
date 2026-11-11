import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv as PyG_GATConv
    from torch_geometric.nn import global_max_pool as pyg_global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

def native_global_max_pool(x, batch, size=None):
    num_graphs = int(batch.max().item() + 1) if size is None else size
    out = torch.full((num_graphs, x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=False)
    return torch.nan_to_num(out, neginf=0.0)

class NativeGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.zeros(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        x_h = self.lin(x).view(num_nodes, self.heads, self.out_channels)
        alpha_src = (x_h * self.att_src).sum(dim=-1)
        alpha_dst = (x_h * self.att_dst).sum(dim=-1)

        alpha = alpha_src[src] + alpha_dst[dst]
        alpha = F.leaky_relu(alpha, 0.2)

        alpha_exp = torch.exp(alpha)
        alpha_sum = torch.zeros(num_nodes, self.heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha_exp), alpha_exp)
        alpha_norm = alpha_exp / alpha_sum[dst].clamp(min=1e-6)
        alpha_norm = F.dropout(alpha_norm, p=self.dropout, training=self.training)

        msg = x_h[src] * alpha_norm.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)

        return out.view(num_nodes, self.heads * self.out_channels)

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, heads=4, out_channels=1, dropout=0.5):
        super().__init__()
        per_head = hidden_dim // heads

        if HAS_PYG:
            self.conv1 = PyG_GATConv(in_channels, per_head, heads=heads, dropout=dropout)
            self.conv2 = PyG_GATConv(hidden_dim, per_head, heads=heads, dropout=dropout)
        else:
            self.conv1 = NativeGATConv(in_channels, per_head, heads=heads, dropout=dropout)
            self.conv2 = NativeGATConv(hidden_dim, per_head, heads=heads, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, ptr=None):
        h = F.elu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index))
        h = self.dropout(h)

        if HAS_PYG:
            h_pool = pyg_global_max_pool(h, batch)
        else:
            h_pool = native_global_max_pool(h, batch)

        return self.fc(h_pool).squeeze(-1)
