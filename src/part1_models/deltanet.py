import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class DeltaAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, d_head=64, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.inner_dim = n_heads * d_head
        self.d_conv = d_conv

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.gate_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.q_conv = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=d_conv, padding=d_conv - 1, groups=self.inner_dim)
        self.k_conv = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=d_conv, padding=d_conv - 1, groups=self.inner_dim)
        self.v_conv = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=d_conv, padding=d_conv - 1, groups=self.inner_dim)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        D = self.d_head

        q_raw = self.q_conv(self.q_proj(x).transpose(1, 2))[:, :, :L].transpose(1, 2)
        k_raw = self.k_conv(self.k_proj(x).transpose(1, 2))[:, :, :L].transpose(1, 2)
        v_raw = self.v_conv(self.v_proj(x).transpose(1, 2))[:, :, :L].transpose(1, 2)

        q = F.silu(q_raw).view(B, L, H, D)
        k = F.silu(k_raw).view(B, L, H, D)
        v = F.silu(v_raw).view(B, L, H, D)

        beta = torch.sigmoid(self.beta_proj(x)).view(B, L, H, 1)
        gate = F.silu(self.gate_proj(x).view(B, L, H, D))

        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        o_list = []

        k_all = k_norm.unsqueeze(-1).unbind(dim=1)
        v_all = v.unsqueeze(-1).unbind(dim=1)
        q_all = q_norm.unsqueeze(-1).unbind(dim=1)
        b_all = beta.unsqueeze(-1).unbind(dim=1)

        for k_t, v_t, q_t, b_t in zip(k_all, v_all, q_all, b_all):
            v_pred = torch.matmul(S, k_t)
            err = v_t - v_pred
            S = S + b_t * torch.matmul(err, k_t.transpose(-1, -2))
            o_t = torch.matmul(S, q_t).squeeze(-1)
            o_list.append(o_t)

        o = torch.stack(o_list, dim=1)
        o = o * gate
        return self.out_proj(o.reshape(B, L, self.inner_dim))

class DeltaNetDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, d_head=64, d_ff=1024, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = DeltaAttention(d_model, n_heads=n_heads, d_head=d_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

class DeltaNetCipherModel(nn.Module):
    def __init__(self, vocab_size=65, num_classes=27, d_model=256, n_heads=4, d_head=64, d_ff=1024, n_layers=4, dropout=0.1, max_seq_len=16384):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DeltaNetDecoderLayer(d_model, n_heads=n_heads, d_head=d_head, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)
