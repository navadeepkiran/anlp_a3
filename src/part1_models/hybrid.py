import math
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

class InternalDeltaAttention(nn.Module):
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

        for t in range(L):
            k_t = k_norm[:, t, :, :].unsqueeze(-1)
            v_t = v[:, t, :, :].unsqueeze(-1)
            q_t = q_norm[:, t, :, :].unsqueeze(-1)
            b_t = beta[:, t, :, :].unsqueeze(-1)

            v_pred = torch.matmul(S, k_t)
            err = v_t - v_pred
            S = S + b_t * torch.matmul(err, k_t.transpose(-1, -2))

            o_t = torch.matmul(S, q_t).squeeze(-1)
            o_list.append(o_t)

        o = torch.stack(o_list, dim=1)
        o = o * gate
        return self.out_proj(o.reshape(B, L, self.inner_dim))

class HybridDeltaBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, d_head=64, d_conv=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = InternalDeltaAttention(d_model, n_heads=n_heads, d_head=d_head, d_conv=d_conv)
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

class InternalSelectiveSSM(nn.Module):
    def __init__(self, d_inner, d_state=16, dt_rank=16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u):
        B, L, D = u.shape
        N = self.d_state

        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(u)
        delta_raw = x_dbl[:, :, :self.dt_rank]
        B_t = x_dbl[:, :, self.dt_rank:self.dt_rank + N]
        C_t = x_dbl[:, :, self.dt_rank + N:]

        delta = F.softplus(self.dt_proj(delta_raw))

        delta_A = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        delta_B_u = torch.einsum("bld,bln,bld->bldn", delta, B_t, u)

        h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)
        y_list = []
        for t in range(L):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C_t[:, t])
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        return y + u * self.D

class HybridMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, d_ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.norm1 = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )
        self.ssm = InternalSelectiveSSM(d_inner=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
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
        res = x
        x_norm = self.norm1(x)
        B, L, D = x_norm.shape
        xz = self.in_proj(x_norm)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(z)
        mamba_out = self.out_proj(y)
        x = res + self.dropout1(mamba_out)
        x = x + self.ffn(self.norm2(x))
        return x

class HybridCipherModel(nn.Module):
    def __init__(self, vocab_size=65, num_classes=27, d_model=256, d_state=16, d_conv=4, expand=2, n_heads=4, d_head=64, d_ff=1024, n_layers=4, dropout=0.1, max_seq_len=16384):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        layers = []
        for i in range(n_layers):
            if i % 2 == 0:
                layers.append(HybridMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, d_ff=d_ff, dropout=dropout))
            else:
                layers.append(HybridDeltaBlock(d_model, n_heads=n_heads, d_head=d_head, d_conv=d_conv, d_ff=d_ff, dropout=dropout))

        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)
