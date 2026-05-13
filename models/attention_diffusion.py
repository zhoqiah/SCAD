"""Paper-style attention diffusion on a cosine-similarity token graph (after DADGNN-refined H)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDiffusion(nn.Module):
    """
    Build sparse A from cosine similarity > tau_s, symmetric normalize, then
    s^{t+1} = (1-gamma) * A_tilde @ s^t + gamma * s^0, with
    s^0 = softmax(H @ q_s) (masked). Output z_s = W_s (sum_j s^T_j h_j).
    """

    def __init__(self, d_model: int, d_s: int, tau_s: float = 0.65, gamma: float = 0.15, T: int = 5):
        super().__init__()
        self.tau_s = tau_s
        self.gamma = gamma
        self.T = T
        self.q_s = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.q_s, std=0.02)
        self.Ws = nn.Linear(d_model, d_s)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        H: (B, L, D), mask: (B, L) non-zero for valid tokens
        """
        B, L, _ = H.shape
        m = (mask > 0).float()
        m2 = m.unsqueeze(1) * m.unsqueeze(2)

        h_norm = F.normalize(H, p=2, dim=-1, eps=1e-8)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))
        sim = sim.masked_fill(m2 == 0, 0.0)
        A = sim * (sim > self.tau_s).float()
        A = A * m2

        diag_vals = torch.diagonal(A, dim1=-2, dim2=-1)
        diag_vals = torch.where(m > 0, torch.clamp(diag_vals, min=1.0), diag_vals)
        eye = torch.eye(L, device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
        A = A * (1.0 - eye) + torch.diag_embed(diag_vals)

        deg = A.sum(dim=-1).clamp(min=1e-6)
        inv_sqrt = deg.rsqrt()
        A_tilde = inv_sqrt.unsqueeze(1) * A * inv_sqrt.unsqueeze(2)

        scores = torch.matmul(H, self.q_s)
        scores = scores.masked_fill(m == 0, float("-inf"))
        s0 = torch.softmax(scores, dim=-1)
        s0 = s0 * m
        s0 = s0 / (s0.sum(dim=-1, keepdim=True) + 1e-8)
        s0 = torch.where(torch.isfinite(s0), s0, torch.zeros_like(s0))

        s = s0
        for _ in range(self.T):
            s = (1.0 - self.gamma) * torch.bmm(A_tilde, s.unsqueeze(-1)).squeeze(-1) + self.gamma * s0

        ctx = (s.unsqueeze(-1) * H).sum(dim=1)
        return self.Ws(ctx)
