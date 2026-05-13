"""Continuous domain subspace + soft mixing (paper Eqs. 9–10 style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousDomainSubspace(nn.Module):
    def __init__(self, d_in: int, d_u: int, num_domains: int, mix_temperature: float = 0.1):
        super().__init__()
        self.K = num_domains
        self.tau = mix_temperature
        self.P = nn.Parameter(torch.randn(self.K, d_u, d_in) * 0.02)
        self.b = nn.Parameter(torch.zeros(self.K, d_u))
        self.prototypes = nn.Parameter(torch.randn(self.K, d_in) * 0.02)

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        """h_cls: (B, d_in) -> z_u: (B, d_u)"""
        z_per_k = torch.einsum("kdu,bd->bku", self.P, h_cls) + self.b.unsqueeze(0)
        z_per_k = F.relu(z_per_k)
        logits = torch.einsum("kd,bd->bk", self.prototypes, h_cls) / self.tau
        w = F.softmax(logits, dim=-1)
        return (w.unsqueeze(-1) * z_per_k).sum(dim=1)

    def projection_orth_penalty(self) -> torch.Tensor:
        """Penalize non-orthogonal projection pairs (paper L_proj)."""
        loss = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        for d in range(self.K):
            for d2 in range(self.K):
                if d == d2:
                    continue
                M = self.P[d].transpose(0, 1) @ self.P[d2]
                loss = loss + (M ** 2).sum()
        n_pairs = max(1, self.K * (self.K - 1))
        return loss / n_pairs
