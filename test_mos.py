#!/usr/bin/env python3
"""Quick test to verify MoS implementation works correctly."""

import torch
import torch.nn.functional as F

# Mock CastedLinear for testing
class CastedLinear(torch.nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class MixtureOfSoftmax(torch.nn.Module):
    """Mixture of Softmax output layer for breaking the softmax bottleneck."""

    def __init__(self, model_dim: int, vocab_size: int, n_mixtures: int = 2, rank: int = 0):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.rank = rank

        if rank > 0:
            self.proj_down = CastedLinear(model_dim, rank, bias=False)
            self.proj_up = CastedLinear(rank, n_mixtures * model_dim, bias=False)
            torch.nn.init.normal_(self.proj_down.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.proj_up.weight, mean=0.0, std=0.02)
        else:
            self.projections = CastedLinear(model_dim, n_mixtures * model_dim, bias=False)
            torch.nn.init.normal_(self.projections.weight, mean=0.0, std=0.02)

        self.gate = CastedLinear(model_dim, n_mixtures, bias=False)
        torch.nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = hidden.shape
        K = self.n_mixtures

        pi = F.softmax(self.gate(hidden), dim=-1)

        if self.rank > 0:
            projected = self.proj_up(self.proj_down(hidden)).view(bsz, seq_len, K, dim)
        else:
            projected = self.projections(hidden).view(bsz, seq_len, K, dim)

        logits = torch.einsum('bskd,vd->bskv', projected, weight_matrix)

        log_probs = F.log_softmax(logits, dim=-1)
        log_pi = torch.log(pi.unsqueeze(-1) + 1e-10)
        mixed_log_probs = torch.logsumexp(log_probs + log_pi, dim=2)

        return mixed_log_probs


def test_mos():
    """Test MoS forward pass."""
    print("Testing MoS implementation...")

    vocab_size = 1024
    model_dim = 512
    batch_size = 2
    seq_len = 16

    hidden = torch.randn(batch_size, seq_len, model_dim)
    weight_matrix = torch.randn(vocab_size, model_dim)

    for K in [1, 2, 3]:
        for rank in [0, 32, 64]:
            mos = MixtureOfSoftmax(model_dim, vocab_size, n_mixtures=K, rank=rank)
            output = mos(hidden, weight_matrix)

            assert output.shape == (batch_size, seq_len, vocab_size), f"Wrong shape: {output.shape}"

            # Verify output is valid log probabilities
            probs = torch.exp(output)
            prob_sum = probs.sum(dim=-1)
            assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4), \
                f"K={K} rank={rank}: probs don't sum to 1: {prob_sum.mean():.6f}"

            # Count parameters
            params = sum(p.numel() for p in mos.parameters())
            size_kb = params / 1024
            print(f"  K={K} rank={rank:>3d}: {params:>10,} params ({size_kb:>7.1f} KB at int8)")

    # Verify NLL loss works correctly with MoS output
    mos = MixtureOfSoftmax(model_dim, vocab_size, n_mixtures=2, rank=64)
    output = mos(hidden, weight_matrix)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = F.nll_loss(output.reshape(-1, vocab_size), targets.reshape(-1))
    assert loss.isfinite(), f"NLL loss is not finite: {loss}"
    print(f"\n  NLL loss test: {loss.item():.4f} (should be ~6.93 for random)")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_mos()
