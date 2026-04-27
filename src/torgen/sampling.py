# src/torgen/sampling.py
"""Sample tornado outbreaks from MDN mixture parameters."""
import torch


def sample_outbreak(
    mdn_params: dict[str, torch.Tensor],
    batch_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """Sample a tornado outbreak from MDN mixture parameters.

    Args:
        mdn_params: dict from MDNHead with pi, mu, L, ef_logits.
        batch_idx: which sample in the batch to use.

    Returns:
        Dict with keys:
            coords: (N, 2) tornado locations in [0, 1]^2
            ef: (N,) integer EF ratings
            count: int, number of tornadoes
            component_idx: (N,) which component each tornado came from
    """
    pi = mdn_params["pi"][batch_idx]          # (K,)
    mu = mdn_params["mu"][batch_idx]          # (K, 2)
    L = mdn_params["L"][batch_idx]            # (K, 2, 2)
    ef_logits = mdn_params["ef_logits"][batch_idx]  # (K, n_ef)

    # Expected total count -- round to integer, no Poisson noise
    # All stochasticity comes from z affecting pi values
    Lambda = pi.sum()
    N = max(round(Lambda.item()), 0)

    if N == 0:
        return {
            "coords": torch.zeros(0, 2),
            "ef": torch.zeros(0, dtype=torch.long),
            "count": 0,
            "component_idx": torch.zeros(0, dtype=torch.long),
        }

    # Sample which component each tornado comes from
    probs = pi / pi.sum()
    comp_idx = torch.multinomial(probs, N, replacement=True)  # (N,)

    # Sample locations from each component's Gaussian
    comp_mu = mu[comp_idx]  # (N, 2)
    comp_L = L[comp_idx]    # (N, 2, 2)

    # x = mu + L @ z where z ~ N(0, I)
    z = torch.randn(N, 2, 1, device=mu.device)
    coords = comp_mu + (comp_L @ z).squeeze(-1)  # (N, 2)
    coords = coords.clamp(0, 1)

    # Sample EF
    comp_ef_logits = ef_logits[comp_idx]  # (N, n_ef)
    ef_probs = torch.softmax(comp_ef_logits, dim=-1)
    ef = torch.multinomial(ef_probs, 1).squeeze(-1)  # (N,)

    return {
        "coords": coords,
        "ef": ef,
        "count": N,
        "component_idx": comp_idx,
    }
