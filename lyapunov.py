import torch


class LyapunovConstraint:
    """Lyapunov fairness constraint: penalizes drift in reward inequality.

    L(s_t) = Gini(cumulative_rewards[t])^2.
    Violation at timestep t = max(0, DeltaL[t] - c * (1 - Gini[t])).
    When Gini is high (unfair), the allowed drift budget shrinks toward zero.

    The violation is distributed differentially across agents in proportion
    to how far each agent's cumulative reward sits above the mean, so only
    above-mean agents pay and starved agents receive no penalty.
    """

    def __init__(self, lambda_lyap=0.1, c=0.5):
        self.lambda_lyap = lambda_lyap
        self.c = c

    def compute_penalty(self, all_rewards, power_multipliers=None):
        """
        Args:
            all_rewards: list of N tensors, each (T, B) — per-agent rewards.
            power_multipliers: optional (N,) tensor of per-agent capacities.
        Returns:
            penalty: (N, T, B) per-agent, per-timestep Lyapunov violation penalty.
        """
        stacked = torch.stack(all_rewards)  # (N, T, B)
        cum_rewards = torch.cumsum(stacked, dim=1)  # (N, T, B)

        gini = self._batch_gini(cum_rewards, power_multipliers)  # (T, B)
        L = gini ** 2  # (T, B) — Gini-squared Lyapunov

        delta_L = torch.zeros_like(L)
        delta_L[0] = L[0]
        delta_L[1:] = L[1:] - L[:-1]

        violation = torch.relu(delta_L - self.c * (1.0 - gini))

        excess = torch.relu(cum_rewards - cum_rewards.mean(dim=0, keepdim=True))  # (N, T, B)
        share = excess / excess.sum(dim=0, keepdim=True).clamp(min=1e-8)  # (N, T, B)

        return self.lambda_lyap * violation.unsqueeze(0) * share

    def _batch_gini(self, cum_rewards, power_multipliers=None):
        """Vectorized Gini coefficient across agents at each (t, b).

        Args:
            cum_rewards: (N, T, B)
            power_multipliers: optional (N,) tensor of per-agent capacities.
        Returns:
            gini: (T, B)
        """
        N = cum_rewards.shape[0]

        shifted = cum_rewards - cum_rewards.min(dim=0, keepdim=True).values
        if power_multipliers is not None:
            shifted = shifted / power_multipliers.view(-1, 1, 1).clamp(min=1e-12)
        sorted_vals, _ = torch.sort(shifted, dim=0)  # (N, T, B)

        weights = (2.0 * torch.arange(1, N + 1, device=cum_rewards.device, dtype=cum_rewards.dtype) - N - 1)
        weights = weights.view(N, 1, 1)

        numerator = (weights * sorted_vals).sum(dim=0)  # (T, B)

        ## TODO check with a simple if sum == 0 then return 0 instead
        denominator = (N * sorted_vals.sum(dim=0)).clamp(min=1e-8)

        return numerator / denominator
