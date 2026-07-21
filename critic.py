import torch
import torch.nn as nn


class Critic(nn.Module):
    """Two-layer MLP critic for value estimation (Appendix B.3).

    Takes the GRU encoder output concatenated with the current time step
    and produces a scalar state-value estimate V(s).
    """

    def __init__(self, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, encoder_out, time_step):
        """
        Args:
            encoder_out: (batch, hidden_size) output from the GRU encoder
            time_step:   (batch, 1) current step index in the episode

        Returns:
            value: (batch,) scalar value estimate
        """
        x = torch.cat([encoder_out, time_step], dim=-1)
        return self.mlp(x).squeeze(-1)
