from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributions import Normal


class GRUEncoder(nn.Module):
    """Encoder: 2 Linear layers with ReLU followed by a GRU unit."""

    def __init__(self, obs_dim, hidden_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, obs, hx):
        x = self.fc(obs)
        hx = self.gru(x, hx)
        return hx


class Actor(nn.Module):
    """Continuous actor for the negotiation game.

    Outputs a Normal distribution whose mean is tanh-squashed and scaled to
    (0, max_proposal).  A single learnable log_std parameter controls the
    standard deviation across all items.
    """

    def __init__(self, hidden_size, action_dim, max_proposal=5.0):
        super().__init__()
        self.max_proposal = max_proposal
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, encoder_out):
        """Return a (raw_action, log_prob, squashed_action) tuple.

        raw_action:      sample from the Normal *before* tanh squashing
        log_prob:        log-probability corrected for the tanh + affine transform
        squashed_action: the final proposal in (0, max_proposal)
        """
        mlp_out = self.mlp(encoder_out)
        if torch.isnan(mlp_out).all():
            print(f"[NaN DEBUG] mlp_out is ALL NaN | shape={mlp_out.shape}")
        if torch.isnan(encoder_out).all():
            print(f"[NaN DEBUG] encoder_out is ALL NaN | shape={encoder_out.shape}")

        mean = torch.tanh(mlp_out) * (self.max_proposal / 2.0)
        if torch.isnan(mean).all():
            print(f"[NaN DEBUG] mean is ALL NaN | mean={mean}")

        std = self.log_std.exp().expand_as(mean)
        if torch.isnan(std).all():
            print(f"[NaN DEBUG] std is ALL NaN | log_std={self.log_std}, std={std}")

        dist = Normal(mean, std)
        raw = dist.rsample()
        if torch.isnan(raw).all():
            print(f"[NaN DEBUG] raw is ALL NaN | raw={raw}, mean={mean}, std={std}")

        squashed = torch.tanh(raw)
        # shift from (-1, 1) to (0, max_proposal)
        action = (squashed + 1.0) * (self.max_proposal / 2.0)

        # log prob with tanh correction:  log p(a) - log(1 - tanh^2(raw)) - log(scale)
        log_prob = dist.log_prob(raw) - torch.log1p(-squashed.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(torch.tensor(self.max_proposal / 2.0))
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, raw

    def evaluate(self, encoder_out, raw_action):
        """Re-evaluate log_prob and entropy for a previously sampled raw_action."""
        mlp_out = self.mlp(encoder_out)
        if torch.isnan(mlp_out).all():
            print(f"[NaN DEBUG] mlp_out is ALL NaN | shape={mlp_out.shape}")
        if torch.isnan(encoder_out).all():
            print(f"[NaN DEBUG] encoder_out is ALL NaN | shape={encoder_out.shape}")

        mean = torch.tanh(mlp_out) * (self.max_proposal / 2.0)
        if torch.isnan(mean).all():
            print(f"[NaN DEBUG] mean is ALL NaN | mean={mean}")

        std = self.log_std.exp().expand_as(mean)
        if torch.isnan(std).all():
            print(f"[NaN DEBUG] std is ALL NaN | log_std={self.log_std}, std={std}")


        mean = torch.tanh(self.mlp(encoder_out)) * (self.max_proposal / 2.0)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        squashed = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action) - torch.log1p(-squashed.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(torch.tensor(self.max_proposal / 2.0))
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class NegotiationAgent(nn.Module):
    """PPO-style agent for the negotiation game (Algorithm 2).

    Owns an encoder and actor network, and holds a reference to an external
    critic that may be shared or swapped independently.
    """

    def __init__(self, obs_dim, action_dim, critic, hidden_size=64, max_proposal=5.0, power_multiplier=1):
        super().__init__()
        self.encoder = GRUEncoder(obs_dim, hidden_size)
        self.actor = Actor(hidden_size, action_dim, max_proposal)
        self.critic = critic
        self.hidden_size = hidden_size
        self.power_multiplier = power_multiplier
        self.won_items = []
        self.total_assigned = 0
        self.total_expired = 0
        self.total_consumed = 0
        self.pending_reward = 0.0
        self.benchmark_mode = False
        self.total_assigned_by_class = defaultdict(int)
        self.total_expired_by_class = defaultdict(int)

    def clear_won_items(self):
        self.won_items = []
        self.total_assigned = 0
        self.total_expired = 0
        self.total_consumed = 0
        self.pending_reward = 0.0
        self.total_assigned_by_class = defaultdict(int)
        self.total_expired_by_class = defaultdict(int)

    def consume(self, elapsed_seconds):
        for _ in range(elapsed_seconds):
            remaining_power = self.power_multiplier
            while remaining_power > 0 and self.won_items:
                head = self.won_items[0]
                if head.timestamp >= head.time_to_live:
                    self.won_items.pop(0)
                    self.total_expired += 1
                    if head.class_id is not None:
                        self.total_expired_by_class[head.class_id] += 1
                    if self.benchmark_mode:
                        self.pending_reward -= head.values[0]
                    else:
                        self.pending_reward -= head.values[0] * (1/self.power_multiplier)
                    continue
                if head.execution_time <= remaining_power:
                    remaining_power -= head.execution_time
                    head.execution_time = 0
                    self.won_items.pop(0)
                    self.total_consumed += 1
                    self.pending_reward += head.values[0]
                else:
                    head.execution_time -= remaining_power
                    remaining_power = 0

            self.tick_queue(1)

            surviving = []
            for item in self.won_items:
                if item.timestamp >= item.time_to_live:
                    self.total_expired += 1
                    if item.class_id is not None:
                        self.total_expired_by_class[item.class_id] += 1
                    if self.benchmark_mode:
                        self.pending_reward -= item.values[0]
                    else:
                        self.pending_reward -= item.values[0] * (1/self.power_multiplier)
                else:
                    surviving.append(item)
            self.won_items = surviving
            

    def tick_queue(self, elapsed_seconds):
        for item in self.won_items:
            item.timestamp += elapsed_seconds

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def act(self, obs, hx):
        """Select an action given an observation and GRU hidden state.

        Returns:
            action:  (batch, action_dim) proposal clamped to (0, max_proposal)
            log_prob: (batch,) log-probability of the action
            raw_action: (batch, action_dim) pre-tanh sample (needed for PPO re-evaluation)
            new_hx: updated GRU hidden state
        """
        hx = self.encoder(obs, hx)
        action, log_prob, _ = self.actor(hx)
        return action, log_prob, _, hx