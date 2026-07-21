import torch
import torch.nn as nn


class AdvantageCalculator:
    """Computes the modified advantage A* for N-Player Proximal Advantage Alignment.

    Implements equation 68 from the paper (Section A.9) with the practical
    regularization from equation 59 (using 1/(1+t) instead of gamma^t):

        A*_t^i = A^i_t + beta * (sum_{k<t} A^i_k) / (1+t) * sum_{j!=i} A^j_t

    where A^i and A^j are the GAE advantages of agent i and all other agents j.
    """

    def __init__(self, agents, gamma=0.9, gae_lambda=0.95,
                 beta=3.0, aa_discount=0.9):
        self.agents = agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.beta = beta
        self.aa_discount = aa_discount

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: (T, B) rewards at each timestep
            values:  (T, B) critic value estimates
            dones:   (T,) or scalar, whether episode ended

        Returns:
            advantages: (T, B)
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=rewards.device)
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        return advantages

    def compute_aligned_advantages(self, all_rewards, all_values, dones,
                                   agent_idx):
        """Compute the modified advantage A* for N-Player Proximal Advantage Alignment.

        Implements equation 68 (Section A.9) with 1/(1+t) regularization:
            A*_t^i = A^i_t + beta * (sum_{k<t} A^i_k) / (1+t) * sum_{j!=i} A^j_t

        Args:
            all_rewards: list of N tensors, each (T, B) rewards per agent
            all_values:  list of N tensors, each (T, B) critic value estimates
            dones:       (T,) done flags
            agent_idx:   index i of the agent to compute A* for

        Returns:
            aligned_advantages: (T, B) the modified A* advantages for agent i
            all_agent_advantages: list of N tensors, each (T, B) raw GAE advantages
        """
        n_agents = len(all_rewards)
        all_agent_advantages = [
            self.compute_gae(all_rewards[j], all_values[j], dones)
            for j in range(n_agents)
        ]

        agent_adv = all_agent_advantages[agent_idx]
        T, B = agent_adv.shape
        aligned = torch.zeros_like(agent_adv)

        cumulative_agent_adv = torch.zeros(B, device=agent_adv.device)

        power_weights = {j: self.agents[j].power_multiplier for j in range(n_agents) if j != agent_idx}
        total_power = sum(power_weights.values())
        if total_power > 0:
            power_weights = {j: w / total_power for j, w in power_weights.items()}

        for t in range(T):
            others_sum = sum(
                power_weights[j] * all_agent_advantages[j][t]
                for j in power_weights
            )
            alignment_term = (
                self.beta
                * cumulative_agent_adv / (1.0 + t)
                * others_sum
            )

            aligned[t] = agent_adv[t] + alignment_term
            cumulative_agent_adv = cumulative_agent_adv + agent_adv[t]

        return aligned, all_agent_advantages
