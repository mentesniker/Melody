import torch
import torch.nn as nn
from agent import NegotiationAgent
from orchestrator import Orchestrator

def collect_trajectory(env, agents, device="cpu"):
    """Run N agent policies for T timesteps and collect trajectory.

    Returns dict with all tensors shaped (T, B, ...) needed by the optimizer,
    keyed by agent index.
    """
    obs_list = env.reset()
    obs_list = [o.to(device) for o in obs_list]

    n_agents = len(agents)
    B = env.batch_size
    T = env.trajectory_length
    has_items = hasattr(env, "orchestrator")

    hx = [agents[i].init_hidden(B).to(device) for i in range(n_agents)]

    all_obs = [[] for _ in range(n_agents)]
    all_actions = [[] for _ in range(n_agents)]
    all_raw = [[] for _ in range(n_agents)]
    all_logp = [[] for _ in range(n_agents)]
    all_rewards = [[] for _ in range(n_agents)]
    all_enc = [[] for _ in range(n_agents)]
    all_timesteps = []

    for t in range(T):
        for i in range(n_agents):
            all_obs[i].append(obs_list[i])

        enc = []
        actions = []
        for i in range(n_agents):
            with torch.no_grad():
                action, logp, raw, new_hx = agents[i].act(obs_list[i], hx[i])
            all_enc[i].append(new_hx)
            enc.append(new_hx)
            all_actions[i].append(action)
            all_raw[i].append(raw)
            all_logp[i].append(logp)
            actions.append(action)

        obs_new, rewards, done = env.step(actions)
        obs_new = [o.to(device) for o in obs_new]

        if has_items:
            clamped = [a.clamp(0.0, env.max_proposal) * env.orchestrator.availability for a in actions]
            env.orchestrator.resolve_bids(agents, clamped)

        for i in range(n_agents):
            all_rewards[i].append(rewards[i])

        all_timesteps.append(
            torch.full((B, 1), t / T, device=device)
        )

        for i in range(n_agents):
            hx[i] = enc[i]
        obs_list = obs_new

    result = {"timesteps": torch.stack(all_timesteps)}
    for i in range(n_agents):
        result[f"enc{i}"] = torch.stack(all_enc[i])
        result[f"raw{i}"] = torch.stack(all_raw[i])
        result[f"logp{i}"] = torch.stack(all_logp[i])
        result[f"r{i}"] = torch.stack(all_rewards[i])
        result[f"obs{i}"] = torch.stack(all_obs[i])

    return result



class PAAOptimizer:
    """Optimizer for Proximal Advantage Alignment (Algorithm 2).

    Handles both critic (L_C) and actor (L_a) loss computation and optimization
    for a NegotiationAgent. Each agent owns encoder+actor parameters (theta) and
    references an external critic with its own parameters (phi).
    """

    def __init__(self, agent, lr_actor=1e-3, lr_critic=1e-3,
                 clip_eps=0.2, entropy_beta=0.005, max_grad_norm=1.0,
                 shared_critic=False):
        self.agent = agent
        self.clip_eps = clip_eps
        self.entropy_beta = entropy_beta
        self.max_grad_norm = max_grad_norm
        self.shared_critic = shared_critic

        self.actor_optimizer = torch.optim.Adam(
            list(agent.encoder.parameters()) + list(agent.actor.parameters()),
            lr=lr_actor,
        )
        self.critic_optimizer = torch.optim.Adam(
            agent.critic.parameters(),
            lr=lr_critic,
        )

    def compute_critic_loss(self, encoder_outputs, time_steps, rewards,
                            dones, gamma=0.9):
        """Compute critic loss L_C using TD error (Algorithm 2, step 2-3).

        L_C = E[(r_t + gamma * V(s_{t+1}) - V(s_t))^2]

        Args:
            encoder_outputs: (T, B, H) encoder hidden states
            time_steps:      (T, B, 1) step indices
            rewards:         (T, B) per-step rewards
            dones:           (T,) done flags
            gamma:           discount factor

        Returns:
            critic_loss: scalar
        """
        T, B = rewards.shape
        values = torch.stack([
            self.agent.critic(encoder_outputs[t], time_steps[t])
            for t in range(T)
        ])

        targets = torch.zeros_like(rewards)
        for t in range(T):
            if t == T - 1:
                next_val = torch.zeros(B, device=rewards.device)
            else:
                next_val = values[t + 1].detach()
            targets[t] = rewards[t] + gamma * next_val

        critic_loss = ((values - targets.detach()) ** 2).mean()
        return critic_loss

    def compute_actor_loss(self, encoder_outputs, raw_actions, old_log_probs,
                           aligned_advantages):
        """Compute actor loss L_a using PPO clipped surrogate (Eq. 9).

        L_a = -E[min(r_n * A*, clip(r_n, 1-eps, 1+eps) * A*)]

        where r_n = pi_new / pi_old is the probability ratio and A* is the
        modified advantage from AdvantageCalculator.

        Args:
            encoder_outputs:     (T, B, H) encoder hidden states
            raw_actions:         (T, B, action_dim) pre-tanh samples stored during rollout
            old_log_probs:       (T, B) log probs under the policy that collected the data
            aligned_advantages:  (T, B) modified advantages A* from AdvantageCalculator

        Returns:
            actor_loss: scalar
        """
        T, B = old_log_probs.shape
        new_log_probs = []
        entropies = []

        for t in range(T):
            lp, ent = self.agent.actor.evaluate(encoder_outputs[t], raw_actions[t])
            new_log_probs.append(lp)
            entropies.append(ent)

        new_log_probs = torch.stack(new_log_probs)
        entropies = torch.stack(entropies)

        ratio = torch.exp(new_log_probs - old_log_probs.detach())

        advantages = aligned_advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy_bonus = -self.entropy_beta * entropies.mean()

        return policy_loss + entropy_bonus

    def optimize_critic(self, encoder_outputs, time_steps, rewards, dones,
                        gamma=0.9):
        """Run one optimization step on the critic (L_C w.r.t. phi).

        Args:
            encoder_outputs: (T, B, H) encoder hidden states (detached)
            time_steps:      (T, B, 1) step indices
            rewards:         (T, B)
            dones:           (T,)
            gamma:           discount factor

        Returns:
            critic_loss: scalar loss value for logging
        """
        loss = self.compute_critic_loss(
            encoder_outputs.detach(), time_steps, rewards, dones, gamma
        )
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()
        return loss.item()

    def optimize_shared_critic(self, all_encoder_outputs, time_steps,
                               all_rewards, dones, gamma=0.9):
        """Optimize a shared critic using the sum of TD losses from all agents.

        Args:
            all_encoder_outputs: list of N tensors, each (T, B, H)
            time_steps:          (T, B, 1)
            all_rewards:         list of N tensors, each (T, B)
            dones:               (T,)
            gamma:               discount factor

        Returns:
            mean critic loss (scalar) for logging
        """
        total_loss = sum(
            self.compute_critic_loss(
                enc.detach(), time_steps, rew, dones, gamma
            )
            for enc, rew in zip(all_encoder_outputs, all_rewards)
        )
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()
        return total_loss.item() / len(all_rewards)

    def optimize_actor(self, encoder_outputs, raw_actions, old_log_probs,
                       aligned_advantages):
        """Run one optimization step on the actor (L_a w.r.t. theta).

        Args:
            encoder_outputs:    (T, B, H) encoder hidden states
            raw_actions:        (T, B, action_dim) pre-tanh samples
            old_log_probs:      (T, B) log probs from collection policy
            aligned_advantages: (T, B) modified advantages A*

        Returns:
            actor_loss: scalar loss value for logging
        """
        loss = self.compute_actor_loss(
            encoder_outputs, raw_actions, old_log_probs, aligned_advantages
        )
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.encoder.parameters())
            + list(self.agent.actor.parameters()),
            self.max_grad_norm,
        )
        self.actor_optimizer.step()
        return loss.item()

    def optimize(self, agent, encoder_outputs, time_steps, raw_actions,
                 old_log_probs, rewards, dones, aligned_advantages,
                 gamma=0.9):
        """Full optimization step for one agent (Algorithm 2, lines 4-9).

        Runs critic update followed by actor update for the given
        NegotiationAgent.

        Args:
            agent:               NegotiationAgent to optimize
            encoder_outputs:     (T, B, H) encoder hidden states
            time_steps:          (T, B, 1) step indices
            raw_actions:         (T, B, action_dim) pre-tanh samples
            old_log_probs:       (T, B) log probs from collection policy
            rewards:             (T, B) per-step rewards for this agent
            dones:               (T,) done flags
            aligned_advantages:  (T, B) modified advantages A*
            gamma:               discount factor

        Returns:
            dict with 'critic_loss' and 'actor_loss' for logging
        """
        critic_loss = self.optimize_critic(
            encoder_outputs, time_steps, rewards, dones, gamma
        )
        actor_loss = self.optimize_actor(
            encoder_outputs, raw_actions, old_log_probs, aligned_advantages
        )
        return {"critic_loss": critic_loss, "actor_loss": actor_loss}
