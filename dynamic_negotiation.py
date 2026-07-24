import gc
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

from dynamic_negotiation_env import DynamicNegotiationEnv
from negotiation_item import NegotiationItem, make_mec_item_classes
from critic import Critic
from agent import NegotiationAgent
from orchestrator import Orchestrator
from advantage import AdvantageCalculator
from optimizer import PAAOptimizer, collect_trajectory
from benchmark import Benchmark, BenchmarkScenario, CurriculumEnv

matplotlib.rcParams.update({"font.size": 16})
torch.set_num_threads(os.cpu_count())

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path="plots_dynamic/training_curves.png"):
    n_agents = 0
    while f"mean_r{n_agents}" in history:
        n_agents += 1

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    iters = np.arange(1, len(history["mean_r0"]) + 1)
    window = max(1, len(iters) // 20)

    def smooth(x, w):
        if w <= 1:
            return np.array(x)
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    ax = axes[0, 0]
    rewards = []
    for i in range(n_agents):
        ri = np.array(history[f"mean_r{i}"])
        rewards.append(ri)
        ax.plot(iters, smooth(ri, window), label=f"Agent {i}", alpha=0.9)
        ax.fill_between(iters, smooth(ri, window) - np.std(ri) * 0.3,
                         smooth(ri, window) + np.std(ri) * 0.3, alpha=0.10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Average Reward per Agent (Dynamic Items)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    combined = sum(rewards)
    label = " + ".join(f"R{i}" for i in range(n_agents))
    ax.plot(iters, smooth(combined, window), color="green", label=label)
    ax.fill_between(iters, smooth(combined, window) - np.std(combined) * 0.3,
                     smooth(combined, window) + np.std(combined) * 0.3,
                     alpha=0.15, color="green")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Combined Reward")
    ax.set_title(f"Social Welfare ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i in range(n_agents):
        ax.plot(iters, smooth(history[f"critic_loss{i}"], window), label=f"Critic {i}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Loss (TD Error)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for i in range(n_agents):
        ax.plot(iters, smooth(history[f"actor_loss{i}"], window), label=f"Actor {i}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Actor Loss (PPO + AA)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("PAA Training Curves — Dynamic Item Count", fontsize=20, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")



# ---------------------------------------------------------------------------
# Trajectory replay buffer
# ---------------------------------------------------------------------------

def _concat_trajectories(traj_list, n_agents):
    if len(traj_list) == 1:
        return traj_list[0]
    keys = ["timesteps"]
    for i in range(n_agents):
        keys.extend([f"enc{i}", f"raw{i}", f"logp{i}", f"r{i}", f"obs{i}"])
    return {k: torch.cat([t[k] for t in traj_list], dim=0) for k in keys}


class TrajectoryReplayBuffer:

    def __init__(self, capacity=200):
        self.capacity = capacity
        self.buffer = []
        self._idx = 0

    def store(self, traj, scenario_name="default"):
        entry = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in traj.items()
        }
        entry["_scenario"] = scenario_name
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self._idx % self.capacity] = entry
        self._idx += 1

    def sample(self, n=1, rng=None):
        if len(self.buffer) == 0:
            return []
        if rng is None:
            rng = np.random.default_rng()
        by_scenario = defaultdict(list)
        for i, entry in enumerate(self.buffer):
            by_scenario[entry["_scenario"]].append(i)
        picked = set()
        for indices in by_scenario.values():
            picked.add(indices[rng.integers(len(indices))])
        remaining = n - len(picked)
        if remaining > 0:
            pool = [i for i in range(len(self.buffer)) if i not in picked]
            if pool:
                extras = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                picked.update(extras.tolist())
        return [self.buffer[i] for i in picked]

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Dynamic trainer (wraps PAATrainer with dynamic env)
# ---------------------------------------------------------------------------

class DynamicPAATrainer:
    """Training loop for PAA on the dynamic-item-count environment.

    Uses the same PAA algorithm but with DynamicNegotiationEnv where the
    number of active items changes randomly each round.
    """

    def __init__(
        self,
        n_agents=2,
        max_items=100,
        min_items=1,
        max_new_items=None,
        max_proposal=5.0,
        trajectory_length=50,
        batch_size=1024,
        hidden_size=64,
        gamma=0.9,
        gae_lambda=0.95,
        beta=3.0,
        aa_discount=0.9,
        lr_actor=1e-3,
        lr_critic=1e-3,
        clip_eps=0.2,
        entropy_beta=0.005,
        max_grad_norm=1.0,
        shared_critic=False,
        self_play=False,
        aggregated_loss=True,
        device="cpu",
        env_class=None,
        item_classes=None,
        overflow_percentage=0.0,
        curriculum=False,
        replay_buffer_size=200,
        replay_ratio=0.25,
        replay_samples=1,
        lookahead_steps=1,
        min_value=0.0,
        user_delay_lambda=0.0,
        arrivals_poisson_lambda=None,
        agent_to_orch_delays=None,
        orch_to_agent_delays=None,
        class_probabilities=None,
        power_multipliers=None,
        lambda_a_low=None,
        lambda_a_med=None,
        lambda_a_high=None,
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.max_items = max_items
        self.min_value = min_value
        self.lookahead_steps = lookahead_steps
        self.user_delay_lambda = user_delay_lambda
        self.arrivals_poisson_lambda = arrivals_poisson_lambda
        self.class_probabilities = class_probabilities
        self.agent_to_orch_delays = agent_to_orch_delays if agent_to_orch_delays is not None else [0.0] * n_agents
        self.orch_to_agent_delays = orch_to_agent_delays if orch_to_agent_delays is not None else [0.0] * n_agents
        self.shared_critic = shared_critic or self_play
        self.self_play = self_play
        self.aggregated_loss = aggregated_loss
        if self_play and not aggregated_loss:
            raise ValueError(
                "aggregated_loss=False is incompatible with self_play=True: "
                "a single shared model cannot have individual gradient propagation"
            )
        self.hidden_size = hidden_size
        self.curriculum = curriculum
        self.replay_ratio = replay_ratio
        self.replay_samples = replay_samples
        self.replay_buffer = TrajectoryReplayBuffer(capacity=replay_buffer_size) if curriculum else None
        self._rng = np.random.default_rng()

        self.replay_rng = np.random.default_rng()

        self.orchestrator = Orchestrator(batch_size, max_items, n_agents, min_value=min_value,
                                          orch_to_agent_delays=self.orch_to_agent_delays)

        mec_env_kwargs = {}
        if self.arrivals_poisson_lambda is not None:
            mec_env_kwargs["user_delay_lambda"] = self.user_delay_lambda
            mec_env_kwargs["arrivals_poisson_lambda"] = self.arrivals_poisson_lambda
            mec_env_kwargs["agent_to_orch_delays"] = self.agent_to_orch_delays

        if curriculum:
            if self.arrivals_poisson_lambda is not None:
                low = BenchmarkScenario("Low Load", lambda_a=lambda_a_low or max(1, int(0.3 * max_items // n_agents)))
                medium = BenchmarkScenario("Medium Load", lambda_a=lambda_a_med or max(1, int(0.5 * max_items // n_agents)))
                high = BenchmarkScenario("High Load", lambda_a=lambda_a_high or max(1, int(max_items // n_agents)))
            else:
                low = BenchmarkScenario("Low Load", 1, max(1, int(0.3 * max_items)))
                medium = BenchmarkScenario("Medium Load", max(1, int(0.3 * max_items)), max(2, int(0.5 * max_items)))
                high = BenchmarkScenario("High Load", max(1, int(0.6 * max_items)), max_items)
            self._curriculum_scenarios = [low, medium, high]
            self.env = CurriculumEnv(
                all_scenarios=self._curriculum_scenarios,
                n_agents=n_agents,
                max_items=max_items,
                max_proposal=max_proposal,
                trajectory_length=trajectory_length,
                batch_size=batch_size,
                min_items=min_items,
                item_classes=item_classes,
                orchestrator=self.orchestrator,
                overflow_percentage=overflow_percentage,
                class_probabilities=class_probabilities,
                **mec_env_kwargs,
            )
        else:
            env_cls = env_class if env_class is not None else DynamicNegotiationEnv
            self.env = env_cls(
                n_agents=n_agents,
                max_items=max_items,
                max_proposal=max_proposal,
                trajectory_length=trajectory_length,
                batch_size=batch_size,
                min_items=min_items,
                item_classes=item_classes,
                orchestrator=self.orchestrator,
                overflow_percentage=overflow_percentage,
                class_probabilities=class_probabilities,
                **mec_env_kwargs,
            )

        obs_dim = self.env.obs_dim
        action_dim = self.env.action_dim

        if self_play:
            single_critic = Critic(hidden_size).to(device)
            single_agent = NegotiationAgent(
                obs_dim, action_dim, single_critic,
                hidden_size=hidden_size, max_proposal=max_proposal,
            ).to(device)
            self.critics = [single_critic] * n_agents
            self.agents = [single_agent] * n_agents
        else:
            if shared_critic:
                shared_critic_obj = Critic(hidden_size).to(device)
                self.critics = [shared_critic_obj] * n_agents
            else:
                self.critics = [Critic(hidden_size).to(device) for _ in range(n_agents)]

            if power_multipliers is None:
                power_multipliers = [(i+1)*250 for i in range(n_agents)]
            self.agents = [
                NegotiationAgent(
                    obs_dim, action_dim, self.critics[i],
                    hidden_size=hidden_size, max_proposal=max_proposal,
                    power_multiplier=power_multipliers[i],
                ).to(device)
                for i in range(n_agents)
            ]

        self.env.agents = self.agents

        self.adv_calc = AdvantageCalculator(
            self.agents,
            gamma=gamma, gae_lambda=gae_lambda,
            beta=beta, aa_discount=aa_discount,
        )

        if self_play:
            single_opt = PAAOptimizer(
                single_agent, lr_actor=lr_actor, lr_critic=lr_critic,
                clip_eps=clip_eps, entropy_beta=entropy_beta,
                max_grad_norm=max_grad_norm,
                shared_critic=True,
            )
            self.optimizers = [single_opt] * n_agents
        else:
            self.optimizers = [
                PAAOptimizer(
                    self.agents[i], lr_actor=lr_actor, lr_critic=lr_critic,
                    clip_eps=clip_eps, entropy_beta=entropy_beta,
                    max_grad_norm=max_grad_norm,
                    shared_critic=shared_critic,
                )
                for i in range(n_agents)
            ]

            if shared_critic:
                shared_critic_opt = torch.optim.Adam(
                    shared_critic_obj.parameters(), lr=lr_critic
                )
                for opt in self.optimizers:
                    opt.critic_optimizer = shared_critic_opt

        self.best_return = -float("inf")
        self.train_history = {}
        for i in range(n_agents):
            self.train_history[f"mean_r{i}"] = []
            self.train_history[f"critic_loss{i}"] = []
            self.train_history[f"actor_loss{i}"] = []

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)
        self.replay_rng = np.random.default_rng(seed)
        self.env._rng = np.random.default_rng(seed)

    def train(self, n_iterations=1000, log_interval=10):
        prev_assigned = [0] * self.n_agents
        prev_consumed = [0] * self.n_agents
        prev_expired = [0] * self.n_agents
        prev_orch_arrived = 0
        prev_orch_overflowed = 0
        prev_orch_expired = 0
        prev_orch_promoted = 0

        accumulated_trajs = []
        last_critic_losses = [0.0] * self.n_agents
        last_actor_losses = [0.0] * self.n_agents

        for iteration in range(1, n_iterations + 1):
            if self.curriculum:
                progress = iteration / n_iterations
                if progress < 0.33:
                    phase = 0
                elif progress < 0.66:
                    phase = 1
                else:
                    phase = 2
                self.env.set_active_scenarios([self._curriculum_scenarios[phase]])

            traj = collect_trajectory(
                self.env, self.agents, device=self.device
            )

            sleep_seconds = 6
            self.orchestrator.tick(sleep_seconds, sleep_seconds)
            for agent in self.agents:
                agent.consume(sleep_seconds)
            for i in range(self.n_agents):
                traj[f"r{i}"] = traj[f"r{i}"] + self.agents[i].pending_reward
                self.agents[i].pending_reward = 0

            if self.curriculum:
                scenario_name = getattr(self.env, '_last_scenario_name', 'default')
                self.replay_buffer.store(traj, scenario_name)

            accumulated_trajs.append(traj)
            mean_rewards = [traj[f"r{i}"].mean().item() for i in range(self.n_agents)]

            is_lookahead_boundary = (iteration % self.lookahead_steps == 0) or (iteration == n_iterations)

            if is_lookahead_boundary:
                merged = _concat_trajectories(accumulated_trajs, self.n_agents)
                accumulated_trajs = []

                timesteps = merged["timesteps"]
                T = timesteps.shape[0]
                dones = torch.zeros(T)

                enc = [merged[f"enc{i}"] for i in range(self.n_agents)]
                rewards = [merged[f"r{i}"] for i in range(self.n_agents)]

                values = []
                critic_losses = []
                if self.shared_critic:
                    for i in range(self.n_agents):
                        with torch.no_grad():
                            v = torch.stack([
                                self.critics[i](enc[i][t], timesteps[t]) for t in range(T)
                            ])
                        values.append(v)
                    closs = self.optimizers[0].optimize_shared_critic(
                        [enc[i] for i in range(self.n_agents)],
                        timesteps, rewards, dones, self.gamma,
                    )
                    critic_losses = [closs] * self.n_agents
                else:
                    for i in range(self.n_agents):
                        with torch.no_grad():
                            v = torch.stack([
                                self.critics[i](enc[i][t], timesteps[t]) for t in range(T)
                            ])
                        values.append(v)
                        closs = self.optimizers[i].optimize_critic(
                            enc[i], timesteps, rewards[i], dones, self.gamma
                        )
                        critic_losses.append(closs)

                aligned_advs = []
                with torch.no_grad():
                    for i in range(self.n_agents):
                        aligned, _ = self.adv_calc.compute_aligned_advantages(
                            rewards, values, dones, agent_idx=i
                        )
                        aligned_advs.append(aligned)

                actor_losses = []
                if self.aggregated_loss:
                    if self.self_play:
                        opt = self.optimizers[0]
                        total_loss = sum(
                            opt.compute_actor_loss(
                                enc[i], merged[f"raw{i}"], merged[f"logp{i}"], aligned_advs[i]
                            )
                            for i in range(self.n_agents)
                        ) / self.n_agents
                        opt.actor_optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(
                            list(opt.agent.encoder.parameters())
                            + list(opt.agent.actor.parameters()),
                            opt.max_grad_norm,
                        )
                        opt.actor_optimizer.step()
                        actor_losses = [total_loss.item()] * self.n_agents
                    else:
                        for i in range(self.n_agents):
                            aloss = self.optimizers[i].optimize_actor(
                                enc[i], merged[f"raw{i}"], merged[f"logp{i}"], aligned_advs[i]
                            )
                            actor_losses.append(aloss)
                else:
                    for i in range(self.n_agents):
                        powers = {j: self.agents[j].power_multiplier for j in range(self.n_agents) if j != i}
                        total_p = sum(powers.values())
                        other_advs = sum(
                            (powers[j] / total_p) * aligned_advs[j] for j in powers
                        )
                        aloss = self.optimizers[i].optimize_actor(
                            enc[i], merged[f"raw{i}"], merged[f"logp{i}"], other_advs
                        )
                        actor_losses.append(aloss)

                last_critic_losses = critic_losses
                last_actor_losses = actor_losses

            if self.curriculum and len(self.replay_buffer) > 10 and self._rng.random() < self.replay_ratio:
                replay_trajs = self.replay_buffer.sample(self.replay_samples, rng=self.replay_rng)
                for replay_traj in replay_trajs:
                    self._replay_optimize(replay_traj)

            for i in range(self.n_agents):
                self.train_history[f"mean_r{i}"].append(mean_rewards[i])
                self.train_history[f"critic_loss{i}"].append(last_critic_losses[i])
                self.train_history[f"actor_loss{i}"].append(last_actor_losses[i])

            combined_return = sum(mean_rewards)
            if combined_return > self.best_return:
                self.best_return = combined_return

            if iteration % log_interval == 0:
                sep = "-" * 60
                print(f"\n{sep}")
                print(f"  Iteration {iteration:>5d}  |  Best={self.best_return:.4f}")
                print(sep)

                delta_orch_arrived = self.orchestrator.total_arrived - prev_orch_arrived
                delta_orch_dispatched = sum(a.total_assigned for a in self.agents) - sum(prev_assigned)
                delta_orch_overflowed = self.orchestrator.total_overflowed - prev_orch_overflowed
                delta_orch_expired = self.orchestrator.total_expired - prev_orch_expired
                delta_orch_promoted = self.orchestrator.total_promoted - prev_orch_promoted

                header = f"  {'Agent':>5} | {'Reward':>12} | {'Power':>5} | {'Queue':>5} | {'Assigned':>8} | {'Consumed':>8} | {'Expired':>7} | Classes"
                print(header)
                print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+--------")
                for i, agent in enumerate(self.agents):
                    delta_assigned = agent.total_assigned - prev_assigned[i]
                    delta_consumed = agent.total_consumed - prev_consumed[i]
                    delta_expired = agent.total_expired - prev_expired[i]
                    avg_reward = sum(self.train_history[f"mean_r{i}"][-log_interval:]) / log_interval
                    class_counts = defaultdict(int)
                    for item in agent.won_items:
                        class_counts[int(item.values[0])] += 1
                    breakdown = ", ".join(f"c{k}:{v}" for k, v in sorted(class_counts.items()))
                    print(f"  {i:>5} | {avg_reward:>12.2e} | {agent.power_multiplier:>5} | {len(agent.won_items):>5} | {delta_assigned:>8} | {delta_consumed:>8} | {delta_expired:>7} | {breakdown}")
                    prev_assigned[i] = agent.total_assigned
                    prev_consumed[i] = agent.total_consumed
                    prev_expired[i] = agent.total_expired

                prev_orch_arrived = self.orchestrator.total_arrived
                prev_orch_overflowed = self.orchestrator.total_overflowed
                prev_orch_expired = self.orchestrator.total_expired
                prev_orch_promoted = self.orchestrator.total_promoted

                print(f"\n  Orchestrator:  arrived={delta_orch_arrived}  dispatched={delta_orch_dispatched}  overflowed={delta_orch_overflowed}  promoted={delta_orch_promoted}  expired={delta_orch_expired}")
                print(sep)

        print(f"Training complete. Best combined return: {self.best_return:.4f}")

    def _replay_optimize(self, replay_traj, baseline_logps=None):
        timesteps = replay_traj["timesteps"].to(self.device)
        T = timesteps.shape[0]
        B = timesteps.shape[1]
        dones = torch.zeros(T)

        fresh_enc = []
        for i in range(self.n_agents):
            obs_i = replay_traj[f"obs{i}"].to(self.device)
            hx = self.agents[i].init_hidden(B).to(self.device)
            enc_steps = []
            for t in range(T):
                hx = self.agents[i].encoder(obs_i[t], hx)
                enc_steps.append(hx)
            fresh_enc.append(torch.stack(enc_steps))

        rewards = [replay_traj[f"r{i}"].to(self.device) for i in range(self.n_agents)]

        values = []
        if self.shared_critic:
            for i in range(self.n_agents):
                with torch.no_grad():
                    v = torch.stack([
                        self.critics[i](fresh_enc[i][t], timesteps[t]) for t in range(T)
                    ])
                values.append(v)
            self.optimizers[0].optimize_shared_critic(
                [fresh_enc[i] for i in range(self.n_agents)],
                timesteps, rewards, dones, self.gamma,
            )
        else:
            for i in range(self.n_agents):
                with torch.no_grad():
                    v = torch.stack([
                        self.critics[i](fresh_enc[i][t], timesteps[t]) for t in range(T)
                    ])
                values.append(v)
                self.optimizers[i].optimize_critic(
                    fresh_enc[i], timesteps, rewards[i], dones, self.gamma
                )

        aligned_advs = []
        with torch.no_grad():
            for i in range(self.n_agents):
                aligned, _ = self.adv_calc.compute_aligned_advantages(
                    rewards, values, dones, agent_idx=i
                )
                aligned_advs.append(aligned)

        if self.aggregated_loss:
            if self.self_play:
                opt = self.optimizers[0]
                old_logps = [baseline_logps[i] if baseline_logps is not None
                             else replay_traj[f"logp{i}"].to(self.device)
                             for i in range(self.n_agents)]
                total_loss = sum(
                    opt.compute_actor_loss(
                        fresh_enc[i], replay_traj[f"raw{i}"].to(self.device),
                        old_logps[i], aligned_advs[i]
                    )
                    for i in range(self.n_agents)
                ) / self.n_agents
                opt.actor_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(opt.agent.encoder.parameters())
                    + list(opt.agent.actor.parameters()),
                    opt.max_grad_norm,
                )
                opt.actor_optimizer.step()
            else:
                for i in range(self.n_agents):
                    old_lp = baseline_logps[i] if baseline_logps is not None else replay_traj[f"logp{i}"].to(self.device)
                    self.optimizers[i].optimize_actor(
                        fresh_enc[i], replay_traj[f"raw{i}"].to(self.device),
                        old_lp, aligned_advs[i]
                    )
        else:
            for i in range(self.n_agents):
                old_lp = baseline_logps[i] if baseline_logps is not None else replay_traj[f"logp{i}"].to(self.device)
                powers = {j: self.agents[j].power_multiplier for j in range(self.n_agents) if j != i}
                total_p = sum(powers.values())
                other_advs = sum(
                    (powers[j] / total_p) * aligned_advs[j] for j in powers
                )
                self.optimizers[i].optimize_actor(
                    fresh_enc[i], replay_traj[f"raw{i}"].to(self.device),
                    old_lp, other_advs
                )


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class DynamicNegotiationVerifier:
    """End-to-end verification for the dynamic-item-count negotiation game.

    Trains PAA, PPO agents on environments where each
    round has a random number of active items. League evaluation also uses
    random item counts per episode.
    """

    def __init__(
        self,
        # --- Environment ---
        n_agents=2,              # Number of competing agents. More agents = harder coordination.
        max_items=100,           # Max item slots per round. Higher = more biddable items, easier for agents.
        min_items=1,             # Min item slots per round. Raises the floor on available items.
        max_proposal=5.0,        # Upper bound of continuous bid range [0, max_proposal].
        maximum_number_of_different_classes=10,  # Number of item types (value=1..N, ttl=N..1). More classes = richer strategy space.
        overflow_percentage=0.0, # Fraction of items routed to overflow queue instead of main grid. Higher = more pressure on TTL management.
        min_value=0.0,           # Minimum item value threshold for the orchestrator.
        # --- Model architecture ---
        hidden_size=64,          # GRU encoder + actor/critic hidden dim. Larger = more capacity but slower.
        # --- PPO / RL ---
        trajectory_length=50,    # Timesteps per trajectory rollout. With T=1, each trajectory is a single decision.
        batch_size=1024,         # Parallel environments. B=1 means fully sequential updates.
        gamma=0.9,               # Reward discount factor. Lower = more myopic agents.
        gae_lambda=0.95,         # GAE lambda for bias-variance tradeoff. Higher = lower bias, higher variance.
        beta=3.0,                # PAA advantage alignment strength. 0 = pure PPO; higher = more cooperative.
        melody_beta=None,        # Melody beta. If None, defaults to beta.
        aa_discount=0.9,         # Discount on cumulative advantage alignment term over time.
        lr_actor=1e-3,           # Actor learning rate. Too high = unstable; too low = slow convergence.
        lr_critic=1e-3,          # Critic learning rate. Usually same as actor or slightly higher.
        clip_eps=0.2,            # PPO clipping epsilon. Smaller = more conservative policy updates.
        entropy_beta=0.005,      # Entropy bonus coefficient. Higher = more exploration, less premature convergence.
        max_grad_norm=1.0,       # Gradient clipping norm. Prevents exploding gradients.
        shared_critic=False,     # All agents share one critic. Gives a global view of state value.
        self_play=False,         # All agent slots share one policy. Forces symmetric strategies.
        aggregated_loss=True,    # Sum actor losses across agents before backprop (must be True if self_play).
        # --- Curriculum / replay ---
        curriculum=False,        # Enable curriculum learning with progressive scenario difficulty.
        replay_buffer_size=200,  # Max stored trajectories for replay. Only used when curriculum=True.
        replay_ratio=0.25,       # Probability of replaying a past trajectory each iteration.
        replay_samples=1,        # Number of trajectories sampled per replay step.
        # --- Lookahead ---
        lookahead_steps=1,       # Collect N trajectories before optimizing. 1 = standard per-iteration updates.
        # --- Infrastructure ---
        device=None,             # "cpu" or "cuda". Auto-detected if None.
        plot_dir="plots_dynamic",  # Directory for saving training curves and benchmark plots.
        training_seed=None,      # Random seed for training. None = random. Shared across all variants for fair comparison.
        benchmark_seed=123,      # Random seed for benchmark evaluation. Ensures identical item sequences.
        # --- Melody-specific ---
        # Best-of-N trajectory selection: sample N candidates, keep the one with the best composite score.
        n_candidates=8,          # Number of candidate trajectories per iteration. Higher = better selection but N× slower.
        # Lyapunov fairness constraint: penalizes reward inequality drift.
        # Violation = max(0, delta_Var - lyap_c * (1 - Gini)), charged only to above-mean agents.
        lambda_lyap=0.5,         # Lyapunov penalty multiplier. Higher = stronger fairness enforcement on rewards.
        lyap_c=0.1,              # Allowed drift budget scaling. Lower = tighter constraint (less inequality tolerated).
        # Trajectory scoring weights (used to pick the best candidate out of N):
        alpha_queue=1.0,         # Weight on total queue congestion. Higher = prefer candidates with shorter queues.
        alpha_penalty=1.0,       # Weight on orchestrator penalties (expired items). Higher = prefer fewer expirations.
        alpha_gini=5.0,          # Weight on Gini fairness in candidate scoring. Higher = prefer fairer reward distributions.
        # --- MEC delay model ---
        user_delay_lambda=0.0,   # Lambda for Poisson user-to-agent delay distribution.
        arrivals_poisson_lambda=None,  # Lambda for Poisson per-agent item arrivals. None = use legacy uniform draw.
        agent_to_orch_delays=None,  # Per-agent constant uplink delays [N]. None = all zeros.
        orch_to_agent_delays=None,  # Per-agent constant downlink delays [N]. None = all zeros.
        class_probabilities=None,  # Per-class arrival probabilities [K]. None = uniform.
        power_multipliers=None,  # Per-agent power multipliers [N]. None = [(i+1)*250 for each agent].
        lambda_a_low=None,       # Poisson lambda for Low Load curriculum scenario. None = auto from max_items.
        lambda_a_med=None,       # Poisson lambda for Medium Load curriculum scenario. None = auto from max_items.
        lambda_a_high=None,      # Poisson lambda for High Load curriculum scenario. None = auto from max_items.
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.plot_dir = plot_dir
        if training_seed is None:
            training_seed = random.randint(0, 2**31 - 1)
            print(f"No training seed provided — using random seed: {training_seed}")
        self.training_seed = training_seed
        self.benchmark_seed = benchmark_seed
        self.n_agents = n_agents
        self.max_items = max_items
        self.min_items = min_items
        self.max_proposal = max_proposal
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.beta = beta
        self.melody_beta = melody_beta if melody_beta is not None else beta
        self.aa_discount = aa_discount
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_eps = clip_eps
        self.entropy_beta = entropy_beta
        self.max_grad_norm = max_grad_norm
        self.shared_critic = shared_critic
        self.self_play = self_play
        self.aggregated_loss = aggregated_loss
        self.overflow_percentage = overflow_percentage
        self.curriculum = curriculum
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio
        self.replay_samples = replay_samples
        self.lookahead_steps = lookahead_steps
        self.min_value = min_value
        self.n_candidates = n_candidates
        self.lambda_lyap = lambda_lyap
        self.lyap_c = lyap_c
        self.alpha_queue = alpha_queue
        self.alpha_penalty = alpha_penalty
        self.alpha_gini = alpha_gini
        self.user_delay_lambda = user_delay_lambda
        self.arrivals_poisson_lambda = arrivals_poisson_lambda
        self.agent_to_orch_delays = agent_to_orch_delays
        self.class_probabilities = class_probabilities
        self.orch_to_agent_delays = orch_to_agent_delays
        self.power_multipliers = power_multipliers
        self.lambda_a_low = lambda_a_low
        self.lambda_a_med = lambda_a_med
        self.lambda_a_high = lambda_a_high

        self._item_classes = make_mec_item_classes(n_agents)

        self._set_seed(training_seed)
        self.paa_trainer = DynamicPAATrainer(
            n_agents=n_agents,
            max_items=max_items,
            min_items=min_items,
            max_proposal=max_proposal,
            trajectory_length=trajectory_length,
            batch_size=batch_size,
            hidden_size=hidden_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            beta=beta,
            aa_discount=aa_discount,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            clip_eps=clip_eps,
            entropy_beta=entropy_beta,
            max_grad_norm=max_grad_norm,
            shared_critic=shared_critic,
            self_play=self_play,
            aggregated_loss=aggregated_loss,
            device=device,
            item_classes=self._item_classes,
            overflow_percentage=overflow_percentage,
            curriculum=curriculum,
            replay_buffer_size=replay_buffer_size,
            replay_ratio=replay_ratio,
            replay_samples=replay_samples,
            lookahead_steps=lookahead_steps,
            min_value=min_value,
            user_delay_lambda=user_delay_lambda,
            arrivals_poisson_lambda=arrivals_poisson_lambda,
            agent_to_orch_delays=agent_to_orch_delays,
            orch_to_agent_delays=orch_to_agent_delays,
            class_probabilities=class_probabilities,
            power_multipliers=power_multipliers,
            lambda_a_low=lambda_a_low,
            lambda_a_med=lambda_a_med,
            lambda_a_high=lambda_a_high,
        )
        self.paa_trainer.seed(training_seed)

    def _set_seed(self, seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._seed_trainers(seed)

    def _seed_trainers(self, seed):
        if hasattr(self, 'paa_trainer'):
            self.paa_trainer.seed(seed)
        if hasattr(self, 'ppo_trainer'):
            self.ppo_trainer.seed(seed)
        if hasattr(self, 'melody_trainer'):
            self.melody_trainer.seed(seed)

    def _free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------

    def train_paa(self, n_iterations=500, log_interval=10):
        print("\n--- Training PAA agents (dynamic items) ---")
        self.paa_trainer.train(n_iterations=n_iterations, log_interval=log_interval)
        self._free_memory()

    def train_ppo(self, n_iterations=500, log_interval=50):
        print("\n--- Training PPO baseline (dynamic items, beta=0) ---")
        self.ppo_trainer = DynamicPAATrainer(
            n_agents=self.n_agents,
            max_items=self.max_items,
            min_items=self.min_items,
            max_proposal=self.max_proposal,
            trajectory_length=self.trajectory_length,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            beta=0.0,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            shared_critic=self.shared_critic,
            aggregated_loss=self.aggregated_loss,
            device=self.device,
            item_classes=self._item_classes,
            overflow_percentage=self.overflow_percentage,
            curriculum=False,
            min_value=self.min_value,
            user_delay_lambda=self.user_delay_lambda,
            arrivals_poisson_lambda=self.arrivals_poisson_lambda,
            agent_to_orch_delays=self.agent_to_orch_delays,
            orch_to_agent_delays=self.orch_to_agent_delays,
            class_probabilities=self.class_probabilities,
            power_multipliers=self.power_multipliers,
        )
        self.ppo_trainer.seed(self.training_seed)
        self.ppo_trainer.train(n_iterations=n_iterations, log_interval=log_interval)
        self._free_memory()
        print("PPO baseline training complete.")

    def train_melody(self, n_iterations=500, log_interval=50):
        from memoized_melody import MemoizedMelodyTrainer
        print(f"\n--- Training Melody (Lyapunov + Best-of-N, beta={self.melody_beta}) ---")
        self.melody_trainer = MemoizedMelodyTrainer(
            n_candidates=self.n_candidates,
            lambda_lyap=self.lambda_lyap,
            lyap_c=self.lyap_c,
            alpha_queue=self.alpha_queue,
            alpha_penalty=self.alpha_penalty,
            alpha_gini=self.alpha_gini,
            beta=self.melody_beta,
            n_agents=self.n_agents,
            max_items=self.max_items,
            min_items=self.min_items,
            max_proposal=self.max_proposal,
            trajectory_length=self.trajectory_length,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            aa_discount=self.aa_discount,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            clip_eps=self.clip_eps,
            entropy_beta=self.entropy_beta,
            max_grad_norm=self.max_grad_norm,
            shared_critic=self.shared_critic,
            aggregated_loss=self.aggregated_loss,
            device=self.device,
            item_classes=self._item_classes,
            overflow_percentage=self.overflow_percentage,
            min_value=self.min_value,
            curriculum=self.curriculum,
            replay_buffer_size=self.replay_buffer_size,
            replay_ratio=self.replay_ratio,
            replay_samples=self.replay_samples,
            lookahead_steps=self.lookahead_steps,
            user_delay_lambda=self.user_delay_lambda,
            arrivals_poisson_lambda=self.arrivals_poisson_lambda,
            agent_to_orch_delays=self.agent_to_orch_delays,
            orch_to_agent_delays=self.orch_to_agent_delays,
            class_probabilities=self.class_probabilities,
            power_multipliers=self.power_multipliers,
            lambda_a_low=self.lambda_a_low,
            lambda_a_med=self.lambda_a_med,
            lambda_a_high=self.lambda_a_high,
        )
        self.melody_trainer.seed(self.training_seed)
        self.melody_trainer.train(n_iterations=n_iterations, log_interval=log_interval)
        self._free_memory()
        print("Melody training complete.")

    # ------------------------------------------------------------------
    # League evaluation with random item counts
    # ------------------------------------------------------------------

    def run_benchmark(self, n_episodes=5, episode_length=100, benchmark_seed=None):
        benchmark_seed = benchmark_seed if benchmark_seed is not None else self.benchmark_seed
        print("\n--- Running benchmark evaluation ---")

        shared_kwargs = dict(
            n_agents=self.n_agents,
            max_proposal=self.max_proposal,
            item_classes=self._item_classes,
            overflow_percentage=self.overflow_percentage,
            batch_size=1,
            user_delay_lambda=self.user_delay_lambda,
            arrivals_poisson_lambda=self.arrivals_poisson_lambda,
            agent_to_orch_delays=self.agent_to_orch_delays,
            orch_to_agent_delays=self.orch_to_agent_delays,
            class_probabilities=self.class_probabilities,
        )

        for agent in self.paa_trainer.agents:
            agent.eval()
        if hasattr(self, "ppo_trainer"):
            for agent in self.ppo_trainer.agents:
                agent.eval()
        if hasattr(self, "melody_trainer"):
            for agent in self.melody_trainer.agents:
                agent.eval()

        benchmark_kwargs = dict(max_items=self.max_items, seed=benchmark_seed)
        if self.arrivals_poisson_lambda is not None:
            benchmark_kwargs["lambda_a_low"] = max(1, int(0.3 * self.max_items // self.n_agents))
            benchmark_kwargs["lambda_a_med"] = max(1, int(0.5 * self.max_items // self.n_agents))
            benchmark_kwargs["lambda_a_high"] = max(1, int(self.max_items // self.n_agents))
        benchmark = Benchmark(**benchmark_kwargs)

        print("\nRunning PAA benchmark...")
        paa_bench = benchmark.run(
            self.paa_trainer.agents, shared_kwargs,
            n_episodes=n_episodes, episode_length=episode_length,
            device=self.device,
        )

        ppo_bench = None
        if hasattr(self, "ppo_trainer"):
            print("\nRunning PPO benchmark...")
            ppo_bench = benchmark.run(
                self.ppo_trainer.agents, shared_kwargs,
                n_episodes=n_episodes, episode_length=episode_length,
                device=self.device,
            )

        melody_bench = None
        if hasattr(self, "melody_trainer"):
            print("\nRunning Melody benchmark...")
            melody_bench = benchmark.run(
                self.melody_trainer.agents, shared_kwargs,
                n_episodes=n_episodes, episode_length=episode_length,
                device=self.device,
            )

        if ppo_bench is not None:
            benchmark.compare(
                paa_bench, ppo_bench, n_agents=self.n_agents,
                melody_results=melody_bench,
                save_path=os.path.join(self.plot_dir, "benchmark.png"),
            )

        self.benchmark_results = (paa_bench, ppo_bench, melody_bench)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self):
        os.makedirs(self.plot_dir, exist_ok=True)

        plot_training_curves(
            self.paa_trainer.train_history,
            save_path=os.path.join(self.plot_dir, "training", "paa_training_curves.png"),
        )

        if hasattr(self, "ppo_trainer"):
            plot_training_curves(
                self.ppo_trainer.train_history,
                save_path=os.path.join(self.plot_dir, "training", "ppo_training_curves.png"),
            )

        if hasattr(self, "melody_trainer"):
            plot_training_curves(
                self.melody_trainer.train_history,
                save_path=os.path.join(self.plot_dir, "training", "melody_training_curves.png"),
            )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, n_iterations=500, ppo_iterations=500, melody_iterations=500,
            log_interval=10, n_eval_episodes=1, eval_episode_length=100,
            training_seed=None, benchmark_seed=None):
        training_seed = training_seed if training_seed is not None else self.training_seed
        benchmark_seed = benchmark_seed if benchmark_seed is not None else self.benchmark_seed

        print("=" * 60)
        print("  Dynamic Negotiation Game — PAA Verification Pipeline")
        print(f"  Items per round: random in [{self.min_items}, {self.max_items}]")
        print("=" * 60)

        print(f"\nDevice: {self.device}")
        print(f"N agents: {self.n_agents}")
        print(f"Max items: {self.max_items}")
        print(f"Min items: {self.min_items}")
        print(f"Training iterations: {n_iterations}")
        print(f"Batch size: {self.batch_size}")
        print(f"Beta (AA weight): {self.beta}")
        print(f"Gamma: {self.gamma}")
        print(f"Shared critic: {self.shared_critic}")
        print(f"Self-play: {self.self_play}")
        print(f"Training seed: {training_seed}")
        print(f"Benchmark seed: {benchmark_seed}")
        print()

        self._set_seed(training_seed)
        self.train_paa(n_iterations=n_iterations, log_interval=log_interval)
        self._set_seed(training_seed)
        self.train_melody(n_iterations=melody_iterations, log_interval=log_interval)
        self._set_seed(training_seed)
        self.train_ppo(n_iterations=ppo_iterations, log_interval=log_interval)

        self.run_benchmark(n_episodes=n_eval_episodes, episode_length=eval_episode_length, benchmark_seed=benchmark_seed)
        self.plot_all()

        self._free_memory()

        print("\n" + "=" * 60)
        print("  Dynamic negotiation pipeline complete.")
        print("  Plots saved to:", self.plot_dir)
        print("=" * 60)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    verifier = DynamicNegotiationVerifier(
        n_agents=5,
        maximum_number_of_different_classes=3,
        max_items=40,
        min_items=1,
        max_proposal=5.0,
        trajectory_length=1,
        batch_size=1,
        hidden_size=128,
        gamma=0.95,
        gae_lambda=0.95,
        beta=15,
        melody_beta=15,
        aa_discount=0.9,
        lr_actor=1e-3,
        lr_critic=1e-3,
        clip_eps=0.1,
        entropy_beta=0.01,
        max_grad_norm=1.0,
        shared_critic=True,
        self_play=False,
        device=device,
        plot_dir="plots_dynamic",
        overflow_percentage=0.0,
        aggregated_loss=True,
        curriculum=True,
        replay_buffer_size=50,
        replay_ratio=0.4,
        replay_samples=10,
        lookahead_steps=5,
        # --- Melody-specific ---
        # Best-of-N trajectory selection: sample N candidates, keep the one with the best composite score.
        n_candidates=8,          # Number of candidate trajectories per iteration. Higher = better selection but N× slower.
        # Lyapunov fairness constraint: penalizes reward inequality drift.
        # Violation = max(0, delta_Var - lyap_c * (1 - Gini)), charged only to above-mean agents.
        lambda_lyap=0.5,         # Lyapunov penalty multiplier. Higher = stronger fairness enforcement on rewards.
        lyap_c=0.1,              # Allowed drift budget scaling. Lower = tighter constraint (less inequality tolerated).
        # Trajectory scoring weights (used to pick the best candidate out of N):
        alpha_queue=20.0,         # Weight on total queue congestion. Higher = prefer candidates with shorter queues.
        alpha_penalty=30.0,       # Weight on orchestrator penalties (expired items). Higher = prefer fewer expirations.
        alpha_gini=2000.0,          # Weight on Gini fairness in candidate scoring. Higher = prefer fairer reward distributions.
        # --- Curriculum lambda overrides ---
        lambda_a_low=1,
        lambda_a_med=2,
        lambda_a_high=3,
    )

    verifier.run(
        n_iterations=3000,
        ppo_iterations=3000,
        melody_iterations=3000,
        log_interval=300,
        eval_episode_length=1000,
        training_seed=5,
        benchmark_seed=7,
    )
