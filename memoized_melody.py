import copy
import random
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from dynamic_negotiation import DynamicPAATrainer, _concat_trajectories
from optimizer import collect_trajectory
from benchmark import _gini
from lyapunov import LyapunovConstraint


class TrajectoryScorer:
    """Scores trajectory candidates for best-of-N selection."""

    def __init__(self, alpha_queue=1.0, alpha_penalty=1.0, alpha_gini=1.0):
        self.alpha_queue = alpha_queue
        self.alpha_penalty = alpha_penalty
        self.alpha_gini = alpha_gini

    def score(self, traj, agents, orchestrator, n_agents):
        unique_agents = list({id(a): a for a in agents}.values())
        total_queue = sum(len(a.won_items) for a in unique_agents)

        total_penalty = sum(
            orchestrator.pending_penalties[i].sum().item()
            for i in range(n_agents)
        )

        per_agent_rewards = [traj[f"r{i}"].sum().item() for i in range(n_agents)]
        lowest = min(per_agent_rewards)
        if lowest < 0:
            per_agent_rewards = [r - lowest for r in per_agent_rewards]
        capacities = [agents[i].power_multiplier for i in range(n_agents)]
        gini = _gini(per_agent_rewards, capacities=capacities)

        return -(
            self.alpha_queue * total_queue
            + self.alpha_penalty * total_penalty
            + self.alpha_gini * gini
        )


def _snapshot_state(env, orchestrator, agents):
    """Capture full mutable state of env/orchestrator/agents for later restore."""
    snap = {}

    snap["item_grid"] = copy.deepcopy(orchestrator.item_grid)
    snap["availability"] = orchestrator.availability.clone()
    snap["overflow_items"] = copy.deepcopy(orchestrator.overflow_items)
    snap["pending_penalties"] = [p.clone() for p in orchestrator.pending_penalties]
    snap["prev_proposals"] = [p.clone() for p in orchestrator.prev_proposals]
    snap["total_expired"] = orchestrator.total_expired
    snap["total_arrived"] = orchestrator.total_arrived
    snap["total_overflowed"] = orchestrator.total_overflowed
    snap["total_promoted"] = orchestrator.total_promoted

    snap["current_step"] = env.current_step

    unique_agents = {id(a): a for a in agents}
    snap["agent_states"] = {}
    for aid, agent in unique_agents.items():
        snap["agent_states"][aid] = {
            "won_items": copy.deepcopy(agent.won_items),
            "total_assigned": agent.total_assigned,
            "total_expired": agent.total_expired,
            "total_consumed": agent.total_consumed,
            "pending_reward": agent.pending_reward,
        }

    snap["torch_rng"] = torch.get_rng_state()
    snap["numpy_rng"] = np.random.get_state()
    snap["env_rng"] = copy.deepcopy(env._rng.bit_generator.state)
    snap["py_rng"] = random.getstate()

    return snap


def _restore_state(env, orchestrator, agents, snap):
    """Restore mutable state from a snapshot."""
    orchestrator.item_grid = copy.deepcopy(snap["item_grid"])
    orchestrator.availability = snap["availability"].clone()
    orchestrator.overflow_items = copy.deepcopy(snap["overflow_items"])
    orchestrator.pending_penalties = [p.clone() for p in snap["pending_penalties"]]
    orchestrator.prev_proposals = [p.clone() for p in snap["prev_proposals"]]
    orchestrator.total_expired = snap["total_expired"]
    orchestrator.total_arrived = snap["total_arrived"]
    orchestrator.total_overflowed = snap["total_overflowed"]
    orchestrator.total_promoted = snap["total_promoted"]

    env.current_step = snap["current_step"]

    unique_agents = {id(a): a for a in agents}
    for aid, agent in unique_agents.items():
        state = snap["agent_states"][aid]
        agent.won_items = copy.deepcopy(state["won_items"])
        agent.total_assigned = state["total_assigned"]
        agent.total_expired = state["total_expired"]
        agent.total_consumed = state["total_consumed"]
        agent.pending_reward = state["pending_reward"]

    torch.set_rng_state(snap["torch_rng"].clone())
    np.random.set_state(snap["numpy_rng"])
    env._rng.bit_generator.state = copy.deepcopy(snap["env_rng"])
    random.setstate(snap["py_rng"])


class MemoizedMelodyTrainer(DynamicPAATrainer):
    """PAA trainer with Lyapunov fairness constraint and best-of-N trajectory selection."""

    def __init__(
        self,
        n_candidates=4,
        lambda_lyap=0.1,
        lyap_c=0.5,
        alpha_queue=1.0,
        alpha_penalty=1.0,
        alpha_gini=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_candidates = n_candidates
        self.lyapunov = LyapunovConstraint(lambda_lyap, lyap_c)
        self.scorer = TrajectoryScorer(alpha_queue, alpha_penalty, alpha_gini)

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

            # --- Best-of-N trajectory selection ---
            snapshot = _snapshot_state(self.env, self.orchestrator, self.agents)

            best_traj = None
            best_score = -float("inf")
            best_state = None

            for k in range(self.n_candidates):
                _restore_state(self.env, self.orchestrator, self.agents, snapshot)
                torch.manual_seed(iteration * self.n_candidates + k)

                traj_k = collect_trajectory(
                    self.env, self.agents, device=self.device
                )

                sleep_seconds = 6
                self.orchestrator.tick(sleep_seconds, sleep_seconds)
                for agent in self.agents:
                    agent.consume(sleep_seconds)
                for i in range(self.n_agents):
                    traj_k[f"r{i}"] = traj_k[f"r{i}"] + self.agents[i].pending_reward
                    self.agents[i].pending_reward = 0

                score_k = self.scorer.score(
                    traj_k, self.agents, self.orchestrator, self.n_agents
                )

                if score_k > best_score:
                    best_score = score_k
                    best_traj = {
                        key: val.detach().clone() if isinstance(val, torch.Tensor) else val
                        for key, val in traj_k.items()
                    }
                    best_state = _snapshot_state(self.env, self.orchestrator, self.agents)

            _restore_state(self.env, self.orchestrator, self.agents, best_state)
            traj = best_traj

            raw_rewards = [traj[f"r{i}"] for i in range(self.n_agents)]

            with torch.no_grad():
                pm_tensor = torch.tensor(
                    [a.power_multiplier for a in self.agents],
                    dtype=torch.float32,
                    device=self.device,
                )
                lyap_penalty = self.lyapunov.compute_penalty(raw_rewards, power_multipliers=pm_tensor)
                for i in range(self.n_agents):
                    traj[f"r{i}"] = raw_rewards[i] - lyap_penalty[i]

            if self.curriculum:
                scenario_name = getattr(self.env, '_last_scenario_name', 'default')
                self.replay_buffer.store(traj, scenario_name)

            accumulated_trajs.append(traj)
            mean_rewards = [raw_rewards[i].mean().item() for i in range(self.n_agents)]

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
                replay_trajs = self.replay_buffer.sample(self.replay_samples)
                for replay_traj in replay_trajs:
                    self._replay_optimize(replay_traj)

            # --- Logging ---

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
                print(f"  [Melody] Iter {iteration:>5d}  |  Best={self.best_return:.4f}  |  Candidate score={best_score:.4f}")
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
