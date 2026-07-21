"""FAIRMEC protocol implementation.

Deterministic, rule-based federation protocol where orchestrators bid on
pending requests based on capacity feasibility and delay. Lowest-delay bid
wins. No training required.

Based on: "FAIRMEC: Fair Resource Sharing in Federated Edge Computing"
"""

from collections import defaultdict

import numpy as np
import torch

from negotiation_item import NegotiationItem
from dynamic_negotiation_env import DynamicNegotiationEnv
from orchestrator import Orchestrator
from benchmark import _compute_metrics, BenchmarkScenario, ChaosEnv, Benchmark

FAIRMEC_DEBUG = False
_FAIRMEC_PRINT_INTERVAL = 100


def _is_debug_step(t, episode_length):
    return FAIRMEC_DEBUG and (t == 0 or t % _FAIRMEC_PRINT_INTERVAL == 0 or t == episode_length - 1)


class FairMecItem(NegotiationItem):

    def __init__(self, n_agents, value, ttl, execution_time=1, class_id=None, origin_agent=0):
        super().__init__(n_agents, value, ttl, execution_time=execution_time, class_id=class_id)
        self.origin_agent = origin_agent


class FairMecEnv(DynamicNegotiationEnv):

    def _generate_round(self):
        B = self.batch_size
        M = self.max_items
        K = self.maximum_number_of_different_classes

        self.orchestrator.availability = torch.zeros(B, M)

        for b in range(B):
            survivors = []
            for item in self.orchestrator.item_grid[b]:
                if item is None:
                    continue
                if item.timestamp >= item.time_to_live:
                    penalty = item.values[0] if self.orchestrator.benchmark_mode else item.time_to_live * item.values[0]
                    for a in range(self.n_agents):
                        self.orchestrator.pending_penalties[a][b] += penalty
                    self.orchestrator.total_expired += 1
                else:
                    survivors.append(item)

            if self.arrivals_poisson_lambda is not None:
                new_items = []
                for agent_idx in range(self.n_agents):
                    n_agent = self._rng.poisson(self.arrivals_poisson_lambda)
                    for _ in range(n_agent):
                        idx = self._rng.choice(K, p=self.class_probabilities)
                        item = FairMecItem(
                            self.n_agents,
                            value=float(self._item_classes[idx].values[0]),
                            ttl=self._item_classes[idx].time_to_live + self.ttl_lambda,
                            execution_time=self._item_classes[idx].execution_time,
                            class_id=int(idx),
                            origin_agent=agent_idx,
                        )
                        item.timestamp = float(self._rng.poisson(self.user_delay_lambda)) + self.agent_to_orch_delays[agent_idx]
                        new_items.append(item)
            else:
                n_new = self._rng.integers(self.min_items, self.max_new_items + 1)
                new_items = []
                for _ in range(int(n_new)):
                    idx = self._rng.choice(K, p=self.class_probabilities)
                    origin = int(self._rng.integers(0, self.n_agents))
                    item = FairMecItem(
                        self.n_agents,
                        value=float(self._item_classes[idx].values[0]),
                        ttl=self._item_classes[idx].time_to_live + self.ttl_lambda,
                        execution_time=self._item_classes[idx].execution_time,
                        class_id=int(idx),
                        origin_agent=origin,
                    )
                    new_items.append(item)
            self.orchestrator.total_arrived += len(new_items)

            on_grid = list(survivors)

            old_overflow = self.orchestrator.overflow_items[b]
            slots_left = M - len(on_grid)
            promoted = old_overflow[:slots_left]
            on_grid += promoted
            self.orchestrator.total_promoted += len(promoted)
            remaining_overflow = old_overflow[slots_left:]

            slots_left = M - len(on_grid)
            on_grid += new_items[:slots_left]
            leftover_new = new_items[slots_left:]

            new_overflow = remaining_overflow + leftover_new
            self.orchestrator.overflow_items[b] = new_overflow
            self.orchestrator.total_overflowed += len(new_overflow)

            self.orchestrator.item_grid[b] = [None] * M
            k = len(on_grid)
            self.orchestrator.availability[b, :k] = 1.0
            for slot_pos, item in enumerate(on_grid):
                self.orchestrator.item_grid[b][slot_pos] = item

        self.values = [self._agent_values(i) for i in range(self.n_agents)]


class FairMecChaosEnv(FairMecEnv):

    def __init__(self, scenarios, **kwargs):
        super().__init__(**kwargs)
        self._scenarios = scenarios

    def _generate_round(self):
        scenario = self._scenarios[self._rng.integers(0, len(self._scenarios))]
        if scenario.lambda_a is not None:
            self.arrivals_poisson_lambda = scenario.lambda_a
        else:
            self.min_items = scenario.min_items
            self.max_new_items = scenario.max_new_items
        super()._generate_round()


class FairMecAgent:

    def __init__(self, power_multiplier=1):
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

    def consume(self, elapsed_seconds, _debug=False, _agent_idx=None):
        pre_consumed = self.total_consumed
        pre_expired = self.total_expired
        pre_reward = self.pending_reward
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
                        self.pending_reward -= head.values[0] * (1 / self.power_multiplier)
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
                        self.pending_reward -= item.values[0] * (1 / self.power_multiplier)
                else:
                    surviving.append(item)
            self.won_items = surviving
        if _debug:
            d_consumed = self.total_consumed - pre_consumed
            d_expired = self.total_expired - pre_expired
            d_reward = self.pending_reward - pre_reward
            print(f"[FAIRMEC]   Agent {_agent_idx} consume: "
                  f"items_finished_processing={d_consumed}, "
                  f"items_expired_in_queue(exceeded_TTL)={d_expired}, "
                  f"reward_change(+consumed_value/-expiry_penalty)={d_reward:+.2f}, "
                  f"items_still_in_queue={len(self.won_items)}")

    def tick_queue(self, elapsed_seconds):
        for item in self.won_items:
            item.timestamp += elapsed_seconds


# ---------------------------------------------------------------------------
#  Algorithm 1: Local Bid Construction
# ---------------------------------------------------------------------------

def local_bid_construction(agent_idx, item_grid, availability, agents,
                           n_agents, inter_orch_delays, rng, _debug=False):
    agent = agents[agent_idx]
    queued_exec_time = sum(item.execution_time for item in agent.won_items)

    available_items = []
    M = len(item_grid)
    for m in range(M):
        if availability[m].item() <= 0:
            continue
        item = item_grid[m]
        if item is None:
            continue
        available_items.append((m, item))

    if not available_items:
        if _debug:
            print(f"[FAIRMEC]   Agent {agent_idx}: pending_work_in_queue(exec_time_sum)={queued_exec_time:.1f}, "
                  f"items_visible_on_grid=0, feasible_bids_produced=0")
        return []

    items_by_class = defaultdict(list)
    all_class_ids = set()
    for m, item in available_items:
        cid = item.class_id if item.class_id is not None else 0
        items_by_class[cid].append((m, item))
        all_class_ids.add(cid)

    sorted_class_ids = sorted(all_class_ids, reverse=True)
    min_class_id = sorted_class_ids[-1] if sorted_class_ids else 0

    bids = []
    for class_id in sorted_class_ids:
        candidates = items_by_class[class_id]
        indices = list(range(len(candidates)))
        rng.shuffle(indices)

        for idx in indices:
            m, item = candidates[idx]

            origin = getattr(item, 'origin_agent', 0)
            d_oh_os = inter_orch_delays[origin][agent_idx] if origin != agent_idx else 0.0

            queue_drain = queued_exec_time / agent.power_multiplier
            proc_time = item.execution_time / agent.power_multiplier
            total_delay = item.timestamp + d_oh_os + queue_drain + proc_time

            if class_id != min_class_id and total_delay >= item.time_to_live:
                if _debug:
                    print(f"[FAIRMEC]     SKIP grid_slot={m} class={class_id} | "
                          f"estimated_delay={total_delay:.2f}s >= TTL={item.time_to_live:.2f}s (non-lowest class, would miss deadline)")
                continue

            if total_delay >= item.time_to_live:
                if _debug:
                    print(f"[FAIRMEC]     SKIP grid_slot={m} class={class_id} | "
                          f"estimated_delay={total_delay:.2f}s >= TTL={item.time_to_live:.2f}s (infeasible, would miss deadline)")
                continue

            bids.append((m, agent_idx, total_delay))
            queued_exec_time += item.execution_time

            if _debug:
                print(f"[FAIRMEC]     BID  grid_slot={m} class={class_id} | "
                      f"item_age={item.timestamp:.2f}s + inter_orch_transfer={d_oh_os:.2f}s + "
                      f"queue_drain_time={queue_drain:.2f}s + processing_time={proc_time:.2f}s "
                      f"= estimated_delay={total_delay:.2f}s (TTL={item.time_to_live:.2f}s, "
                      f"value={item.values[0]:.1f})")

    if _debug:
        print(f"[FAIRMEC]   Agent {agent_idx} summary: pending_work_in_queue(exec_time_sum)={sum(it.execution_time for it in agent.won_items):.1f}, "
              f"items_visible_on_grid={len(available_items)}, feasible_bids_produced={len(bids)}")

    return bids


# ---------------------------------------------------------------------------
#  Algorithm 2: Bid Exchange & Atomic Selection
# ---------------------------------------------------------------------------

def fair_mec_select(all_bids, n_agents, max_items, max_proposal, batch_size=1, _debug=False):
    bids_by_item = defaultdict(list)
    for agent_bids in all_bids:
        for (item_idx, agent_idx, delay) in agent_bids:
            bids_by_item[item_idx].append((delay, agent_idx))

    proposals = [torch.zeros(batch_size, max_items) for _ in range(n_agents)]

    wins_per_agent = defaultdict(int)
    for item_idx, item_bids in bids_by_item.items():
        winner_delay, winner_agent = min(item_bids, key=lambda b: (b[0], b[1]))
        proposals[winner_agent][0, item_idx] = max_proposal
        wins_per_agent[winner_agent] += 1
        if _debug:
            losers_str = ", ".join(
                f"agent{a}={d:.2f}s" for d, a in sorted(item_bids, key=lambda b: (b[0], b[1])) if a != winner_agent
            )
            losers_part = f", losing_bids=[{losers_str}]" if losers_str else ", no_other_bids"
            print(f"[FAIRMEC]     grid_slot_{item_idx} → WINNER agent{winner_agent} "
                  f"(delay={winner_delay:.2f}s){losers_part}")

    if _debug:
        wins_str = ", ".join(f"agent{a}: {c} items_won" for a, c in sorted(wins_per_agent.items()))
        print(f"[FAIRMEC]   total_items_with_bids(contested)={len(bids_by_item)}, "
              f"wins_per_agent=[{wins_str}]")

    return proposals


# ---------------------------------------------------------------------------
#  Episode Runner
# ---------------------------------------------------------------------------

def run_fairmec_episode(agents, env, orchestrator, n_agents, max_proposal,
                        episode_length, inter_orch_delays):
    for agent in agents:
        agent.clear_won_items()
        agent.benchmark_mode = True
    orchestrator.benchmark_mode = True

    obs_list = env.reset()
    B = env.batch_size
    rng = np.random.default_rng()

    cumulative_rewards = [0.0] * n_agents
    queue_length_sum = [0.0] * n_agents

    if FAIRMEC_DEBUG:
        power_str = ", ".join(f"agent{i}: processing_speed(power_multiplier)={agents[i].power_multiplier}" for i in range(n_agents))
        print(f"[FAIRMEC] === Episode start ===")
        print(f"[FAIRMEC]  num_agents={n_agents}, episode_length(total_rounds)={episode_length}, "
              f"batch_size(parallel_envs)={B}, max_items(grid_capacity)={env.max_items}")
        print(f"[FAIRMEC]  [{power_str}]")

    for t in range(episode_length):
        dbg = _is_debug_step(t, episode_length)

        if dbg:
            n_avail = int(orchestrator.availability.sum().item())
            n_overflow = sum(len(orchestrator.overflow_items[b]) for b in range(B))
            print(f"[FAIRMEC] --- Round {t}/{episode_length} | "
                  f"items_on_grid(available_to_bid)={n_avail}, "
                  f"items_in_overflow_queue(waiting_for_grid_space)={n_overflow} ---")

        # Algorithm 1: each agent builds its bid set
        if dbg:
            print(f"[FAIRMEC]  Algorithm 1: Local Bid Construction — each agent evaluates items and bids on those it can finish before TTL")
        all_bids = []
        for b in range(B):
            batch_bids = []
            for i in range(n_agents):
                agent_bids = local_bid_construction(
                    i, orchestrator.item_grid[b], orchestrator.availability[b],
                    agents, n_agents, inter_orch_delays, rng, _debug=dbg,
                )
                batch_bids.append(agent_bids)
            all_bids.append(batch_bids)

        # Algorithm 2: select winners, produce synthetic proposals
        if dbg:
            print(f"[FAIRMEC]  Algorithm 2: Bid Exchange & Selection — lowest estimated_completion_delay wins each item")
        proposals = [torch.zeros(B, env.max_items) for _ in range(n_agents)]
        for b in range(B):
            batch_proposals = fair_mec_select(
                all_bids[b], n_agents, env.max_items, max_proposal, batch_size=1,
                _debug=dbg,
            )
            for i in range(n_agents):
                proposals[i][b] = batch_proposals[i][0]

        # Resolve bids before stepping so availability matches the grid agents bid on
        if dbg:
            print(f"[FAIRMEC]  Resolve bids — highest-bid agent wins each item, item moves to winner's processing queue")
        clamped = [p.clamp(0.0, max_proposal) * orchestrator.availability for p in proposals]
        orchestrator.resolve_bids(agents, clamped)

        if dbg:
            assigned_str = ", ".join(f"agent{i}: {agents[i].total_assigned} total_items_assigned_so_far" for i in range(n_agents))
            print(f"[FAIRMEC]   After resolve: [{assigned_str}]")

        # Step env to advance round (generates new items, computes rewards)
        obs_list, rewards, done = env.step(proposals)

        if dbg:
            print(f"[FAIRMEC]  Tick(6s) & Consume(6s) — advance clocks on grid/overflow items, agents process their queues")
        orchestrator.tick(6, 6)
        for i, agent in enumerate(agents):
            agent.consume(6, _debug=dbg, _agent_idx=i)

        for i in range(n_agents):
            cumulative_rewards[i] += rewards[i].item() + agents[i].pending_reward
            agents[i].pending_reward = 0
            queue_length_sum[i] += len(agents[i].won_items)

        if dbg:
            rew_str = ", ".join(f"agent{i}={cumulative_rewards[i]:.2f}" for i in range(n_agents))
            q_str = ", ".join(f"agent{i}={len(agents[i].won_items)}" for i in range(n_agents))
            print(f"[FAIRMEC]  cumulative_reward(total_value_earned_so_far): [{rew_str}]")
            print(f"[FAIRMEC]  current_queue_length(items_waiting_to_be_processed): [{q_str}]")

    if FAIRMEC_DEBUG:
        print(f"[FAIRMEC] === Episode complete — Final Summary ===")
        for i in range(n_agents):
            print(f"[FAIRMEC]  Agent {i} (processing_speed={agents[i].power_multiplier}): "
                  f"total_reward={cumulative_rewards[i]:.2f}, "
                  f"items_won(assigned_by_bidding)={agents[i].total_assigned}, "
                  f"items_successfully_processed(consumed)={agents[i].total_consumed}, "
                  f"items_expired_before_processing={agents[i].total_expired}, "
                  f"items_left_in_queue={len(agents[i].won_items)}")
        print(f"[FAIRMEC]  Orchestrator: total_items_generated(arrived)={orchestrator.total_arrived}, "
              f"total_items_expired_on_grid={orchestrator.total_expired}")

    avg_queue = [q / max(1, episode_length) for q in queue_length_sum]
    return {
        "rewards": cumulative_rewards,
        "assigned": [agents[i].total_assigned for i in range(n_agents)],
        "consumed": [agents[i].total_consumed for i in range(n_agents)],
        "expired": [agents[i].total_expired for i in range(n_agents)],
        "assigned_by_class": [dict(agents[i].total_assigned_by_class) for i in range(n_agents)],
        "expired_by_class": [dict(agents[i].total_expired_by_class) for i in range(n_agents)],
        "queue_lengths": avg_queue,
        "orch_arrived": orchestrator.total_arrived,
        "orch_expired": orchestrator.total_expired,
    }


# ---------------------------------------------------------------------------
#  FAIRMEC Benchmark Runner
# ---------------------------------------------------------------------------

def run_fairmec_benchmark(config, benchmark, n_episodes=5, episode_length=1000):
    n_agents = config["n_agents"]
    max_proposal = config["max_proposal"]
    max_items = config["max_items"]
    from negotiation_item import make_mec_item_classes
    item_classes = make_mec_item_classes(n_agents)
    overflow_percentage = config.get("overflow_percentage", 0.0)
    inter_orch_delays = config.get("inter_orch_delays")
    if inter_orch_delays is None:
        default_delay = config.get("inter_orch_delay", 2.0)
        inter_orch_delays = [[0.0 if i == j else default_delay for j in range(n_agents)] for i in range(n_agents)]
    user_delay_lambda = config.get("user_delay_lambda", 0.0)
    arrivals_poisson_lambda = config.get("arrivals_poisson_lambda")
    agent_to_orch_delays = config.get("agent_to_orch_delays")
    orch_to_agent_delays = config.get("orch_to_agent_delays")
    class_probabilities = config.get("class_probabilities")

    mec_env_kwargs = {}
    if arrivals_poisson_lambda is not None:
        mec_env_kwargs["user_delay_lambda"] = user_delay_lambda
        mec_env_kwargs["arrivals_poisson_lambda"] = arrivals_poisson_lambda
        mec_env_kwargs["agent_to_orch_delays"] = agent_to_orch_delays

    results = {}
    for sc_idx, scenario in enumerate(benchmark.scenarios):
        if FAIRMEC_DEBUG:
            print(f"\n[FAIRMEC] ============================================")
            print(f"[FAIRMEC] Scenario: {scenario.name} | {n_episodes} episodes x {episode_length} rounds")
            if scenario.lambda_a is not None:
                print(f"[FAIRMEC]  arrival_model=Poisson, lambda(avg_new_items_per_agent_per_round)={scenario.lambda_a}")
            else:
                print(f"[FAIRMEC]  arrival_model=Uniform, min_new_items_per_round={scenario.min_items}, "
                      f"max_new_items_per_round={scenario.max_new_items}")
            print(f"[FAIRMEC]  n_agents={n_agents}, max_items(grid_capacity)={max_items}, "
                  f"max_proposal(highest_allowed_bid)={max_proposal}")
            print(f"[FAIRMEC] ============================================")
        print(f"  Running FAIRMEC scenario: {scenario.name} ({n_episodes} episodes)...")
        episode_data = {
            "rewards": [], "assigned": [], "consumed": [],
            "expired": [], "queue_lengths": [],
            "orch_arrived": [], "orch_expired": [],
            "assigned_by_class": [], "expired_by_class": [],
        }

        for ep in range(n_episodes):
            if FAIRMEC_DEBUG:
                print(f"\n[FAIRMEC] >> Starting episode {ep+1}/{n_episodes} for scenario '{scenario.name}'")
            power_multipliers = config.get("power_multipliers", [(i + 1) * 250 for i in range(n_agents)])
            agents = [FairMecAgent(power_multiplier=power_multipliers[i]) for i in range(n_agents)]

            orchestrator = Orchestrator(1, max_items, n_agents,
                                        orch_to_agent_delays=orch_to_agent_delays)

            is_chaos = scenario.name == "Chaos"
            if is_chaos:
                env = FairMecChaosEnv(
                    scenarios=benchmark.chaos_scenarios,
                    n_agents=n_agents,
                    max_items=max_items,
                    max_proposal=max_proposal,
                    trajectory_length=episode_length,
                    batch_size=1,
                    min_items=1,
                    item_classes=item_classes,
                    orchestrator=orchestrator,
                    overflow_percentage=overflow_percentage,
                    class_probabilities=class_probabilities,
                    **mec_env_kwargs,
                )
            else:
                env_kwargs = dict(
                    n_agents=n_agents,
                    max_items=max_items,
                    max_proposal=max_proposal,
                    trajectory_length=episode_length,
                    batch_size=1,
                    item_classes=item_classes,
                    orchestrator=orchestrator,
                    overflow_percentage=overflow_percentage,
                    class_probabilities=class_probabilities,
                    **mec_env_kwargs,
                )
                if scenario.lambda_a is not None:
                    env_kwargs["arrivals_poisson_lambda"] = scenario.lambda_a
                else:
                    env_kwargs["min_items"] = scenario.min_items
                    env_kwargs["max_new_items"] = scenario.max_new_items
                env = FairMecEnv(**env_kwargs)

            env._rng = np.random.default_rng(benchmark.seed * 10000 + sc_idx * 1000 + ep)
            env.agents = agents
            env.zero_bid_rewards = True

            ep_result = run_fairmec_episode(
                agents, env, orchestrator, n_agents, max_proposal,
                episode_length, inter_orch_delays,
            )

            episode_data["rewards"].append(ep_result["rewards"])
            episode_data["assigned"].append(ep_result["assigned"])
            episode_data["consumed"].append(ep_result["consumed"])
            episode_data["expired"].append(ep_result["expired"])
            episode_data["queue_lengths"].append(ep_result["queue_lengths"])
            episode_data["orch_arrived"].append(ep_result["orch_arrived"])
            episode_data["orch_expired"].append(ep_result["orch_expired"])
            episode_data["assigned_by_class"].append(ep_result["assigned_by_class"])
            episode_data["expired_by_class"].append(ep_result["expired_by_class"])

        results[scenario.name] = _compute_metrics(episode_data, n_agents)

    return results
