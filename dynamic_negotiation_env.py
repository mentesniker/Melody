import torch
import numpy as np

from negotiation_item import NegotiationItem, make_mec_item_classes
from orchestrator import Orchestrator


class DynamicNegotiationEnv:
    """Negotiation Game with a random number of active items each round.

    Each round, a random number of items (1 to max_items) are made available.
    The network dimensions are fixed at max_items, and unavailable items are
    masked via the availability tensor (zeroed values, zeroed rewards).
    """

    def __init__(self, n_agents=2, max_items=100, max_proposal=5.0,
                 trajectory_length=50, batch_size=1, min_items=1,
                 maximum_number_of_different_classes=10, item_classes=None,
                 max_new_items=None, orchestrator=None, overflow_percentage=0.0,
                 agents=None, ttl_lambda = 0, min_value=0.0,
                 user_delay_lambda=0.0, arrivals_poisson_lambda=None,
                 agent_to_orch_delays=None,
                 class_probabilities=None):
        self.agents = agents
        self.n_agents = n_agents
        self.max_items = max_items
        self.n_items = max_items
        self.max_proposal = max_proposal
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.min_items = min_items
        self.max_new_items = max_new_items if max_new_items is not None else max_items
        self.overflow_percentage = max(0.0, min(1.0, overflow_percentage))
        self.maximum_number_of_different_classes = maximum_number_of_different_classes
        self.obs_dim = (1 + 1 + n_agents + 3) * max_items + 1
        self.action_dim = max_items
        self.current_step = 0
        self.ttl_lambda = ttl_lambda
        self.zero_bid_rewards = False

        self.use_bid_masking = True
        self._processability_masks = [torch.ones(batch_size, max_items) for _ in range(n_agents)]
        self._rng = np.random.default_rng()
        if item_classes is not None:
            self._item_classes = item_classes
            self.maximum_number_of_different_classes = len(item_classes)
        else:
            self._item_classes = make_mec_item_classes(n_agents)
            self.maximum_number_of_different_classes = len(self._item_classes)

        self.user_delay_lambda = user_delay_lambda
        self.arrivals_poisson_lambda = arrivals_poisson_lambda
        self.class_probabilities = class_probabilities
        self.agent_to_orch_delays = agent_to_orch_delays if agent_to_orch_delays is not None else [0.0] * n_agents

        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            self.orchestrator = Orchestrator(batch_size, max_items, n_agents, min_value=min_value)
        #self._generate_round()

    @property
    def item_grid(self):
        return self.orchestrator.item_grid

    @property
    def availability(self):
        return self.orchestrator.availability

    @property
    def _backlog(self):
        return self.orchestrator.overflow_items

    @_backlog.setter
    def _backlog(self, value):
        self.orchestrator.overflow_items = value

    @property
    def prev_proposals(self):
        return self.orchestrator.prev_proposals

    @prev_proposals.setter
    def prev_proposals(self, value):
        self.orchestrator.prev_proposals = value

    def _generate_round(self):
        """Generate random items, preserve unassigned survivors, and merge with overflow.

        Surviving items (not assigned because all bids were <= min_value) stay
        at the front of the grid.  Remaining slots are filled first from the
        overflow queue, then from newly generated items.  Excess goes to overflow.
        """
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
                        item = NegotiationItem(
                            self.n_agents,
                            value=float(self._item_classes[idx].values[0]),
                            ttl=self._item_classes[idx].time_to_live + self.ttl_lambda,
                            execution_time=self._item_classes[idx].execution_time,
                            class_id=int(idx))
                        item.timestamp = float(self._rng.poisson(self.user_delay_lambda)) + self.agent_to_orch_delays[agent_idx]
                        new_items.append(item)
            else:
                n_new = self._rng.integers(self.min_items, self.max_new_items + 1)
                new_items = [
                    NegotiationItem(self.n_agents,
                                    value=float(self._item_classes[idx].values[0]),
                                    ttl=self._item_classes[idx].time_to_live+self.ttl_lambda,
                                    execution_time=self._item_classes[idx].execution_time,
                                    class_id=int(idx))
                    for idx in self._rng.choice(K, size=int(n_new), p=self.class_probabilities)
                ]
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

            if self.overflow_percentage > 0:
                n_overflow = int(self.overflow_percentage * n_new[b])
                overflow_extras = [
                    NegotiationItem(self.n_agents,
                                    value=float(self._item_classes[idx].values[0]),
                                    ttl=self._item_classes[idx].time_to_live+self.ttl_lambda,
                                    execution_time=self._item_classes[idx].execution_time,
                                    class_id=int(idx))
                    for idx in self._rng.choice(K, size=n_overflow, p=self.class_probabilities)
                ]
                self.orchestrator.total_arrived += len(overflow_extras)
                slots_left = M - len(on_grid)
                on_grid += overflow_extras[:slots_left]
                leftover_extras = overflow_extras[slots_left:]
            else:
                leftover_extras = []

            new_overflow = remaining_overflow + leftover_new + leftover_extras
            self.orchestrator.overflow_items[b] = new_overflow
            self.orchestrator.total_overflowed += len(new_overflow)

            self.orchestrator.item_grid[b] = [None] * M
            k = len(on_grid)
            self.orchestrator.availability[b, :k] = 1.0
            for slot_pos, item in enumerate(on_grid):
                self.orchestrator.item_grid[b][slot_pos] = item

        self.values = [self._agent_values(i) for i in range(self.n_agents)]

    def _agent_values(self, agent_idx):
        B = self.batch_size
        M = self.max_items
        vals = torch.zeros(B, M)
        mask = torch.ones(B, M)
        agent = self.agents[agent_idx] if self.agents is not None else None
        for b in range(B):
            if agent is not None:
                queued_exec_time = sum(it.execution_time for it in agent.won_items)
            else:
                queued_exec_time = 0.0
            for m in range(M):
                item = self.orchestrator.item_grid[b][m]
                if item is not None:
                    remaining = item.time_to_live - item.timestamp
                    if remaining > 0:
                        if agent is not None:
                            queue_drain = queued_exec_time / agent.power_multiplier
                            proc_time = item.execution_time / agent.power_multiplier
                            if queue_drain + proc_time >= remaining:
                                mask[b, m] = 0.0
                                continue
                        vals[b, m] = item.worth(agent_idx) / remaining
                    else:
                        mask[b, m] = 0.0
                else:
                    mask[b, m] = 0.0
        self._processability_masks[agent_idx] = mask
        return vals

    def reset(self):
        self.current_step = 0
        self.orchestrator.prev_proposals = [
            torch.zeros(self.batch_size, self.max_items)
            for _ in range(self.n_agents)
        ]
        self._generate_round()
        return self._get_observations()

    def _get_observations(self):
        obs_list = []
        B = self.batch_size
        M = self.max_items
        for i in range(self.n_agents):
            parts = [self.orchestrator.availability, self._agent_values(i)]
            parts.append(self.orchestrator.prev_proposals[i])
            for j in range(self.n_agents):
                if j != i:
                    parts.append(self.orchestrator.prev_proposals[j])

            feasibility = self._processability_masks[i]
            remaining_ttl = torch.zeros(B, M)
            exec_time_norm = torch.zeros(B, M)
            agent = self.agents[i] if self.agents is not None else None
            power = agent.power_multiplier if agent is not None else 1.0
            for b in range(B):
                for m in range(M):
                    item = self.orchestrator.item_grid[b][m]
                    if item is not None:
                        remaining = item.time_to_live - item.timestamp
                        remaining_ttl[b, m] = max(0.0, remaining) / 300.0
                        exec_time_norm[b, m] = item.execution_time / power

            parts.append(feasibility)
            parts.append(remaining_ttl)
            parts.append(exec_time_norm)

            if agent is not None:
                queue_load = len(agent.won_items) / power
                queue_feat = torch.full((B, 1), queue_load)
            else:
                queue_feat = torch.zeros(B, 1)
            parts.append(queue_feat)
            obs_list.append(torch.cat(parts, dim=-1))
        return obs_list

    def step(self, proposals):
        if self.use_bid_masking:
            clamped = [p.clamp(0.0, self.max_proposal) * self.orchestrator.availability
                       * self._processability_masks[i]
                       for i, p in enumerate(proposals)]
        else:
            clamped = [p.clamp(0.0, self.max_proposal) * self.orchestrator.availability
                       for p in proposals]

        rewards = [torch.zeros(self.batch_size) for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            rewards[i] = rewards[i] - self.orchestrator.pending_penalties[i]
            if self.agents is not None:
                rewards[i] = rewards[i] + self.agents[i].pending_reward
                self.agents[i].pending_reward = 0
        self.orchestrator.pending_penalties = [
            torch.zeros(self.batch_size) for _ in range(self.n_agents)
        ]

        self.orchestrator.prev_proposals = [p.detach() for p in clamped]
        self.current_step += 1
        done = self.current_step >= self.trajectory_length

        if not done:
            self._generate_round()

        obs_list = self._get_observations()
        return obs_list, rewards, done
