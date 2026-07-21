import torch


class Orchestrator:
    """Manages the negotiation queue and assigns items to agents after each round.

    The orchestrator holds the queue of items that agents negotiate over and
    resolves bids by assigning each item to the highest-bidding agent.
    """

    def __init__(self, batch_size, max_items, n_agents=0, min_value=0.0, orch_to_agent_delays=None):
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.min_value = 0.1
        self.orch_to_agent_delays = orch_to_agent_delays if orch_to_agent_delays is not None else [0.0] * n_agents
        self.item_grid = [[None] * max_items for _ in range(batch_size)]
        self.availability = torch.zeros(batch_size, max_items)
        self.overflow_items = [[] for _ in range(batch_size)]
        self.total_expired = 0
        self.total_arrived = 0
        self.total_overflowed = 0
        self.total_promoted = 0
        self.benchmark_mode = False
        self.pending_penalties = [torch.zeros(batch_size) for _ in range(n_agents)]
        self.prev_proposals = [
            torch.zeros(batch_size, max_items) for _ in range(n_agents)
        ]

    def resolve_bids(self, agents, proposals):
        """Assign each item to the highest-bidding agent.

        Args:
            agents:    list of NegotiationAgent, length n_agents
            proposals: list of tensors, each (B, M) — clamped bids per agent

        For every active item in every batch element, the agent with the highest
        bid wins the item. Ties are broken by smallest orch_to_agent_delay,
        then by lowest agent index. Won items are appended
        to each agent's ``won_items`` list.
        """
        n_agents = len(agents)
        bids = torch.stack(proposals, dim=0)
        B, M = proposals[0].shape

        total_items = 0
        all_max_items = 0

        for b in range(B):
            for m in range(M):
                if self.availability[b, m].item() <= 0:
                    continue
                item = self.item_grid[b][m]
                if item is None:
                    continue
                total_items += 1
                best_bid = bids[0, b, m].item()
                candidates = [0]
                for a in range(1, n_agents):
                    bid_val = bids[a, b, m].item()
                    if bid_val > best_bid:
                        best_bid = bid_val
                        candidates = [a]
                    elif bid_val == best_bid:
                        candidates.append(a)
                if best_bid <= self.min_value:
                    continue
                if len(candidates) == n_agents:
                    all_max_items += 1
                winner = min(candidates, key=lambda a: (self.orch_to_agent_delays[a], a))
                item.timestamp += self.orch_to_agent_delays[winner]
                agents[winner].won_items.append(item)
                agents[winner].total_assigned += 1
                if item.class_id is not None:
                    agents[winner].total_assigned_by_class[item.class_id] += 1
                self.item_grid[b][m] = None
                self.availability[b, m] = 0.0

        if all_max_items > 0:
            pct = 100.0 * all_max_items / total_items
            print(f"[resolve_bids] {pct:.1f}% of items ({all_max_items}/{total_items}) had all agents bid the max value")

    def tick(self, elapsed_seconds_items_inqueue, elapsed_seconds_overflow_items):
        """Advance timestamps and expire items that exceed their time to live.

        Grid items whose ``timestamp + elapsed`` exceeds ``time_to_live`` are
        replaced with ``None`` and their availability flag is cleared.  Overflow
        items that exceed their TTL are removed and counted in ``total_expired``.
        """
        B = len(self.item_grid)
        for b in range(B):
            M = len(self.item_grid[b])
            for m in range(M):
                item = self.item_grid[b][m]
                if item is None:
                    continue
                if item.timestamp + elapsed_seconds_items_inqueue >= item.time_to_live:
                    penalty = item.values[0] if self.benchmark_mode else item.time_to_live * item.values[0]
                    for a in range(self.n_agents):
                        self.pending_penalties[a][b] += penalty
                    self.item_grid[b][m] = None
                    self.availability[b, m] = 0.0
                    self.total_expired += 1
                else:
                    item.timestamp += elapsed_seconds_items_inqueue

        for b in range(B):
            surviving = []
            for item in self.overflow_items[b]:
                if item.timestamp + elapsed_seconds_overflow_items >= item.time_to_live:
                    penalty = item.values[0] if self.benchmark_mode else item.time_to_live * item.values[0]
                    for a in range(self.n_agents):
                        self.pending_penalties[a][b] += penalty
                    self.total_expired += 1
                else:
                    item.timestamp += elapsed_seconds_overflow_items
                    surviving.append(item)
            self.overflow_items[b] = surviving
