import numpy as np


# MEC workload parameters derived from cloud pricing, latency requirements,
# and instruction-count profiling (Tocze et al.).
#              (worth, ttl, execution_time)
MEC_WORKLOAD = [
    (0.0013,  307, 2),   # Aeneas  – forced alignment (L/H)
    (0.006,  200,   52),   # Julius  – speech recognition (L/L)
    (0.1,  20,   3),   # MR-Leo  – real-time video analysis (H/H)
]


# MEC_WORKLOAD = [
#     (1,  20, 2),   # Aeneas  – forced alignment (L/H)
#     (2,  19,   3),   # Julius  – speech recognition (L/L)
#     (3,  18,   1),   # MR-Leo  – real-time video analysis (H/H)
# ]

def make_mec_item_classes(n_agents):
    return [
        NegotiationItem(n_agents, value=w, ttl=t, execution_time=e, class_id=i)
        for i, (w, t, e) in enumerate(MEC_WORKLOAD)
    ]


class NegotiationItem:
    def __init__(self, n_agents, value, ttl, execution_time=1, class_id=None):
        self.values = np.full(n_agents, value)
        self.time_to_live = ttl
        self.timestamp = 0
        self.execution_time = execution_time
        self.class_id = class_id

    def worth(self, agent_idx):
        return self.values[agent_idx]

    def __repr__(self):
        vals = ", ".join(f"{v:.3f}" for v in self.values)
        return f"NegotiationItem([{vals}], ttl={self.time_to_live}, ts={self.timestamp})"
