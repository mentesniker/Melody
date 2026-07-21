import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

from dynamic_negotiation_env import DynamicNegotiationEnv
from negotiation_item import NegotiationItem
from orchestrator import Orchestrator

matplotlib.rcParams.update({"font.size": 18})

COLORS = {
    "PAA": "#4477AA",
    "Melody": "#228833",
    "PPO": "#EE6677",
    "FAIRMEC": "#CCBB44",
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


class BenchmarkScenario:

    def __init__(self, name, min_items=1, max_new_items=None, lambda_a=None):
        self.name = name
        self.min_items = min_items
        self.max_new_items = max_new_items
        self.lambda_a = lambda_a


class ChaosEnv(DynamicNegotiationEnv):

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


class CurriculumEnv(DynamicNegotiationEnv):

    def __init__(self, all_scenarios, **kwargs):
        super().__init__(**kwargs)
        self._all_scenarios = all_scenarios
        self._active_scenarios = [all_scenarios[0]]
        self._last_scenario_name = all_scenarios[0].name

    def set_active_scenarios(self, scenarios):
        self._active_scenarios = scenarios

    def _generate_round(self):
        scenario = self._active_scenarios[
            self._rng.integers(0, len(self._active_scenarios))
        ]
        self._last_scenario_name = scenario.name
        if scenario.lambda_a is not None:
            self.arrivals_poisson_lambda = scenario.lambda_a
        else:
            self.min_items = scenario.min_items
            self.max_new_items = scenario.max_new_items
        super()._generate_round()


def _gini(values, capacities=None):
    if capacities is not None:
        values = [v / max(c, 1e-12) for v, c in zip(values, capacities)]
    v = sorted(values)
    n = len(v)
    total = sum(v)
    if n == 0 or total == 0:
        return 0.0
    cumulative = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(v))
    return cumulative / (n * total)


def _compute_metrics(episode_data, n_agents, power_multipliers=None):
    rewards = np.array(episode_data["rewards"])
    assigned = np.array(episode_data["assigned"])
    consumed = np.array(episode_data["consumed"])
    expired = np.array(episode_data["expired"])
    queue_lengths = np.array(episode_data["queue_lengths"])
    orch_arrived = np.array(episode_data["orch_arrived"])
    orch_expired = np.array(episode_data["orch_expired"])

    social_welfare = rewards.sum(axis=1)
    mean_rewards = rewards.mean(axis=0)

    shifted = mean_rewards - mean_rewards.min() if mean_rewards.min() < 0 else mean_rewards
    gini = _gini(shifted.tolist(), capacities=power_multipliers)

    if power_multipliers is not None:
        norm_rewards = mean_rewards / np.array(power_multipliers, dtype=np.float64).clip(min=1e-12)
    else:
        norm_rewards = mean_rewards
    max_r = norm_rewards.max()
    min_r = norm_rewards.min()
    min_max_ratio = min_r / max_r if max_r != 0 else 0.0

    total_arrived = orch_arrived.sum()
    total_expired_orch = orch_expired.sum()
    total_expired_agents = expired.sum()
    total_expired_all = total_expired_orch + total_expired_agents
    expiry_rate = total_expired_all / max(1, total_arrived)

    all_class_ids = set()
    for ep_data in episode_data["assigned_by_class"]:
        for agent_dict in ep_data:
            all_class_ids.update(agent_dict.keys())
    class_ids = sorted(all_class_ids)

    agent_expiry_ratio_by_class = {}
    routing_pct_by_class = {}
    total_assigned_all = assigned.sum()
    for cid in class_ids:
        assigned_c = np.array([
            [agent_dict.get(cid, 0) for agent_dict in ep]
            for ep in episode_data["assigned_by_class"]
        ])
        expired_c = np.array([
            [agent_dict.get(cid, 0) for agent_dict in ep]
            for ep in episode_data["expired_by_class"]
        ])
        agent_expiry_ratio_by_class[cid] = [
            float(e / max(1, a))
            for e, a in zip(expired_c.sum(axis=0), assigned_c.sum(axis=0))
        ]
        routing_pct_by_class[cid] = [
            float(a / max(1, total_assigned_all))
            for a in assigned_c.sum(axis=0)
        ]

    return {
        "reward_mean": mean_rewards.tolist(),
        "reward_std": rewards.std(axis=0).tolist(),
        "reward_min": rewards.min(axis=0).tolist(),
        "reward_max": rewards.max(axis=0).tolist(),
        "assigned_mean": assigned.mean(axis=0).tolist(),
        "consumed_mean": consumed.mean(axis=0).tolist(),
        "expired_mean": expired.mean(axis=0).tolist(),
        "queue_length_mean": queue_lengths.mean(axis=0).tolist(),
        "social_welfare_mean": float(social_welfare.mean()),
        "social_welfare_std": float(social_welfare.std()),
        "gini": float(gini),
        "min_max_ratio": float(min_max_ratio),
        "expiry_rate": float(expiry_rate),
        "agent_expiry_ratio": [
            float(e / max(1, a))
            for e, a in zip(expired.sum(axis=0), assigned.sum(axis=0))
        ],
        "routing_pct": [
            float(a / max(1, assigned.sum()))
            for a in assigned.sum(axis=0)
        ],
        "agent_expiry_ratio_by_class": agent_expiry_ratio_by_class,
        "routing_pct_by_class": routing_pct_by_class,
    }


def _run_episode(agents, env, orchestrator, n_agents, max_proposal, episode_length, device):
    for agent in agents:
        agent.clear_won_items()
        agent.benchmark_mode = True
    orchestrator.benchmark_mode = True

    obs_list = env.reset()
    obs_list = [o.to(device) for o in obs_list]
    B = env.batch_size
    hx = [agents[i].init_hidden(B).to(device) for i in range(n_agents)]

    cumulative_rewards = [0.0] * n_agents
    queue_length_sum = [0.0] * n_agents

    for t in range(episode_length):
        actions = []
        with torch.no_grad():
            for i in range(n_agents):
                action, _, _, new_hx = agents[i].act(obs_list[i], hx[i])
                actions.append(action)
                hx[i] = new_hx

        clamped = [a.clamp(0.0, max_proposal) * orchestrator.availability for a in actions]
        orchestrator.resolve_bids(agents, clamped)

        obs_new, rewards, done = env.step(actions)
        obs_new = [o.to(device) for o in obs_new]

        orchestrator.tick(6, 6)
        for agent in agents:
            agent.consume(6)

        for i in range(n_agents):
            cumulative_rewards[i] += rewards[i].item() + agents[i].pending_reward
            agents[i].pending_reward = 0
            queue_length_sum[i] += len(agents[i].won_items)

        obs_list = obs_new

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


class Benchmark:

    def __init__(self, max_items, seed=None, lambda_a_low=None, lambda_a_med=None, lambda_a_high=None):
        self.max_items = max_items
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(0, 100000))
        if lambda_a_low is not None:
            low = BenchmarkScenario("Low Load", lambda_a=lambda_a_low)
            medium = BenchmarkScenario("Medium Load", lambda_a=lambda_a_med)
            high = BenchmarkScenario("High Load", lambda_a=lambda_a_high)
        else:
            low = BenchmarkScenario("Low Load", 1, max(1, int(0.3 * max_items)))
            medium = BenchmarkScenario("Medium Load", max(1, int(0.3 * max_items)), max(2, int(0.5 * max_items)))
            high = BenchmarkScenario("High Load", max(1, int(0.6 * max_items)), max_items)
        self.fixed_scenarios = [low, medium, high]
        self.chaos_scenarios = [low, medium, high]
        self.scenarios = self.fixed_scenarios + [BenchmarkScenario("Chaos")]

    def run(self, agents, shared_kwargs, n_episodes=100, episode_length=50, device="cpu", use_bid_masking=True):
        n_agents = shared_kwargs["n_agents"]
        max_proposal = shared_kwargs["max_proposal"]
        item_classes = shared_kwargs.get("item_classes")
        overflow_percentage = shared_kwargs.get("overflow_percentage", 0.0)
        batch_size = shared_kwargs.get("batch_size", 1)
        user_delay_lambda = shared_kwargs.get("user_delay_lambda", 0.0)
        arrivals_poisson_lambda = shared_kwargs.get("arrivals_poisson_lambda")
        agent_to_orch_delays = shared_kwargs.get("agent_to_orch_delays")
        orch_to_agent_delays = shared_kwargs.get("orch_to_agent_delays")
        class_probabilities = shared_kwargs.get("class_probabilities")
        power_multipliers = [a.power_multiplier for a in agents]

        mec_env_kwargs = {}
        if arrivals_poisson_lambda is not None:
            mec_env_kwargs["user_delay_lambda"] = user_delay_lambda
            mec_env_kwargs["arrivals_poisson_lambda"] = arrivals_poisson_lambda
            mec_env_kwargs["agent_to_orch_delays"] = agent_to_orch_delays

        results = {}
        for sc_idx, scenario in enumerate(self.scenarios):
            print(f"  Running scenario: {scenario.name} ({n_episodes} episodes)...")
            episode_data = {
                "rewards": [], "assigned": [], "consumed": [],
                "expired": [], "queue_lengths": [],
                "orch_arrived": [], "orch_expired": [],
                "assigned_by_class": [], "expired_by_class": [],
            }

            for ep in range(n_episodes):
                orchestrator = Orchestrator(batch_size, self.max_items, n_agents,
                                            orch_to_agent_delays=orch_to_agent_delays)

                is_chaos = scenario.name == "Chaos"
                if is_chaos:
                    env = ChaosEnv(
                        scenarios=self.chaos_scenarios,
                        n_agents=n_agents,
                        max_items=self.max_items,
                        max_proposal=max_proposal,
                        trajectory_length=episode_length,
                        batch_size=batch_size,
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
                        max_items=self.max_items,
                        max_proposal=max_proposal,
                        trajectory_length=episode_length,
                        batch_size=batch_size,
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
                    env = DynamicNegotiationEnv(**env_kwargs)
                env._rng = np.random.default_rng(self.seed * 10000 + sc_idx * 1000 + ep)
                env.agents = agents
                env.use_bid_masking = use_bid_masking
                env.zero_bid_rewards = True

                ep_result = _run_episode(
                    agents, env, orchestrator, n_agents, max_proposal, episode_length, device
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

            results[scenario.name] = _compute_metrics(episode_data, n_agents, power_multipliers)

        return results

    def compare(self, paa_results, ppo_results, n_agents,
                melody_results=None, fairmec_results=None,
                save_path="plots_dynamic/benchmark.png"):
        self._print_table(paa_results, ppo_results, n_agents, melody_results=melody_results, fairmec_results=fairmec_results)
        self._plot(paa_results, ppo_results, n_agents, melody_results=melody_results, fairmec_results=fairmec_results, save_path=save_path)

    def _print_table(self, paa_results, ppo_results, n_agents, melody_results=None, fairmec_results=None):
        has_melody = melody_results is not None
        has_fairmec = fairmec_results is not None
        sep = "-" * 130
        print(f"\n{sep}")
        print("  BENCHMARK RESULTS")
        print(sep)

        for scenario in self.scenarios:
            name = scenario.name
            paa = paa_results[name]
            ppo = ppo_results[name]
            mel = melody_results[name] if has_melody else None
            fmc = fairmec_results[name] if has_fairmec else None

            print(f"\n  Scenario: {name}")
            headers = f"  {'':>7} | {'PAA':>12}"
            if has_melody:
                headers += f" | {'Melody':>12}"
            headers += f" | {'PPO':>12}"
            if has_fairmec:
                headers += f" | {'FAIRMEC':>12}"
            print(headers)
            col_sep = f"  {'-'*7}-+-{'-'*12}"
            if has_melody:
                col_sep += f"-+-{'-'*12}"
            col_sep += f"-+-{'-'*12}"
            if has_fairmec:
                col_sep += f"-+-{'-'*12}"
            print(col_sep)

            for label, key in [("Welfare", "social_welfare_mean"), ("Gini", "gini"),
                               ("Min/Max", "min_max_ratio"), ("TotExp%", "expiry_rate")]:
                line = f"  {label:>7} | {paa[key]:>12.4f}"
                if has_melody:
                    line += f" | {mel[key]:>12.4f}"
                line += f" | {ppo[key]:>12.4f}"
                if has_fairmec:
                    line += f" | {fmc[key]:>12.4f}"
                print(line)

            print(f"\n  Per-agent reward (mean +/- std):")
            for i in range(n_agents):
                line = f"    Agent {i}: PAA={paa['reward_mean'][i]:>8.4f}+/-{paa['reward_std'][i]:.4f}  "
                if has_melody:
                    line += f"Melody={mel['reward_mean'][i]:>8.4f}+/-{mel['reward_std'][i]:.4f}  "
                line += f"PPO={ppo['reward_mean'][i]:>8.4f}+/-{ppo['reward_std'][i]:.4f}"
                if has_fairmec:
                    line += f"  FAIRMEC={fmc['reward_mean'][i]:>8.4f}+/-{fmc['reward_std'][i]:.4f}"
                print(line)

            print(f"\n  Per-agent expiry ratio (expired / assigned):")
            for i in range(n_agents):
                line = f"    Agent {i}: PAA={paa['agent_expiry_ratio'][i]:>8.4f}  "
                if has_melody:
                    line += f"Melody={mel['agent_expiry_ratio'][i]:>8.4f}  "
                line += f"PPO={ppo['agent_expiry_ratio'][i]:>8.4f}"
                if has_fairmec:
                    line += f"  FAIRMEC={fmc['agent_expiry_ratio'][i]:>8.4f}"
                print(line)

            class_ids = sorted(paa.get("agent_expiry_ratio_by_class", {}).keys())
            if class_ids:
                print(f"\n  Per-agent expiry ratio BY CLASS:")
                for cid in class_ids:
                    print(f"    Class {cid}:")
                    for i in range(n_agents):
                        line = f"      Agent {i}: PAA={paa['agent_expiry_ratio_by_class'][cid][i]:>8.4f}  "
                        if has_melody:
                            line += f"Melody={mel['agent_expiry_ratio_by_class'][cid][i]:>8.4f}  "
                        line += f"PPO={ppo['agent_expiry_ratio_by_class'][cid][i]:>8.4f}"
                        if has_fairmec:
                            line += f"  FAIRMEC={fmc['agent_expiry_ratio_by_class'][cid][i]:>8.4f}"
                        print(line)

            print(f"\n  Per-agent routing % (assigned / total assigned):")
            for i in range(n_agents):
                line = f"    Agent {i}: PAA={paa['routing_pct'][i]:>8.4f}  "
                if has_melody:
                    line += f"Melody={mel['routing_pct'][i]:>8.4f}  "
                line += f"PPO={ppo['routing_pct'][i]:>8.4f}"
                if has_fairmec:
                    line += f"  FAIRMEC={fmc['routing_pct'][i]:>8.4f}"
                print(line)

            if class_ids:
                print(f"\n  Per-agent routing % BY CLASS:")
                for cid in class_ids:
                    print(f"    Class {cid}:")
                    for i in range(n_agents):
                        line = f"      Agent {i}: PAA={paa['routing_pct_by_class'][cid][i]:>8.4f}  "
                        if has_melody:
                            line += f"Melody={mel['routing_pct_by_class'][cid][i]:>8.4f}  "
                        line += f"PPO={ppo['routing_pct_by_class'][cid][i]:>8.4f}"
                        if has_fairmec:
                            line += f"  FAIRMEC={fmc['routing_pct_by_class'][cid][i]:>8.4f}"
                        print(line)

        print(f"\n{sep}")

    def _plot(self, paa_results, ppo_results, n_agents, melody_results=None, fairmec_results=None, save_path="plots_dynamic/benchmark.png"):
        has_melody = melody_results is not None
        has_fairmec = fairmec_results is not None
        scenario_names = [s.name for s in self.scenarios]

        y = np.arange(n_agents)
        n_groups = 2 + int(has_melody) + int(has_fairmec)
        width = 0.8 / n_groups
        horiz = n_agents >= 6

        def _offsets():
            return [(i - (n_groups - 1) / 2) * width for i in range(n_groups)]

        def _build_groups():
            groups = [("PAA", COLORS["PAA"])]
            if has_melody:
                groups.append(("Melody", COLORS["Melody"]))
            groups.append(("PPO", COLORS["PPO"]))
            if has_fairmec:
                groups.append(("FAIRMEC", COLORS["FAIRMEC"]))
            return groups

        group_info = _build_groups()

        parts = [g[0] for g in group_info]
        base_title = "Benchmark: " + " vs ".join(parts)

        root = os.path.dirname(save_path)

        metric_keys = ["per_agent_reward", "social_welfare", "fairness", "expiry_ratio", "routing"]
        metric_titles = ["Per-Agent Reward", "Social Welfare", "Fairness & Efficiency",
                          "Per-Agent Queue Expiry Ratio", "Job Routing Distribution"]

        def _fig_size(metric_idx):
            if metric_idx in (0, 3, 4):
                if horiz:
                    return (14, max(8, 0.9 * n_agents + 3))
                return (max(12, 1.0 * n_agents + 5), 9)
            return (14, 9)

        def _style_ax(ax, metric_idx):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            grid_axis = "x" if (horiz and metric_idx in (0, 3, 4)) else "y"
            ax.grid(True, alpha=0.3, axis=grid_axis)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=14)

        skip_annot = n_agents >= 10

        for name in scenario_names:
            out_dir = os.path.join(root, "benchmark", _slug(name))
            os.makedirs(out_dir, exist_ok=True)

            all_res = [paa_results[name]]
            if has_melody:
                all_res.append(melody_results[name])
            all_res.append(ppo_results[name])
            if has_fairmec:
                all_res.append(fairmec_results[name])

            for mi, (mkey, mtitle) in enumerate(zip(metric_keys, metric_titles)):
                fig, ax = plt.subplots(1, 1, figsize=_fig_size(mi))

                if mi == 0:
                    for (res, offset), (label, c) in zip(zip(all_res, _offsets()), group_info):
                        if horiz:
                            ax.barh(y + offset, res["reward_mean"], width, xerr=res["reward_std"],
                                    label=label, color=c, capsize=3, alpha=0.85)
                        else:
                            ax.bar(y + offset, res["reward_mean"], width, yerr=res["reward_std"],
                                   label=label, color=c, capsize=3, alpha=0.85)
                    if horiz:
                        ax.set_ylabel("Agent")
                        ax.set_xlabel("Mean Reward")
                        ax.set_yticks(y)
                        ax.set_yticklabels([f"A{i}" for i in range(n_agents)])
                    else:
                        ax.set_xlabel("Agent")
                        ax.set_ylabel("Mean Reward")
                        ax.set_xticks(y)
                        ax.set_xticklabels([f"A{i}" for i in range(n_agents)], rotation=45, ha="right")

                elif mi == 1:
                    sw_x = np.arange(1)
                    for (res, offset), (label, c) in zip(zip(all_res, _offsets()), group_info):
                        ax.bar(sw_x + offset, [res["social_welfare_mean"]], width,
                               yerr=[res["social_welfare_std"]],
                               label=label, color=c, capsize=4, alpha=0.85)
                    ax.set_xticks(sw_x)
                    ax.set_xticklabels(["Social Welfare"])
                    ax.set_ylabel("Sum of Rewards")

                elif mi == 2:
                    rate_metrics = ["gini", "min_max_ratio", "expiry_rate"]
                    rate_labels = ["Gini\n(lower=fairer)", "Min/Max\n(higher=fairer)", "Expiry %\n(lower=better)"]
                    mx = np.arange(len(rate_metrics))
                    for (res, offset), (label, c) in zip(zip(all_res, _offsets()), group_info):
                        vals = [res[m] for m in rate_metrics]
                        ax.bar(mx + offset, vals, width, label=label, color=c, alpha=0.85)
                        for k, v in enumerate(vals):
                            ax.text(mx[k] + offset, v + 0.01, f"{v:.3f}",
                                    ha="center", va="bottom", fontsize=13)
                    ax.set_xticks(mx)
                    ax.set_xticklabels(rate_labels)

                elif mi == 3:
                    class_ids = sorted(all_res[0].get("agent_expiry_ratio_by_class", {}).keys())
                    hatches = ['', '//', 'xx', '..', 'oo']
                    for (res, offset), (label, c) in zip(zip(all_res, _offsets()), group_info):
                        by_class = res.get("agent_expiry_ratio_by_class", {})
                        accum = np.zeros(n_agents)
                        for ci, cid in enumerate(class_ids):
                            vals = np.array(by_class.get(cid, [0.0] * n_agents))
                            if horiz:
                                ax.barh(y + offset, vals, width, left=accum,
                                        label=f"{label} C{cid}",
                                        color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                        edgecolor='white', linewidth=0.5)
                            else:
                                ax.bar(y + offset, vals, width, bottom=accum,
                                       label=f"{label} C{cid}",
                                       color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                       edgecolor='white', linewidth=0.5)
                            if not skip_annot:
                                for k in range(n_agents):
                                    if vals[k] > 0.01:
                                        if horiz:
                                            ax.text(accum[k] + vals[k] / 2, y[k] + offset,
                                                    f"{vals[k]:.2f}", ha="center", va="center",
                                                    fontsize=10, fontweight="bold")
                                        else:
                                            ax.text(y[k] + offset, accum[k] + vals[k] / 2,
                                                    f"{vals[k]:.2f}", ha="center", va="center",
                                                    fontsize=10, fontweight="bold")
                            accum += vals
                    if horiz:
                        ax.set_ylabel("Agent")
                        ax.set_xlabel("Expired / Assigned")
                        ax.set_yticks(y)
                        ax.set_yticklabels([f"A{i}" for i in range(n_agents)])
                    else:
                        ax.set_xlabel("Agent")
                        ax.set_ylabel("Expired / Assigned")
                        ax.set_xticks(y)
                        ax.set_xticklabels([f"A{i}" for i in range(n_agents)], rotation=45, ha="right")

                elif mi == 4:
                    class_ids = sorted(all_res[0].get("agent_expiry_ratio_by_class", {}).keys())
                    hatches = ['', '//', 'xx', '..', 'oo']
                    for (res, offset), (label, c) in zip(zip(all_res, _offsets()), group_info):
                        by_class = res.get("routing_pct_by_class", {})
                        accum = np.zeros(n_agents)
                        for ci, cid in enumerate(class_ids):
                            vals = np.array([v * 100 for v in by_class.get(cid, [0.0] * n_agents)])
                            if horiz:
                                ax.barh(y + offset, vals, width, left=accum,
                                        label=f"{label} C{cid}",
                                        color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                        edgecolor='white', linewidth=0.5)
                            else:
                                ax.bar(y + offset, vals, width, bottom=accum,
                                       label=f"{label} C{cid}",
                                       color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                       edgecolor='white', linewidth=0.5)
                            if not skip_annot:
                                for k in range(n_agents):
                                    if vals[k] > 0.5:
                                        if horiz:
                                            ax.text(accum[k] + vals[k] / 2, y[k] + offset,
                                                    f"{vals[k]:.1f}%", ha="center", va="center",
                                                    fontsize=10, fontweight="bold")
                                        else:
                                            ax.text(y[k] + offset, accum[k] + vals[k] / 2,
                                                    f"{vals[k]:.1f}%", ha="center", va="center",
                                                    fontsize=10, fontweight="bold")
                            accum += vals
                    if horiz:
                        ax.set_ylabel("Agent")
                        ax.set_xlabel("Routing %")
                        ax.set_yticks(y)
                        ax.set_yticklabels([f"A{i}" for i in range(n_agents)])
                    else:
                        ax.set_xlabel("Agent")
                        ax.set_ylabel("Routing %")
                        ax.set_xticks(y)
                        ax.set_xticklabels([f"A{i}" for i in range(n_agents)], rotation=45, ha="right")

                ax.set_title(f"{name} — {mtitle}", fontsize=18)
                _style_ax(ax, mi)

                fig.tight_layout()
                fpath = os.path.join(out_dir, f"{mkey}.png")
                plt.savefig(fpath, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved benchmark plot to {fpath}")
