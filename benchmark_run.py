"""Train models once, save checkpoints, then benchmark with multiple seeds.

Usage:
    python benchmark_run.py                          # train + benchmark
    python benchmark_run.py --mode train             # train only, save checkpoints
    python benchmark_run.py --mode benchmark          # load checkpoints, run multi-seed benchmark
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import NegotiationAgent
from benchmark import Benchmark
from critic import Critic
from dynamic_negotiation import DynamicNegotiationVerifier
from fair_mec import run_fairmec_benchmark
from negotiation_item import NegotiationItem, make_mec_item_classes


DEFAULTS = dict(
    n_agents=3,
    maximum_number_of_different_classes=3,
    max_items=20,
    min_items=1,
    max_proposal=5.0,
    trajectory_length=50,
    batch_size=4,
    hidden_size=128,
    gamma=0.95,
    gae_lambda=0.95,
    beta=3,
    aa_discount=0.9,
    lr_actor=5e-4,
    lr_critic=7e-4,
    clip_eps=0.1,
    entropy_beta=0.02,
    max_grad_norm=1.0,
    shared_critic=True,
    self_play=False,
    overflow_percentage=0.0,
    aggregated_loss=True,
    curriculum=True,
    replay_buffer_size=80,
    replay_ratio=0.4,
    replay_samples=6,
    n_candidates=10,
    lambda_lyap=0.5,
    lyap_c=0.1,
    alpha_queue=20.0,
    alpha_penalty=30.0,
    alpha_gini=40000.0,
    n_iterations=100,
    ppo_iterations=100,
    melody_iterations=100,
    log_interval=200,
    eval_episode_length=100,
    training_seed=42,
    # --- MEC delay model ---
    user_delay_lambda=1,
    arrivals_poisson_lambda=2,
    agent_to_orch_delays=[1, 1, 2],
    orch_to_agent_delays=[1, 1, 2],
    lambda_a_low=1,
    lambda_a_med=2,
    lambda_a_high=3,
    class_probabilities=[0.7, 0.2, 0.1],
    inter_orch_delays=[[0, 1, 2],
                       [1, 0, 1],
                       [2, 1, 0]],
    power_multipliers=[8, 12, 16],
)

 

SCALAR_METRICS = [
    "social_welfare_mean", "social_welfare_std",
    "gini", "min_max_ratio", "expiry_rate",
]
PER_AGENT_METRICS = [
    "reward_mean", "reward_std",
    "agent_expiry_ratio", "routing_pct",
]


def build_config(**overrides):
    cfg = {**DEFAULTS, **overrides}
    cfg["obs_dim"] = (1 + 1 + cfg["n_agents"] + 3) * cfg["max_items"] + 1
    cfg["action_dim"] = cfg["max_items"]
    return cfg


def _make_item_classes(config):
    return make_mec_item_classes(config["n_agents"])


# --------------------------------------------------------------------------- #
#  Save / Load
# --------------------------------------------------------------------------- #

def save_model(trainer, path):
    seen_ids = set()
    unique = []
    for i, agent in enumerate(trainer.agents):
        aid = id(agent)
        if aid not in seen_ids:
            seen_ids.add(aid)
            unique.append((i, agent.state_dict()))
    torch.save(unique, path)


def load_agents(path, config, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    n_agents = config["n_agents"]
    obs_dim = config["obs_dim"]
    action_dim = config["action_dim"]
    hidden_size = config["hidden_size"]
    max_proposal = config["max_proposal"]
    shared_critic = config["shared_critic"]
    self_play = config["self_play"]

    if self_play:
        _, sd = data[0]
        critic = Critic(hidden_size).to(device)
        agent = NegotiationAgent(
            obs_dim, action_dim, critic,
            hidden_size=hidden_size, max_proposal=max_proposal,
        ).to(device)
        agent.load_state_dict(sd)
        agent.eval()
        return [agent] * n_agents

    power_multipliers = config.get("power_multipliers", [(i+1)*250 for i in range(n_agents)])
    agents = []
    for i in range(n_agents):
        critic = Critic(hidden_size).to(device)
        agent = NegotiationAgent(
            obs_dim, action_dim, critic,
            hidden_size=hidden_size, max_proposal=max_proposal,
            power_multiplier=power_multipliers[i],
        ).to(device)
        _, sd = data[i]
        agent.load_state_dict(sd)
        agent.eval()
        agents.append(agent)

    if shared_critic:
        shared = agents[0].critic
        for agent in agents[1:]:
            agent.critic = shared

    return agents


# --------------------------------------------------------------------------- #
#  Train
# --------------------------------------------------------------------------- #

def train_and_save(config, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    verifier_kwargs = {
        k: config[k] for k in [
            "n_agents", "max_items", "min_items", "max_proposal",
            "maximum_number_of_different_classes", "overflow_percentage",
            "hidden_size", "trajectory_length", "batch_size",
            "gamma", "gae_lambda", "beta", "aa_discount",
            "lr_actor", "lr_critic", "clip_eps", "entropy_beta", "max_grad_norm",
            "shared_critic", "self_play", "aggregated_loss",
            "curriculum", "replay_buffer_size", "replay_ratio", "replay_samples",
            "n_candidates", "lambda_lyap", "lyap_c",
            "alpha_queue", "alpha_penalty", "alpha_gini",
            "user_delay_lambda", "arrivals_poisson_lambda",
            "agent_to_orch_delays", "orch_to_agent_delays",
            "class_probabilities", "power_multipliers",
            "lambda_a_low", "lambda_a_med", "lambda_a_high",
        ]
    }
    verifier = DynamicNegotiationVerifier(
        **verifier_kwargs,
        device=device,
        plot_dir=str(save_dir / "plots"),
        training_seed=config["training_seed"],
    )

    seed = config["training_seed"]
    n_iter = config["n_iterations"]
    ppo_iter = config["ppo_iterations"]
    mel_iter = config["melody_iterations"]
    log_int = config["log_interval"]

    verifier._set_seed(seed)
    verifier.train_paa(n_iterations=n_iter, log_interval=log_int)
    save_model(verifier.paa_trainer, save_dir / "paa_agents.pt")
    print(f"  Saved PAA agents to {save_dir / 'paa_agents.pt'}")

    verifier._set_seed(seed)
    verifier.train_melody(n_iterations=mel_iter, log_interval=log_int)
    save_model(verifier.melody_trainer, save_dir / "melody_agents.pt")
    print(f"  Saved Melody agents to {save_dir / 'melody_agents.pt'}")

    verifier._set_seed(seed)
    verifier.train_ppo(n_iterations=ppo_iter, log_interval=log_int)
    save_model(verifier.ppo_trainer, save_dir / "ppo_agents.pt")
    print(f"  Saved PPO agents to {save_dir / 'ppo_agents.pt'}")

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {save_dir / 'config.json'}")


# --------------------------------------------------------------------------- #
#  Multi-seed benchmark
# --------------------------------------------------------------------------- #

BY_CLASS_METRICS = [
    "agent_expiry_ratio_by_class", "routing_pct_by_class",
]


def aggregate_results(all_seed_results, n_agents):
    """Aggregate benchmark results per scenario across seeds.

    all_seed_results: list of dicts, each dict is scenario_name -> metrics.
    Returns: dict of scenario_name -> metric_name -> {"mean": ..., "std": ...}
    """
    scenarios = list(all_seed_results[0].keys())
    aggregated = {}

    for scenario in scenarios:
        agg = {}
        seed_metrics = [r[scenario] for r in all_seed_results]

        for metric in SCALAR_METRICS:
            values = [m[metric] for m in seed_metrics]
            agg[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        for metric in PER_AGENT_METRICS:
            values = np.array([m[metric] for m in seed_metrics])
            agg[metric] = {
                "mean": np.mean(values, axis=0).tolist(),
                "std": np.std(values, axis=0).tolist(),
            }

        for metric in BY_CLASS_METRICS:
            all_class_ids = set()
            for m in seed_metrics:
                if metric in m:
                    all_class_ids.update(m[metric].keys())
            class_ids = sorted(all_class_ids)
            by_class_agg = {}
            for cid in class_ids:
                values = np.array([m[metric].get(cid, [0.0] * n_agents) for m in seed_metrics])
                by_class_agg[cid] = {
                    "mean": np.mean(values, axis=0).tolist(),
                    "std": np.std(values, axis=0).tolist(),
                }
            agg[metric] = by_class_agg

        aggregated[scenario] = agg

    return aggregated


def plot_aggregated_results(paa_agg, ppo_agg, melody_agg, n_agents, save_path, fairmec_agg=None):
    from benchmark import COLORS, _slug

    has_fairmec = fairmec_agg is not None
    scenarios = list(paa_agg.keys())

    y = np.arange(n_agents)
    n_groups = 4 if has_fairmec else 3
    width = 0.8 / n_groups
    horiz = n_agents >= 6
    skip_annot = n_agents >= 10

    def _offsets():
        return [(i - (n_groups - 1) / 2) * width for i in range(n_groups)]

    title_parts = ["PAA", "Melody", "PPO"]
    if has_fairmec:
        title_parts.append("FAIRMEC")

    root = Path(save_path).parent

    metric_keys = ["per_agent_reward", "social_welfare", "fairness", "expiry_ratio", "routing"]
    metric_titles = ["Per-Agent Reward", "Social Welfare", "Fairness & Efficiency",
                      "Per-Agent Queue Expiry Ratio", "Job Routing Distribution"]

    def _fig_size(mi):
        if mi in (0, 3, 4):
            if horiz:
                return (14, max(8, 0.9 * n_agents + 3))
            return (max(12, 1.0 * n_agents + 5), 9)
        return (14, 9)

    def _style_ax(ax, mi):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        grid_axis = "x" if (horiz and mi in (0, 3, 4)) else "y"
        ax.grid(True, alpha=0.3, axis=grid_axis)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=14)

    for scenario in scenarios:
        out_dir = root / "aggregated" / _slug(scenario)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_agg = [
            ("PAA", paa_agg[scenario], COLORS["PAA"]),
            ("Melody", melody_agg[scenario], COLORS["Melody"]),
            ("PPO", ppo_agg[scenario], COLORS["PPO"]),
        ]
        if has_fairmec:
            all_agg.append(("FAIRMEC", fairmec_agg[scenario], COLORS["FAIRMEC"]))

        for mi, (mkey, mtitle) in enumerate(zip(metric_keys, metric_titles)):
            fig, ax = plt.subplots(1, 1, figsize=_fig_size(mi))

            if mi == 0:
                for (label, agg, c), off in zip(all_agg, _offsets()):
                    means = agg["reward_mean"]["mean"]
                    stds = agg["reward_mean"]["std"]
                    if horiz:
                        ax.barh(y + off, means, width, xerr=stds, label=label, color=c, capsize=3, alpha=0.85)
                        if not skip_annot:
                            for k in range(n_agents):
                                ax.text(means[k] + stds[k] + 0.02, y[k] + off, f"{means[k]:.2f}",
                                        va="center", fontsize=12)
                    else:
                        ax.bar(y + off, means, width, yerr=stds, label=label, color=c, capsize=3, alpha=0.85)
                        if not skip_annot:
                            for k in range(n_agents):
                                ax.text(y[k] + off, means[k] + stds[k] + 0.01, f"{means[k]:.2f}",
                                        ha="center", va="bottom", fontsize=12)
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
                for (label, agg, c), off in zip(all_agg, _offsets()):
                    val = agg["social_welfare_mean"]["mean"]
                    err = agg["social_welfare_mean"]["std"]
                    ax.bar(sw_x + off, [val], width, yerr=[err],
                           label=label, color=c, capsize=4, alpha=0.85)
                    ax.text(sw_x[0] + off, val + err + 0.01, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=13)
                ax.set_xticks(sw_x)
                ax.set_xticklabels(["Social Welfare"])
                ax.set_ylabel("Sum of Rewards")

            elif mi == 2:
                rate_metrics = ["gini", "min_max_ratio", "expiry_rate"]
                rate_labels = ["Gini\n(lower=fairer)", "Min/Max\n(higher=fairer)", "Expiry %\n(lower=better)"]
                mx = np.arange(len(rate_metrics))
                for (label, agg, c), off in zip(all_agg, _offsets()):
                    vals = [agg[m]["mean"] for m in rate_metrics]
                    errs = [agg[m]["std"] for m in rate_metrics]
                    ax.bar(mx + off, vals, width, yerr=errs, label=label, color=c, capsize=3, alpha=0.85)
                    for k, v in enumerate(vals):
                        ax.text(mx[k] + off, v + errs[k] + 0.01, f"{v:.3f}",
                                ha="center", va="bottom", fontsize=13)
                ax.set_xticks(mx)
                ax.set_xticklabels(rate_labels)

            elif mi == 3:
                class_ids = sorted(all_agg[0][1].get("agent_expiry_ratio_by_class", {}).keys())
                hatches = ['', '//', 'xx', '..', 'oo']
                for (label, agg, c), off in zip(all_agg, _offsets()):
                    by_class = agg.get("agent_expiry_ratio_by_class", {})
                    accum = np.zeros(n_agents)
                    for ci, cid in enumerate(class_ids):
                        class_data = by_class.get(cid, {"mean": [0.0] * n_agents})
                        vals = np.array(class_data["mean"])
                        if horiz:
                            ax.barh(y + off, vals, width, left=accum,
                                    label=f"{label} C{cid}",
                                    color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                    edgecolor='white', linewidth=0.5)
                        else:
                            ax.bar(y + off, vals, width, bottom=accum,
                                   label=f"{label} C{cid}",
                                   color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                   edgecolor='white', linewidth=0.5)
                        accum += vals
                    if not skip_annot:
                        for k in range(n_agents):
                            if horiz:
                                ax.text(accum[k] + 0.01, y[k] + off, f"{accum[k]:.2f}",
                                        va="center", fontsize=12)
                            else:
                                ax.text(y[k] + off, accum[k] + 0.01, f"{accum[k]:.2f}",
                                        ha="center", va="bottom", fontsize=12)
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
                class_ids = sorted(all_agg[0][1].get("agent_expiry_ratio_by_class", {}).keys())
                hatches = ['', '//', 'xx', '..', 'oo']
                for (label, agg, c), off in zip(all_agg, _offsets()):
                    by_class = agg.get("routing_pct_by_class", {})
                    accum = np.zeros(n_agents)
                    for ci, cid in enumerate(class_ids):
                        class_data = by_class.get(cid, {"mean": [0.0] * n_agents})
                        vals = np.array([v * 100 for v in class_data["mean"]])
                        if horiz:
                            ax.barh(y + off, vals, width, left=accum,
                                    label=f"{label} C{cid}",
                                    color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                    edgecolor='white', linewidth=0.5)
                        else:
                            ax.bar(y + off, vals, width, bottom=accum,
                                   label=f"{label} C{cid}",
                                   color=c, alpha=0.85, hatch=hatches[ci % len(hatches)],
                                   edgecolor='white', linewidth=0.5)
                        accum += vals
                    if not skip_annot:
                        for k in range(n_agents):
                            if horiz:
                                ax.text(accum[k] + 0.5, y[k] + off, f"{accum[k]:.1f}%",
                                        va="center", fontsize=12)
                            else:
                                ax.text(y[k] + off, accum[k] + 0.5, f"{accum[k]:.1f}%",
                                        ha="center", va="bottom", fontsize=12)
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

            ax.set_title(f"{scenario} — {mtitle} (across seeds)", fontsize=18)
            _style_ax(ax, mi)

            fig.tight_layout()
            fpath = str(out_dir / f"{mkey}.png")
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved aggregated benchmark plot to {fpath}")


def print_aggregated_table(paa_agg, ppo_agg, melody_agg, n_agents, fairmec_agg=None):
    has_fairmec = fairmec_agg is not None
    scenarios = list(paa_agg.keys())
    sep = "=" * 140

    print(f"\n{sep}")
    print("  MULTI-SEED AGGREGATED BENCHMARK RESULTS")
    print(sep)

    for scenario in scenarios:
        paa = paa_agg[scenario]
        ppo = ppo_agg[scenario]
        mel = melody_agg[scenario]
        fmc = fairmec_agg[scenario] if has_fairmec else None

        print(f"\n  Scenario: {scenario}")
        header = f"  {'':>10} | {'PAA':>20} | {'Melody':>20} | {'PPO':>20}"
        divider = f"  {'-'*10}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}"
        if has_fairmec:
            header += f" | {'FAIRMEC':>20}"
            divider += f"-+-{'-'*20}"
        print(header)
        print(divider)

        for label, key in [("Welfare", "social_welfare_mean"), ("Gini", "gini"),
                           ("Min/Max", "min_max_ratio"), ("TotExp%", "expiry_rate")]:
            p = paa[key]
            m = mel[key]
            o = ppo[key]
            line = (f"  {label:>10} | {p['mean']:>8.4f} +/- {p['std']:<7.4f} | "
                    f"{m['mean']:>8.4f} +/- {m['std']:<7.4f} | "
                    f"{o['mean']:>8.4f} +/- {o['std']:<7.4f}")
            if has_fairmec:
                f_ = fmc[key]
                line += f" | {f_['mean']:>8.4f} +/- {f_['std']:<7.4f}"
            print(line)

        print(f"\n  Per-agent reward (mean +/- std across seeds):")
        for i in range(n_agents):
            pr = paa["reward_mean"]
            mr = mel["reward_mean"]
            or_ = ppo["reward_mean"]
            ps = paa["reward_std"]
            ms = mel["reward_std"]
            os_ = ppo["reward_std"]
            line = (f"    Agent {i}: PAA={pr['mean'][i]:>8.4f}+/-{ps['mean'][i]:.4f}  "
                    f"Melody={mr['mean'][i]:>8.4f}+/-{ms['mean'][i]:.4f}  "
                    f"PPO={or_['mean'][i]:>8.4f}+/-{os_['mean'][i]:.4f}")
            if has_fairmec:
                fr = fmc["reward_mean"]
                fs = fmc["reward_std"]
                line += f"  FAIRMEC={fr['mean'][i]:>8.4f}+/-{fs['mean'][i]:.4f}"
            print(line)

        print(f"\n  Per-agent expiry ratio (mean across seeds):")
        for i in range(n_agents):
            pe = paa["agent_expiry_ratio"]
            me = mel["agent_expiry_ratio"]
            oe = ppo["agent_expiry_ratio"]
            line = (f"    Agent {i}: PAA={pe['mean'][i]:>8.4f}+/-{pe['std'][i]:.4f}  "
                    f"Melody={me['mean'][i]:>8.4f}+/-{me['std'][i]:.4f}  "
                    f"PPO={oe['mean'][i]:>8.4f}+/-{oe['std'][i]:.4f}")
            if has_fairmec:
                fe = fmc["agent_expiry_ratio"]
                line += f"  FAIRMEC={fe['mean'][i]:>8.4f}+/-{fe['std'][i]:.4f}"
            print(line)

        class_ids = sorted(paa.get("agent_expiry_ratio_by_class", {}).keys())
        if class_ids:
            print(f"\n  Per-agent expiry ratio BY CLASS (mean across seeds):")
            for cid in class_ids:
                print(f"    Class {cid}:")
                pe = paa["agent_expiry_ratio_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                me = mel["agent_expiry_ratio_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                oe = ppo["agent_expiry_ratio_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                for i in range(n_agents):
                    line = (f"      Agent {i}: PAA={pe['mean'][i]:>8.4f}+/-{pe['std'][i]:.4f}  "
                            f"Melody={me['mean'][i]:>8.4f}+/-{me['std'][i]:.4f}  "
                            f"PPO={oe['mean'][i]:>8.4f}+/-{oe['std'][i]:.4f}")
                    if has_fairmec:
                        fe = fmc["agent_expiry_ratio_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                        line += f"  FAIRMEC={fe['mean'][i]:>8.4f}+/-{fe['std'][i]:.4f}"
                    print(line)

        print(f"\n  Per-agent routing % (mean across seeds):")
        for i in range(n_agents):
            pp = paa["routing_pct"]
            mp = mel["routing_pct"]
            op = ppo["routing_pct"]
            line = (f"    Agent {i}: PAA={pp['mean'][i]:>8.4f}+/-{pp['std'][i]:.4f}  "
                    f"Melody={mp['mean'][i]:>8.4f}+/-{mp['std'][i]:.4f}  "
                    f"PPO={op['mean'][i]:>8.4f}+/-{op['std'][i]:.4f}")
            if has_fairmec:
                fp = fmc["routing_pct"]
                line += f"  FAIRMEC={fp['mean'][i]:>8.4f}+/-{fp['std'][i]:.4f}"
            print(line)

        if class_ids:
            print(f"\n  Per-agent routing % BY CLASS (mean across seeds):")
            for cid in class_ids:
                print(f"    Class {cid}:")
                pp = paa["routing_pct_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                mp = mel["routing_pct_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                op = ppo["routing_pct_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                for i in range(n_agents):
                    line = (f"      Agent {i}: PAA={pp['mean'][i]:>8.4f}+/-{pp['std'][i]:.4f}  "
                            f"Melody={mp['mean'][i]:>8.4f}+/-{mp['std'][i]:.4f}  "
                            f"PPO={op['mean'][i]:>8.4f}+/-{op['std'][i]:.4f}")
                    if has_fairmec:
                        fp = fmc["routing_pct_by_class"].get(cid, {"mean": [0]*n_agents, "std": [0]*n_agents})
                        line += f"  FAIRMEC={fp['mean'][i]:>8.4f}+/-{fp['std'][i]:.4f}"
                    print(line)

    print(f"\n{sep}")


def run_multi_seed_benchmark(save_dir, benchmark_seeds, n_episodes=5, episode_length=1000):
    save_dir = Path(save_dir)
    with open(save_dir / "config.json") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_classes = _make_item_classes(config)

    shared_kwargs = dict(
        n_agents=config["n_agents"],
        max_proposal=config["max_proposal"],
        item_classes=item_classes,
        overflow_percentage=config["overflow_percentage"],
        batch_size=1,
        user_delay_lambda=config.get("user_delay_lambda", 0.0),
        arrivals_poisson_lambda=config.get("arrivals_poisson_lambda"),
        agent_to_orch_delays=config.get("agent_to_orch_delays"),
        orch_to_agent_delays=config.get("orch_to_agent_delays"),
        class_probabilities=config.get("class_probabilities"),
    )

    model_names = ["paa", "ppo", "melody"]
    all_results = {name: [] for name in model_names}
    fairmec_results_all = []

    for seed_idx, seed in enumerate(benchmark_seeds):
        print(f"\n{'='*60}")
        print(f"  Benchmark seed {seed_idx+1}/{len(benchmark_seeds)}: {seed}")
        print(f"{'='*60}")

        benchmark_kwargs = dict(max_items=config["max_items"], seed=seed)
        for k in ("lambda_a_low", "lambda_a_med", "lambda_a_high"):
            if config.get(k) is not None:
                benchmark_kwargs[k] = config[k]
        benchmark = Benchmark(**benchmark_kwargs)

        for model_name in model_names:
            pt_path = save_dir / f"{model_name}_agents.pt"
            if not pt_path.exists():
                print(f"  Skipping {model_name} — checkpoint not found at {pt_path}")
                continue

            agents = load_agents(pt_path, config, device=device)
            print(f"\n  Running {model_name.upper()} benchmark (seed={seed})...")
            results = benchmark.run(
                agents, shared_kwargs,
                n_episodes=n_episodes, episode_length=episode_length,
                device=device,
            )
            all_results[model_name].append(results)

        print(f"\n  Running FAIRMEC benchmark (seed={seed})...")
        fairmec_results = run_fairmec_benchmark(
            config, benchmark,
            n_episodes=n_episodes, episode_length=episode_length,
        )
        fairmec_results_all.append(fairmec_results)

    paa_agg = aggregate_results(all_results["paa"], config["n_agents"])
    ppo_agg = aggregate_results(all_results["ppo"], config["n_agents"])
    melody_agg = aggregate_results(all_results["melody"], config["n_agents"])
    fairmec_agg = aggregate_results(fairmec_results_all, config["n_agents"])

    print_aggregated_table(paa_agg, ppo_agg, melody_agg, config["n_agents"], fairmec_agg=fairmec_agg)
    plot_aggregated_results(
        paa_agg, ppo_agg, melody_agg, config["n_agents"],
        save_path=str(save_dir / "aggregated_benchmark.png"),
        fairmec_agg=fairmec_agg,
    )

    agg_path = save_dir / "aggregated_results.json"
    serializable = {
        name: {
            scenario: {
                metric: agg_data
                for metric, agg_data in scenario_data.items()
            }
            for scenario, scenario_data in agg.items()
        }
        for name, agg in [("paa", paa_agg), ("ppo", ppo_agg), ("melody", melody_agg), ("fairmec", fairmec_agg)]
    }
    with open(agg_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved aggregated results to {agg_path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train models once, checkpoint, then benchmark with multiple seeds."
    )
    parser.add_argument("--mode", choices=["train", "benchmark", "both"], default="both")
    parser.add_argument("--save-dir", default="checkpoints/default")
    parser.add_argument("--training-seed", type=int, default=DEFAULTS["training_seed"])
    parser.add_argument(
        "--benchmark-seeds", nargs="+", type=int,
        default=[42, 137, 256, 512, 999, 1234, 2048, 3141, 4096, 7777],
    )
    parser.add_argument("--n-iterations", type=int, default=DEFAULTS["n_iterations"])
    parser.add_argument("--ppo-iterations", type=int, default=DEFAULTS["ppo_iterations"])
    parser.add_argument("--melody-iterations", type=int, default=DEFAULTS["melody_iterations"])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=DEFAULTS["eval_episode_length"])
    parser.add_argument("--lambda-a-low", type=int, default=DEFAULTS["lambda_a_low"])
    parser.add_argument("--lambda-a-med", type=int, default=DEFAULTS["lambda_a_med"])
    parser.add_argument("--lambda-a-high", type=int, default=DEFAULTS["lambda_a_high"])
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        config = build_config(
            training_seed=args.training_seed,
            n_iterations=args.n_iterations,
            ppo_iterations=args.ppo_iterations,
            melody_iterations=args.melody_iterations,
            lambda_a_low=args.lambda_a_low,
            lambda_a_med=args.lambda_a_med,
            lambda_a_high=args.lambda_a_high,
        )
        train_and_save(config, args.save_dir)

    if args.mode in ("benchmark", "both"):
        run_multi_seed_benchmark(
            args.save_dir, args.benchmark_seeds,
            n_episodes=args.n_episodes,
            episode_length=args.episode_length,
        )


if __name__ == "__main__":
    main()
