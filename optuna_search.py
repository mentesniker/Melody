"""Optuna hyperparameter search for MELODY.

Optimizes MELODY's Gini coefficient (lower=better) and Min/Max ratio
(higher=better) across all benchmark load scenarios.

Usage:
    python optuna_search.py                              # 50 trials, default settings
    python optuna_search.py --n-trials 100 --storage sqlite:///optuna.db
    python optuna_search.py --resume --storage sqlite:///optuna.db  # resume
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import optuna
import torch

from benchmark import Benchmark
from benchmark_run import (
    build_config,
    save_model,
    load_agents,
    aggregate_results,
    _make_item_classes,
)
from dynamic_negotiation import DynamicNegotiationVerifier

STUDY_DIR = None
BENCHMARK_SEEDS = None
N_EPISODES = None
EPISODE_LENGTH = None
TRAINING_SEED = None
BATCH_SIZE = None
TRAJECTORY_LENGTH = None
MELODY_ITERATIONS_OVERRIDE = None


def train_melody_only(config, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    verifier_kwargs = {
        k: config[k] for k in [
            "n_agents", "max_items", "min_items", "max_proposal",
            "maximum_number_of_different_classes", "overflow_percentage",
            "hidden_size", "trajectory_length", "batch_size",
            "gamma", "gae_lambda", "beta", "melody_beta", "aa_discount",
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

    verifier._set_seed(config["training_seed"])
    verifier.train_melody(
        n_iterations=config["melody_iterations"],
        log_interval=999999,
    )
    save_model(verifier.melody_trainer, save_dir / "melody_agents.pt")

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    del verifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_melody_only(save_dir, seeds, n_episodes, episode_length):
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

    pt_path = save_dir / "melody_agents.pt"
    agents = load_agents(pt_path, config, device=device)

    all_results = []
    for seed in seeds:
        benchmark_kwargs = dict(max_items=config["max_items"], seed=seed)
        for k in ("lambda_a_low", "lambda_a_med", "lambda_a_high"):
            if config.get(k) is not None:
                benchmark_kwargs[k] = config[k]
        benchmark = Benchmark(**benchmark_kwargs)

        results = benchmark.run(
            agents, shared_kwargs,
            n_episodes=n_episodes, episode_length=episode_length,
            device=device,
        )
        all_results.append(results)

    return aggregate_results(all_results, config["n_agents"])


def objective(trial):
    params = {
        "melody_beta": trial.suggest_float("melody_beta", 0.5, 5.0),
        "n_candidates": trial.suggest_int("n_candidates", 3, 20),
        "lambda_lyap": trial.suggest_float("lambda_lyap", 100, 5000, log=True),
        "lyap_c": trial.suggest_float("lyap_c", 0.0001, 0.01, log=True),
        "alpha_queue": trial.suggest_float("alpha_queue", 5.0, 50.0),
        "alpha_penalty": trial.suggest_float("alpha_penalty", 5.0, 50.0),
        "alpha_gini": trial.suggest_float("alpha_gini", 1000, 100000, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99),
        "lr_actor": trial.suggest_float("lr_actor", 0.001, 0.1, log=True),
        "lr_critic": trial.suggest_float("lr_critic", 0.001, 0.1, log=True),
        "entropy_beta": trial.suggest_float("entropy_beta", 0.005, 0.05, log=True),
    }

    if MELODY_ITERATIONS_OVERRIDE is not None:
        params["melody_iterations"] = MELODY_ITERATIONS_OVERRIDE
    else:
        params["melody_iterations"] = trial.suggest_int("melody_iterations", 50, 200, step=25)

    if BATCH_SIZE is not None:
        params["batch_size"] = BATCH_SIZE
    if TRAJECTORY_LENGTH is not None:
        params["trajectory_length"] = TRAJECTORY_LENGTH

    config = build_config(training_seed=TRAINING_SEED, **params)
    trial_dir = STUDY_DIR / f"trial_{trial.number}"

    try:
        train_melody_only(config, trial_dir)
    except Exception as e:
        print(f"Trial {trial.number} training failed: {e}")
        return float("inf")

    try:
        melody_agg = benchmark_melody_only(
            trial_dir, seeds=BENCHMARK_SEEDS,
            n_episodes=N_EPISODES, episode_length=EPISODE_LENGTH,
        )
    except Exception as e:
        print(f"Trial {trial.number} benchmark failed: {e}")
        return float("inf")

    scenarios = list(melody_agg.keys())
    mean_min_max = np.mean([melody_agg[s]["min_max_ratio"]["mean"] for s in scenarios])

    trial.set_user_attr("mean_min_max_ratio", float(mean_min_max))
    for s in scenarios:
        trial.set_user_attr(f"min_max_{s}", float(melody_agg[s]["min_max_ratio"]["mean"]))

    print(f"Trial {trial.number}: mean_min_max={mean_min_max:.4f}")

    return mean_min_max


def save_results(study, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    best_params = best.params
    best_params["mean_min_max_ratio"] = best.value

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            })
    trials_data.sort(key=lambda x: x["value"], reverse=True)

    with open(output_dir / "study_summary.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    try:
        df = study.trials_dataframe()
        df.to_csv(output_dir / "optuna_history.csv", index=False)
    except Exception:
        pass

    print(f"\n{'=' * 70}")
    print("  OPTUNA SEARCH RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Best trial: #{best.number} (mean_min_max={best.value:.4f})")
    print(f"\n  Top 5 trials:")
    for entry in trials_data[:5]:
        print(f"    #{entry['number']:>3d}  mean_min_max={entry['value']:.4f}")

    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    cli_args = []
    param_to_flag = {
        "melody_beta": "--melody-beta",
        "melody_iterations": "--melody-iterations",
        "gamma": "--gamma",
        "lr_actor": "--lr-actor",
        "lr_critic": "--lr-critic",
    }
    for k, v in best.params.items():
        flag = param_to_flag.get(k)
        if flag:
            cli_args.append(f"{flag} {v}")
    if cli_args:
        print(f"\n  Validate with full benchmark:")
        print(f"    python benchmark_run.py --mode both {' '.join(cli_args)}")

    print(f"\n  Results saved to {output_dir}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for MELODY fairness metrics."
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="melody_hpo")
    parser.add_argument("--storage", default="sqlite:///optuna.db",
                        help="Optuna storage URL")
    parser.add_argument("--output-dir", default="optuna_results")
    parser.add_argument("--benchmark-seeds", nargs="+", type=int, default=[42, 137, 256])
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=50)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: use DEFAULTS)")
    parser.add_argument("--trajectory-length", type=int, default=None,
                        help="Override trajectory length (default: use DEFAULTS)")
    parser.add_argument("--melody-iterations", type=int, default=None,
                        help="Fix melody iterations instead of searching (default: search 50-200)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study from storage")
    parser.add_argument("--sampler", choices=["tpe", "cmaes", "random"], default="tpe")
    args = parser.parse_args()

    global STUDY_DIR, BENCHMARK_SEEDS, N_EPISODES, EPISODE_LENGTH, TRAINING_SEED
    global BATCH_SIZE, TRAJECTORY_LENGTH, MELODY_ITERATIONS_OVERRIDE
    STUDY_DIR = Path(args.output_dir) / args.study_name
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_SEEDS = args.benchmark_seeds
    N_EPISODES = args.n_episodes
    EPISODE_LENGTH = args.episode_length
    TRAINING_SEED = args.training_seed
    BATCH_SIZE = args.batch_size
    TRAJECTORY_LENGTH = args.trajectory_length
    MELODY_ITERATIONS_OVERRIDE = args.melody_iterations

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=TRAINING_SEED)
    elif args.sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=TRAINING_SEED)
    else:
        sampler = optuna.samplers.RandomSampler(seed=TRAINING_SEED)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    study.set_user_attr("n_trials", args.n_trials)
    study.set_user_attr("benchmark_seeds", args.benchmark_seeds)
    study.set_user_attr("n_episodes", args.n_episodes)
    study.set_user_attr("episode_length", args.episode_length)
    study.set_user_attr("training_seed", args.training_seed)
    study.set_user_attr("sampler", args.sampler)
    study.set_user_attr("batch_size", args.batch_size)
    study.set_user_attr("trajectory_length", args.trajectory_length)
    study.set_user_attr("melody_iterations_override", args.melody_iterations)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    save_results(study, STUDY_DIR)


if __name__ == "__main__":
    main()
