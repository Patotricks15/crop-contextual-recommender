"""Command-line entry points for training (replay), evaluation, and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recommender.data import build_bandit_log, load_observations, random_holdout_split
from recommender.evaluation import evaluate_frozen, replay_train
from recommender.features import build_feature_tables, load_feature_spec
from recommender.models import EpsilonGreedyPopularity, LinUCB, RandomPolicy, feasible_arms

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 1.0


def _load():
    observations = load_observations(ROOT / "data" / "observations.csv")
    log = build_bandit_log(observations)
    split = random_holdout_split(log)
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    tables = build_feature_tables(log, split, spec)
    return log, split, tables


def train_command(_: argparse.Namespace) -> None:
    log, split, tables = _load()
    policy = LinUCB(n_arms=log.n_arms, n_features=log.n_features, alpha=ALPHA)
    rewards = replay_train(
        policy, log, split.train_indices, tables.context_scaler, tables.feasible_range
    )
    print(
        json.dumps(
            {
                "n_rounds_trained": int(rewards.shape[0]),
                "mean_training_reward": round(float(rewards.mean()), 4),
                "last_100_mean_reward": round(float(rewards[-100:].mean()), 4),
            }
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    log, split, tables = _load()
    indices = split.validation_indices if args.partition == "validation" else split.test_indices

    rows = []
    learning_curve = {}
    for use_constraints in (True, False):
        feasible = tables.feasible_range if use_constraints else None
        label = "ConstrainedLinUCB" if use_constraints else "LinUCB"

        policy = LinUCB(n_arms=log.n_arms, n_features=log.n_features, alpha=ALPHA)
        train_rewards = replay_train(
            policy, log, split.train_indices, tables.context_scaler, feasible
        )
        report = evaluate_frozen(policy, log, indices, tables.context_scaler, feasible, label)
        rows.append(report.as_row())
        learning_curve[label] = {
            "mean_reward_first_25_rounds": round(float(train_rewards[:25].mean()), 4),
            "mean_reward_first_100_rounds": round(float(train_rewards[:100].mean()), 4),
            "mean_reward_all_training_rounds": round(float(train_rewards.mean()), 4),
        }

    random_policy = RandomPolicy()
    rows.append(
        evaluate_frozen(
            random_policy, log, indices, tables.context_scaler, tables.feasible_range, "Random"
        ).as_row()
    )

    epsilon_greedy = EpsilonGreedyPopularity(n_arms=log.n_arms)
    replay_train(
        epsilon_greedy, log, split.train_indices, tables.context_scaler, tables.feasible_range
    )
    rows.append(
        evaluate_frozen(
            epsilon_greedy, log, indices, tables.context_scaler, tables.feasible_range,
            "EpsilonGreedy",
        ).as_row()
    )

    print(
        json.dumps(
            {"test_set_generalisation": rows, "online_learning_curve": learning_curve}, indent=2
        )
    )


def recommend_command(args: argparse.Namespace) -> None:
    log, split, tables = _load()
    policy = LinUCB(n_arms=log.n_arms, n_features=log.n_features, alpha=ALPHA)
    replay_train(policy, log, split.train_indices, tables.context_scaler, tables.feasible_range)

    raw_context = np.array(
        [args.N, args.P, args.K, args.temperature, args.humidity, args.ph, args.rainfall]
    )
    context = tables.context_scaler.transform(raw_context)
    candidates = feasible_arms(raw_context, tables.feasible_range)
    scores = policy.scores(context, candidates)
    order = candidates[scores.argsort()[::-1]]

    payload = [
        {"rank": rank, "crop": str(log.arm_names[arm]), "score": round(float(score), 4)}
        for rank, (arm, score) in enumerate(
            zip(order, sorted(scores, reverse=True), strict=True), start=1
        )
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.set_defaults(func=train_command)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.set_defaults(func=evaluate_command)

    recommend = commands.add_parser("recommend")
    for feature in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        recommend.add_argument(f"--{feature}", type=float, required=True)
    recommend.set_defaults(func=recommend_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
