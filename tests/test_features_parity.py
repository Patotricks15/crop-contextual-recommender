"""Offline/online parity: the serialized-policy recommendation must match
the ranking the live, in-process policy actually produces.
"""

from __future__ import annotations

from recommender.evaluation import replay_train
from recommender.models import LinUCB, feasible_arms
from recommender.serving.payload import recommend_from_payload, serialize_policy


def test_serialized_payload_matches_live_policy_choice(log, split, feature_tables):
    policy = LinUCB(n_arms=log.n_arms, n_features=log.n_features, alpha=1.0)
    replay_train(
        policy, log, split.train_indices, feature_tables.context_scaler,
        feature_tables.feasible_range,
    )

    round_idx = split.test_indices[0]
    raw_context = log.context[round_idx]
    context = feature_tables.context_scaler.transform(raw_context)
    candidates = feasible_arms(raw_context, feature_tables.feasible_range)
    expected_arm = log.arm_names[policy.select_arm(context, candidates)]

    payload = serialize_policy(policy, feature_tables.context_scaler, feature_tables.feasible_range,
                                log.arm_names)
    raw_context_dict = dict(
        zip(
            ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            raw_context.tolist(),
            strict=True,
        )
    )
    served = recommend_from_payload(payload, raw_context_dict)

    assert served["crop"] == expected_arm
