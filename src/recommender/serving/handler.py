"""AWS Lambda entry point serving the constrained LinUCB crop recommendation.

The serialized policy (per-arm ``A``/``b`` matrices, scaler, constraint
table) is read once per container from S3 at cold start; each request is
then a handful of small matrix inversions over at most ``n_arms`` feasible
crops, which is cheap enough to run inline rather than precomputing a
top-N table like the other portfolio projects (a per-context recommendation
cannot be precomputed the way a per-user list can).
"""

from __future__ import annotations

import json
import os
from typing import Any

from recommender.data.loader import CONTEXT_COLUMNS
from recommender.serving.payload import recommend_from_payload

try:  # pragma: no cover - boto3 is provided by the Lambda runtime.
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None

ARTIFACTS_BUCKET = os.environ.get("ARTIFACTS_BUCKET", "crop-bandit-dev-artifacts")
ARTIFACT_KEY = os.environ.get("ARTIFACT_KEY", "models/linucb_policy.json")

_payload = None


def _get_payload() -> dict[str, Any]:
    """Load and cache the serialized policy once per container."""
    global _payload
    if _payload is None:
        if boto3 is None:
            raise RuntimeError("boto3 is unavailable; cannot reach S3.")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        client = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")
        body = client.get_object(Bucket=ARTIFACTS_BUCKET, Key=ARTIFACT_KEY)["Body"].read()
        _payload = json.loads(body)
    return _payload


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def parse_request(event: dict[str, Any]) -> dict[str, float]:
    """Extract and validate the raw soil/climate context from an event.

    Args:
        event: API Gateway proxy event, or a bare dict for direct invocation.

    Returns:
        A mapping of every :data:`CONTEXT_COLUMNS` entry to a float.

    Raises:
        ValueError: If a required feature is missing or not numeric.
    """
    params = event.get("queryStringParameters") or {}
    if not params and set(CONTEXT_COLUMNS) & set(event):
        params = event

    context = {}
    for col in CONTEXT_COLUMNS:
        if col not in params:
            raise ValueError(f"Missing required context feature '{col}'.")
        try:
            context[col] = float(params[col])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{col}' must be numeric, got {params[col]!r}.") from None
    return context


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Return the recommended crop for a requested soil/climate context."""
    try:
        raw_context = parse_request(event)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    try:
        payload = _get_payload()
    except Exception as exc:  # noqa: BLE001 - surface as 503, never leak a stack trace.
        return _response(503, {"error": "policy artifact unavailable", "detail": str(exc)})

    recommendation = recommend_from_payload(payload, raw_context)
    return _response(200, recommendation)
