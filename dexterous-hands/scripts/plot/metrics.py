"""Resolve a short metric query (e.g. ``consecutive_successes``) to a full
W&B logged key (e.g. ``Commands/goal_rot/consecutive_successes``)."""

from __future__ import annotations


class MetricResolutionError(ValueError):
    pass


def resolve_metric(available_keys, query: str) -> str:
    """Match `query` against the run's logged keys.

    Preference order: exact match, then ``.../<query>`` suffix, then substring.
    Raises if the chosen tier is ambiguous or nothing matches.
    """
    keys = [k for k in available_keys if not str(k).startswith("_")]

    if query in keys:
        return query

    suffix = [k for k in keys if str(k).endswith("/" + query)]
    if len(suffix) == 1:
        return suffix[0]
    if len(suffix) > 1:
        raise MetricResolutionError(
            f"Metric '{query}' is ambiguous; matches: {sorted(suffix)}"
        )

    substr = [k for k in keys if query in str(k)]
    if len(substr) == 1:
        return substr[0]
    if len(substr) > 1:
        raise MetricResolutionError(
            f"Metric '{query}' is ambiguous; matches: {sorted(substr)}"
        )

    raise MetricResolutionError(
        f"Metric '{query}' not found among {len(keys)} logged keys."
    )
