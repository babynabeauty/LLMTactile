"""Small compatibility fixes for the pinned Genesis sensor implementation."""

from functools import wraps


def install_sensor_reset_fix():
    """Reset indexed drift rows in-place; tensor advanced indexing returns a copy.

    Genesis SimpleSensor.reset uses `_cur_random_walk[envs_idx].fill_(0)`.
    For a tensor/list of environment indices that leaves the original state
    unchanged. Preserve the upstream reset chain and explicitly assign the rows.
    """
    from genesis.engine.sensors.base_sensor import SimpleSensor

    original = SimpleSensor.reset.__func__
    if getattr(original, "_dex_indexed_reset", False):
        return

    @wraps(original)
    def reset(cls, metadata, ground_truth, envs_idx):
        original(cls, metadata, ground_truth, envs_idx)
        metadata._cur_random_walk[envs_idx] = 0.0

    reset._dex_indexed_reset = True
    SimpleSensor.reset = classmethod(reset)
