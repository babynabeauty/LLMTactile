from types import SimpleNamespace

from registry import sync_learning_rate
import torch

from smooth_checkpoint import materialize_inference_buffers


def test_restored_optimizer_rate_replaces_the_config_rate():
    alg = SimpleNamespace(learning_rate=1e-3, optimizer=SimpleNamespace(param_groups=[{"lr": 1.7e-4}, {"lr": 1.7e-4}]))
    sync_learning_rate(alg)
    assert alg.learning_rate == 1.7e-4


def test_algorithms_without_an_adaptive_rate_are_left_alone():
    alg = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]))
    sync_learning_rate(alg)
    assert not hasattr(alg, "learning_rate")


def test_rnd_inference_buffers_are_made_writable_before_reload():
    module = torch.nn.Module()
    with torch.inference_mode():
        module.register_buffer("normalizer_std", torch.ones(3))
    assert module.normalizer_std.is_inference()
    materialize_inference_buffers(module)
    assert not module.normalizer_std.is_inference()
    module.load_state_dict({"normalizer_std": torch.full((3,), 2.0)})
    torch.testing.assert_close(module.normalizer_std, torch.full((3,), 2.0))
