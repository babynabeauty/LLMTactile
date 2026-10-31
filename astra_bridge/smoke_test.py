"""Integration checks through the same client used by Astra; no model calls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from client import call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    description = call("describe")
    joints = description["joint_order"]
    obs = call("reset", {"seed": 100})
    status_before = call("status")
    time.sleep(0.25)
    assert call("status") == status_before, "Simulation advanced while idle"
    for bad in [{"actions": {"unknown": 0}}, {"actions": {joints[0]: 2}}, {"steps": 11},
                {"steps": True}, {"actions": {joints[0]: True}}, {"expected_episode": -1}]:
        try:
            call("step", bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Accepted invalid request: {bad}")
        assert call("status") == status_before
    initial_q = obs["joint_positions_rad"].copy()
    action = {name: 0.25 if "abd" not in name else 0.0 for name in joints}
    obs = call("step", {"actions": action, "steps": 4})
    assert abs(obs["sim_time_s"] - 4 * description["control_dt_s"]) < 1e-8
    assert max(abs(obs["joint_positions_rad"][n] - initial_q[n]) for n in joints) > 1e-4
    assert obs["last_actions"] == action
    active = []
    terminations = 0
    resets = 0
    start_steps = 0
    while call("status")["total_steps"] < args.steps:
        if obs["terminated"]:
            terminal_status = call("status")
            terminal_obs = call("observe")
            try:
                call("step", {"actions": {}, "steps": 1})
            except RuntimeError:
                pass
            else:
                raise AssertionError("Stepped after termination")
            assert call("status") == terminal_status
            assert call("observe") == terminal_obs
            terminations += 1
            resets += 1
            obs = call("reset", {"seed": 100 + resets})
        remaining = args.steps - obs["total_steps"]
        # Deliberately exercise open/close changes, not an Astra policy.
        phase = (obs["total_steps"] // 80) % 2
        targets = {name: (0.35 if phase else 0.0) if "abd" not in name else 0.0 for name in joints}
        obs = call("step", {"actions": targets, "steps": min(10, remaining), "note": "scripted integration test, not Astra"})
        active.append(sum(v["active_taxels"] for v in obs["tactile"].values()))
        assert np.isfinite(list(obs["joint_positions_rad"].values())).all()
        assert Path(obs["rgb_path"]).is_file()
        forbidden = {"object_pose", "object_position_m", "object_quaternion_wxyz", "orientation_error_rad", "reward", "success_metrics"}
        assert not forbidden.intersection(obs)
        if obs["total_steps"] // 100 > start_steps // 100:
            print(f"Verified {obs['total_steps']} control steps; active_taxels={active[-1]}", flush=True)
        start_steps = obs["total_steps"]
    full = call("observe", {"full": True})
    for name, values in full["tactile_full_history"].items():
        data = np.asarray(values)
        assert data.shape[-1] == 6 and np.isfinite(data).all(), name
        assert data.shape[-2] == len(description["tactile_layout"][name]["positions_link_m"])
        summary = full["tactile"][name.removeprefix("tactile_force_torque_")]
        np.testing.assert_allclose(summary["force_sum_N"], data[0, :, :3].sum(axis=0), atol=2e-6)
    output = Path(obs["output_dir"])
    call("close")
    # Check persisted terminal observations still belong to their original episode.
    for episode in sorted(output.glob("episode_*")):
        records = [json.loads(line) for line in (episode / "evaluation.jsonl").read_text().splitlines()]
        assert records[0]["step"] == 0
        assert all(b["step"] == a["step"] + 1 for a, b in zip(records, records[1:]))
        assert not any(r["terminated"] for r in records[:-1])
        stored = np.load(episode / "tactile.npz")
        assert len(stored["sim_time_s"]) == len(records)
        for name in stored["sensor_names"]:
            assert stored[str(name)].shape[-1] == 6
        with (episode / "tactile_stream.npy").open("rb") as stream:
            for index in range(len(records)):
                persisted = np.load(stream, allow_pickle=False)
                expected = np.concatenate([stored[str(n)][index] for n in stored["sensor_names"]], axis=-2)
                np.testing.assert_array_equal(persisted, expected)
            assert stream.read(1) == b"", "Unexpected extra tactile frames"
    report = {"passed": True, "control_steps": obs["total_steps"], "explicit_resets": resets,
              "terminal_guard_checks": terminations, "joint_response": True,
              "paused_while_idle": True, "invalid_actions_rejected": True,
              "min_active_taxels": min(active), "max_active_taxels": max(active),
              "six_component_tactile": True, "actor_observation_boundary": True}
    report["per_step_tactile_persistence"] = True
    report["summary_uses_newest_sensor_sample"] = True
    assert max(active) > 0 and min(active) < max(active), "No varying contact signal observed"
    assert terminations > 0, "Terminal preservation was not exercised"
    (output / "smoke_test.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
