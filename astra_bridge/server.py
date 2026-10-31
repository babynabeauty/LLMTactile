"""Paused, single-environment control bridge for the current Codex agent.

The model is outside this process. This process only simulates, validates,
records and serves observations. All Genesis work stays on the main thread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(ROOT / "dexterous-hands/src"))

from protocol import validate_step


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build_env(output):
    import eden as en  # Must precede torch / genesis imports.
    import genesis as gs
    from eden.envs.base import RLEnvBase
    from eden.options import CameraOptions, CamerasOptions, EventManagerOptions
    from registry import _load_tasks, get_task_config

    en.init(backend=gs.gpu, logging_level=30, log_root_path=str(output / "engine"))
    _load_tasks()
    config = get_task_config(task_name="in_hand_repose", modifiers={
        "robot": "wuji", "obj": "cube", "sensors": "low-hand/force_torque",
        "tactile_imperfection": "clean", "temporal_reduction": "last",
    })
    config.env_options.num_envs = 1
    config.env_options.num_eval_envs = 1
    config.env_options.background_color = (0.12, 0.15, 0.20)
    # Only reset placement survives. No mass/friction/actuator randomization or kicks.
    config.event_options = EventManagerOptions(place_obj=config.event_options.place_obj)
    for term in config.observation_options.proprio.__pydantic_extra__.values():
        if hasattr(term, "noise"):
            term.noise = None
    for options in config.sensors_options.__pydantic_extra__.values():
        if hasattr(options, "sensor") and hasattr(options.sensor, "draw_debug"):
            options.sensor.draw_debug = False
    config.cameras_options = CamerasOptions(rec=CameraOptions(
        cam_pos=(0.37, -0.46, 0.86), cam_lookat=(0.0, 0.015, 0.49),
        cam_resolution=(768, 640), cam_fov=42.0,
    ))
    config.save_as_yaml(str(output / "resolved_config.yaml"), lock_config=False)

    class ManualResetEnv(RLEnvBase):
        _allow_reset = False

        def _reset_idx(self, envs_idx=None):
            if self._allow_reset:
                return super()._reset_idx(envs_idx)
            return {}

        def reset(self, envs_idx=None, *, seed=None):
            self._allow_reset = True
            try:
                return super().reset(envs_idx, seed=seed)
            finally:
                self._allow_reset = False

    env = ManualResetEnv.from_config(config, show_viewer=False, eval_mode=True)
    env.build()
    return env, config


class Bridge:
    def __init__(self, output: Path, label: str):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.env, self.config = build_env(self.output)
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.label = label
        self.controller = self.env.action_manager.get_term("dofs_pos_controller")
        self.robot = self.env.entities["robot"]
        self.joints = list(self.controller.dofs_name)
        self.sensor_names = sorted(n for n in self.env.sensors if n.startswith("tactile_force_torque_"))
        self.episode = -1
        self.total_steps = 0
        self.total_decisions = 0
        self.done = True
        self.closed = False
        self.episode_log = []
        self.last_response_time = time.monotonic()
        self.last_frame = None
        self.records = []
        self.force_records = {n: [] for n in self.sensor_names}
        self.trace = (self.output / "interaction.jsonl").open("a", buffering=1)
        json_write(self.output / "manifest.json", {
            "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "label": label, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "task": "in_hand_repose", "robot": "wuji",
            "sensors": "low-hand/force_torque", "imperfection": "clean", "debug_configuration": True,
            "simulation_dt": self.env.sim_dt, "control_dt": self.env.dt,
            "observation": "RGB + proprioception + tactile summaries/full arrays + goal quaternion",
            "excluded_actor_inputs": ["object_pose", "orientation_error", "reward", "success_metrics"],
            "model_execution": "external current Codex conversation; no model API inside server",
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
                "astra_bridge/server.py", "astra_bridge/protocol.py", "astra_bridge/client.py",
                "Eden/eden/envs/base.py", "Eden/eden/utils/geom.py", "Eden/eden/managers/terms/utils.py",
                "Eden/eden/managers/termination_manager.py",
                "astra_bridge/requirements.lock",
            ]},
        })
        json_write(self.output / "description.json", self.describe())

    def array(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return self.np.asarray(value)

    def vector(self, value):
        return self.array(value).reshape(-1).astype(float).tolist()

    def describe(self):
        lower, upper = self.robot.get_dofs_limit(dofs_idx_local=self.controller.dofs_idx_local)
        scales = self.np.broadcast_to(self.array(self.controller._scale), (1, len(self.joints)))[0]
        offsets = self.array(self.controller._offset).reshape(-1)
        layout = {}
        for name in self.sensor_names:
            sensor = self.env.sensors[name]
            raw = sensor._sensor
            layout[name] = {
                "link": sensor._options.attach_link_name,
                "positions_link_m": self.array(raw.probe_local_pos).reshape(-1, 3).tolist(),
                "normals_link": sensor._options.tactile_probe_local_normal,
                "radius_m": self.vector(raw._options.probe_radius),
                "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
                "frame": "sensor attachment link; force and torque are not world-frame vectors",
            }
        return {
            "robot": "wuji", "task": "in_hand_repose", "control_dt_s": self.env.dt,
            "physics_dt_s": self.env.sim_dt, "default_action_steps": 4, "max_action_steps": 10,
            "action_formula": "requested_joint_target_rad = offset_rad + normalized_action * scale_rad; upstream modifiers/PD act afterwards",
            "action_range": [-1, 1], "joint_order": self.joints,
            "joints": {n: {"index": i, "lower_rad": self.vector(lower)[i], "upper_rad": self.vector(upper)[i],
                           "offset_rad": float(offsets[i]), "scale_rad": float(scales[i])}
                       for i, n in enumerate(self.joints)},
            "tactile_layout": layout,
            "tactile_model": "Upstream KinematicTaxel spring-damper estimate, not solver contact impulse. Torque is about the attachment link origin and includes twist damping; taxel sums are not a calibrated net wrench. Contact filtering is unchanged from upstream.",
            "tactile_history_order": "newest first; summary uses index 0; history samples are separated by physics_dt_s",
            "goal": "world-frame quaternion [w,x,y,z]; translucent ghost shows the target",
            "camera": self.config.cameras_options.rec.model_dump(mode="json"),
            "actor_contract": "No object pose, true pose error, rewards or success metrics. Observe RGB for object state.",
        }

    def read_tactile(self):
        full, summaries = {}, {}
        for name in self.sensor_names:
            sensor = self.env.sensors[name]
            reading = sensor.read()
            count = sensor._sensor.n_probes
            # Genesis preserves the layout's grid axes in each field. Flatten
            # spatial axes, while retaining the temporal substep-history axis.
            force = self.array(reading.force)[0].reshape(-1, count, 3)
            torque = self.array(reading.torque)[0].reshape(-1, count, 3)
            history = self.np.concatenate([force, torque], axis=-1)
            # Genesis TensorRingBuffer.at(0) is the most recent snapshot.
            latest = history[0] if history.ndim == 3 else history
            if not self.np.isfinite(history).all():
                raise RuntimeError(f"Non-finite tactile readings: {name}")
            full[name] = history.copy()
            norms = self.np.linalg.norm(latest[:, :3], axis=-1)
            idx = int(norms.argmax())
            summaries[name.removeprefix("tactile_force_torque_")] = {
                "taxels": len(latest), "active_taxels": int((norms > 1e-5).sum()),
                "force_sum_N": latest[:, :3].sum(axis=0).round(6).tolist(),
                "local_taxel_torque_sum_Nm": latest[:, 3:].sum(axis=0).round(7).tolist(),
                "peak_force_N": round(float(norms[idx]), 6),
                "strongest_taxel": idx, "strongest_force_torque": latest[idx].round(7).tolist(),
            }
        return full, summaries

    def render(self):
        from PIL import Image
        rgb = self.array(self.env.cameras["rec"].render_rgb())
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = rgb.astype(self.np.uint8)
        path = self.episode_dir / "frames" / f"{self.episode_step:06d}.png"
        Image.fromarray(rgb).save(path)
        self.last_frame = str(path)
        return self.last_frame

    def status(self):
        return {"episode": self.episode, "episode_step": getattr(self, "episode_step", 0),
                "sim_time_s": round(getattr(self, "episode_step", 0) * self.env.dt, 9),
                "total_steps": self.total_steps, "total_decisions": self.total_decisions,
                "terminated": self.done, "paused": True, "output_dir": str(self.output)}

    def observe(self, full=False):
        if self.episode < 0:
            raise ValueError("Call reset first")
        raw, summaries = self.read_tactile()
        q = self.robot.get_dofs_pos(dofs_idx_local=self.controller.dofs_idx_local)
        dq = self.robot.get_dofs_vel(dofs_idx_local=self.controller.dofs_idx_local)
        goal = self.env.command_manager.get_term("goal_rot").quat
        result = {**self.status(), "rgb_path": self.last_frame,
                  "joint_positions_rad": dict(zip(self.joints, self.vector(q))),
                  "joint_velocities_rad_s": dict(zip(self.joints, self.vector(dq))),
                  "last_actions": dict(zip(self.joints, self.actions)),
                  "goal_quaternion_wxyz": self.vector(goal), "tactile": summaries}
        if full:
            result["tactile_full_history"] = {k: v.tolist() for k, v in raw.items()}
        # Fail visibly rather than emitting NaNs to the controller.
        json.dumps(result, allow_nan=False)
        return result

    def reset(self, seed=42):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if self.episode >= 0:
            self.finish_episode("terminated" if self.done else "explicit_reset")
        self.episode += 1
        self.episode_step = 0
        self.episode_seed = seed
        self.done = False
        self.actions = [0.0] * len(self.joints)
        self.records = []
        self.force_records = {n: [] for n in self.sensor_names}
        self.episode_dir = self.output / f"episode_{self.episode:03d}"
        (self.episode_dir / "frames").mkdir(parents=True)
        self.env.reset(seed=seed)
        self.render()
        self.record_state()
        result = self.observe()
        self.log({"command": "reset", "seed": seed, "response": result})
        self.last_response_time = time.monotonic()
        return result

    def record_state(self, goal_before=None):
        """Privileged measurements are written for offline scoring, never returned."""
        cmd = self.env.command_manager.get_term("goal_rot")
        obj = self.env.entities["obj"]
        quat = self.array(obj.get_quat()).reshape(4)
        goal = self.array(cmd.quat if goal_before is None else goal_before).reshape(4)
        cosine = abs(float(self.np.dot(quat, goal) / (self.np.linalg.norm(quat) * self.np.linalg.norm(goal))))
        error = 2 * math.acos(min(1.0, max(0.0, cosine)))
        raw, _ = self.read_tactile()
        for name, value in raw.items():
            self.force_records[name].append(value)
        # Append one self-contained NPY array per control step (including reset).
        # The stream survives a later process failure; NPZ remains the convenient
        # indexed export at reset/close. Taxels follow description sensor order.
        with (self.episode_dir / "tactile_stream.npy").open("ab") as tactile_file:
            self.np.save(tactile_file, self.np.concatenate([raw[n] for n in self.sensor_names], axis=-2), allow_pickle=False)
        success = bool(getattr(cmd, "_last_success", self.torch.tensor([False]))[0].item()) if self.episode_step else False
        record = {"step": self.episode_step, "sim_time_s": self.episode_step * self.env.dt,
                  "joint_positions_rad": self.vector(self.robot.get_dofs_pos(self.controller.dofs_idx_local)),
                  "actions": self.actions.copy(), "object_position_m": self.vector(obj.get_pos()),
                  "object_quaternion_wxyz": quat.tolist(), "graded_goal_quaternion_wxyz": goal.tolist(),
                  "orientation_error_rad": error, "goal_reached_event": success and not self.done,
                  "terminated": self.done, "frame": self.last_frame,
                  "object_dropped": bool(self.env.termination_manager.get_term_dones("obj_dropped")[0].item()) if self.done else False,
                  "timed_out": bool(self.env.termination_manager.timeouts[0].item()) if self.done else False,
                  "active_taxels": sum(int((self.np.linalg.norm((v[0] if v.ndim == 3 else v)[:, :3], axis=-1) > 1e-5).sum()) for v in raw.values())}
        self.records.append(record)
        with (self.episode_dir / "evaluation.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def step(self, payload):
        if self.done or self.episode < 0:
            raise ValueError("Episode has ended or is not initialized; call reset explicitly")
        for key, expected in [("expected_episode", self.episode), ("expected_step", self.episode_step)]:
            if key in payload and (type(payload[key]) is not int or payload[key] != expected):
                raise ValueError(f"Stale {key}: expected {expected}")
        actions, steps = validate_step(payload, self.joints, self.actions)
        before = self.observe()
        wait_seconds = time.monotonic() - self.last_response_time
        self.actions = actions
        self.total_decisions += 1
        tensor = self.torch.tensor([actions], device=self.env.device, dtype=self.torch.float32)
        for _ in range(steps):
            goal_before = self.env.command_manager.get_term("goal_rot").quat.detach().clone()
            _, _, terminated, truncated, _ = self.env.step(tensor)
            self.episode_step += 1
            self.total_steps += 1
            self.done = bool((terminated | truncated)[0].item())
            self.render()
            self.record_state(goal_before)
            if self.done:
                break
        result = self.observe()
        self.log({"command": "step", "request": payload, "before": before, "response": result,
                  "decision_wait_seconds": wait_seconds, "wall_time": time.time()})
        if self.done:
            self.save_tactile()
        self.last_response_time = time.monotonic()
        return result

    def log(self, record):
        self.trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def save_tactile(self):
        arrays = {n: self.np.stack(v) for n, v in self.force_records.items() if v}
        arrays["sim_time_s"] = self.np.array([r["sim_time_s"] for r in self.records])
        arrays["sensor_names"] = self.np.array(self.sensor_names)
        arrays["components"] = self.np.array(["fx_N", "fy_N", "fz_N", "tx_Nm", "ty_Nm", "tz_Nm"])
        self.np.savez_compressed(self.episode_dir / "tactile.npz", **arrays)

    def finish_episode(self, reason):
        if not self.records:
            return
        self.save_tactile()
        video = self.episode_dir / "rollout.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(1 / self.env.dt),
                        "-i", str(self.episode_dir / "frames/%06d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(video)], check=True)
        errors = [r["orientation_error_rad"] for r in self.records]
        summary = {"episode": self.episode, "seed": self.episode_seed, "end_reason": reason,
                   "steps": self.episode_step, "duration_sim_s": self.episode_step * self.env.dt,
                   "goal_reached_events": sum(r["goal_reached_event"] for r in self.records),
                   "initial_error_rad": errors[0], "final_error_rad": errors[-1], "minimum_error_rad": min(errors),
                   "terminated": self.done, "object_dropped": self.records[-1]["object_dropped"],
                   "timed_out": self.records[-1]["timed_out"], "video": str(video)}
        json_write(self.episode_dir / "summary.json", summary)
        self.episode_log.append(summary)

    def close(self):
        if not self.closed:
            if self.episode >= 0:
                self.finish_episode("terminated" if self.done else "controller_stopped")
            json_write(self.output / "summary.json", {"label": self.label, "total_control_steps": self.total_steps,
                       "total_decisions": self.total_decisions, "episodes": self.episode_log})
            self.trace.close()
            self.closed = True
        return {"closed": True, "output_dir": str(self.output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="astra_live")
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/astra_bridge" / time.strftime("%Y%m%d_%H%M%S")
    # Reserve the port before expensive GPU initialization.
    server = HTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    bridge = Bridge(output, args.label)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 1_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("Payload must be a JSON object")
                command = self.path.strip("/")
                if command == "describe": result = bridge.describe()
                elif command == "status": result = bridge.status()
                elif command == "reset": result = bridge.reset(payload.get("seed", 42))
                elif command == "observe": result = bridge.observe(payload.get("full", False))
                elif command == "step": result = bridge.step(payload)
                elif command == "close": result = bridge.close()
                else: raise ValueError(f"Unknown command: {command}")
                encoded = json.dumps({"ok": True, "result": result}, allow_nan=False).encode()
                code = 200
            except ValueError as exc:
                encoded = json.dumps({"ok": False, "error": str(exc)}).encode()
                code = 400
            except Exception as exc:
                traceback.print_exc()
                encoded = json.dumps({"ok": False, "error": str(exc)}).encode()
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            print(format % args, flush=True)

    server.RequestHandlerClass = Handler
    print(f"BRIDGE_READY http://127.0.0.1:{args.port} output={bridge.output}", flush=True)
    try:
        while not bridge.closed:
            server.handle_request()
    except KeyboardInterrupt:
        bridge.close()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
