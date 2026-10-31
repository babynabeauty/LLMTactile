"""Real-time per-variant recordings; freeze each failed episode at its terminal frame."""

from pathlib import Path
import cv2
import numpy as np
import torch


class TeacherVideo:
    def __init__(self, env, collector, output):
        self.env, self.collector = env, collector
        self.output = Path(output)
        self.writers = []
        self.frozen = {}
        self.frame = 0

    def __call__(self, env):
        self.collector(env)
        rgb = env.cameras["rec"].render_rgb()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if rgb.ndim == 3:
            rgb = rgb[None]
        if len(rgb) != env.num_envs:
            raise RuntimeError("Expected one rendered frame per object variant")
        for i, img in enumerate(rgb):
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            frame = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            if i in self.frozen:
                frame = self.frozen[i].copy()
            else:
                pulse = env.object_wrench.state
                status = "DROPPED" if bool(self.collector.dropped[i]) else "retained"
                if "goal_pos" in env.command_manager.active_terms and not bool(self.collector.dropped[i]):
                    status = (
                        "carry completed"
                        if bool(self.collector.successes[i])
                        else "lifted" if bool(self.collector.lifted[i]) else "not lifted"
                    )
                lines = [
                    f"variant {int(self.collector.variant[i])} | {self.frame*env.dt:.2f}s | {status}",
                    f"F {float(pulse.force[i].norm()):.3f} N | torque {float(pulse.torque[i].norm()):.4f} Nm",
                    f"remaining {float((pulse.duration[i]-pulse.elapsed[i]).clamp_min(0)):.2f}s",
                ]
                cv2.rectangle(frame, (0, 0), (470, 76), (30, 30, 30), thickness=-1)
                for j, line in enumerate(lines):
                    cv2.putText(
                        frame, line, (10, 22 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                    )
                if self.collector.finished[i]:
                    self.frozen[i] = frame.copy()
            if len(self.writers) <= i:
                writer = cv2.VideoWriter(
                    str(self.output / f"variant-{i:02d}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1 / env.dt,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError("Video encoder did not open")
                self.writers.append(writer)
            self.writers[i].write(frame)
        self.frame += 1

    def close(self):
        for w in self.writers:
            w.release()
