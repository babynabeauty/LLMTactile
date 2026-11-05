"""Passive CPU episode measurements; these never enter observations or rewards."""
import numpy as np


class PostContactResidual:
    """Time-average signed, executed residuals from the first object contact.

    Contact latches at the repair grid's 0.05 N threshold. Samples include the
    first contacting physics step and exclude steps after episode failure.
    """
    def __init__(self, n, action_joints, contact_N=.05):
        self.joints = tuple(int(j) for j in action_joints)
        self.contact_N = contact_N
        self.contacted = np.zeros(n, dtype=bool)
        self.duration = np.zeros(n)
        self.integral = np.zeros((n, len(self.joints)))

    def reset(self, lanes):
        self.contacted[lanes] = False
        self.duration[lanes] = 0.
        self.integral[lanes] = 0.

    def update(self, region_normal_N, residual, dt, active):
        self.contacted |= (np.asarray(region_normal_N).max(-1) > self.contact_N) & active
        elapsed = (self.contacted & active)*dt
        self.duration += elapsed
        self.integral += np.asarray(residual)*elapsed[:, None]

    def metrics(self, lane):
        duration = float(self.duration[lane])
        means = self.integral[lane]/duration if duration else [None]*len(self.joints)
        by_joint = dict(zip(self.joints, means))
        thumb = [by_joint[j] for j in (1, 2) if j in by_joint]
        return dict(post_contact_duration_s=duration,
            post_contact_mean_residual_rad=by_joint,
            post_contact_mean_thumb_flexion_rad=float(np.mean(thumb)) if duration and thumb else None)
