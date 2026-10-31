"""Faithful, dependency-light port of the HydroShear / HydroFOTS marker-motion model.

Source: ``reference/hydroshear/rl/hydroshear_sensors/hydroshear_sensors.py``
(``HydroFotsSensor.get_marker_dilation`` / ``get_marker_shear`` / ``step_hydrosoft_forces``).

The marker math here is copied verbatim from the reference (same coefficients, same
tensor algebra, same stateful hydrosoft force integrator). Only the two things the
reference gets from heavyweight deps are swapped for analytic primitives, since our
comparison indenter is a literal sphere and the elastomer is a flat pad:

  * ``isaacgym`` / ``urdfpy`` / ``open3d`` / ``pytorch_volumetric`` SDF sensors
    -> ``SphereSDF`` (indenter) and ``PlaneSDF`` (elastomer), exact for primitives.
  * mesh point-cloud sampling -> ``fibonacci_hemisphere`` for the indenter surface.

Everything else (the displacement coefficient, Coulomb-clamped tangential force, the
per-marker Gaussian falloffs, the dilation outflow) is the reference model unchanged.

Frame: pad/elastomer surface at z=0, gel occupies z<0 (so elastomer SDF = z, negative
inside), indenter sphere above pressing down in -z. ``reverse_z=False`` matches this
(the reference's z-up/z-down convention is otherwise resolved by its MeshSDFSensor).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --- analytic SDF + point-cloud primitives (replace the reference's mesh/SDF deps) ---------
class SphereSDF:
    """Indenter SDF: signed distance to a sphere (negative inside). center (3,), radius (m)."""

    def __init__(self, center, radius, device="cpu"):
        self.center = torch.as_tensor(center, dtype=torch.float32, device=device).view(1, 1, 3)
        self.radius = float(radius)

    def get_sdf(self, pts):  # pts (E, N, 3) -> (E, N)
        return torch.norm(pts - self.center, dim=-1) - self.radius, None


class PlaneSDF:
    """Elastomer SDF: flat pad, surface at z=0, gel at z<0 (sdf = z, negative inside). Normal +z."""

    def get_sdf(self, pts):  # pts (E, N, 3) -> (E, N)
        return pts[..., 2].clone(), None

    def get_normal_axis(self):
        return 2

    def get_tangent_axes(self):
        return [0, 1]


def fibonacci_hemisphere(n, z_max=0.4):
    """Unit directions on the lower part of a sphere (dir_z <= z_max) via a Fibonacci spiral.

    Only points near/under the contact matter (the model gates the rest by ``heaviside(-sdf)``),
    so we keep the lower cap for efficiency. Returns ``(P, 3)`` float32.
    """
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)            # polar angle, 0=top
    theta = np.pi * (1 + 5**0.5) * k          # golden-angle azimuth
    dirs = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return dirs[dirs[:, 2] <= z_max].astype(np.float32)


# --- the HydroFOTS marker model (verbatim math, primitive SDFs injected) -------------------
class HydroFotsRef:
    """Port of ``HydroFotsSensor``'s marker-motion math (coefficients + algebra unchanged).

    Reference defaults: I=K=E=A=1, mu=1e4, lambda_s=300, lambda_d=700, scales=1000/0.065.
    We pass scales=1 and best-fit them downstream, and expose lambda_s/lambda_d/mu.
    """

    def __init__(self, indenter_sdf, elastomer_sdf, *, mu=1.0e4, E=1.0, A=1.0, K=1.0, I=1.0,
                 lambda_s=300.0, lambda_d=700.0, dilate_scale=1.0, shear_scale=1.0, device="cpu"):
        self.indenter_sdf_sensor = indenter_sdf
        self.elastomer_sdf_sensor = elastomer_sdf
        self.device = device
        t = lambda v: torch.tensor(float(v), device=device)
        self.I, self.K, self.E, self.A, self.mu = t(I), t(K), t(E), t(A), t(mu)
        self.lambda_s, self.lambda_d = t(lambda_s), t(lambda_d)
        self.dilate_scale, self.shear_scale = t(dilate_scale), t(shear_scale)
        self.hydrosoft_forces = None
        self.prev_sdf = None
        self.prev_indenter_pts_in_elastomer = None

    def initialize(self, num_envs, num_indenter_pts):
        self.hydrosoft_forces = torch.zeros((num_envs, num_indenter_pts, 3), device=self.device)
        self.prev_sdf = None
        self.prev_indenter_pts_in_elastomer = None

    def step_hydrosoft_forces(self, sdf, indenter_pts_in_elastomer, reverse_z=False):
        """Stateful hydrosoft force per indenter point (verbatim from the reference)."""
        if self.prev_sdf is None:
            self.prev_sdf = sdf
            self.prev_indenter_pts_in_elastomer = indenter_pts_in_elastomer

        alpha_d = -torch.divide(F.relu(-self.prev_sdf) - F.relu(-sdf), self.prev_sdf - sdf)
        alpha_isnan = torch.isnan(alpha_d)
        alpha_d[alpha_isnan] = -0.5 * (torch.sign(sdf[alpha_isnan]) - 1)
        displacement = alpha_d.unsqueeze(-1) * (self.prev_indenter_pts_in_elastomer - indenter_pts_in_elastomer)

        normal_axis = self.elastomer_sdf_sensor.get_normal_axis()
        tangent_axes = self.elastomer_sdf_sensor.get_tangent_axes()

        fn = self.hydrosoft_forces[:, :, normal_axis] + self.E * self.A * displacement[:, :, normal_axis]
        ft = self.hydrosoft_forces[:, :, tangent_axes] + (self.K * self.A * displacement[:, :, tangent_axes])

        fn_bar = F.relu(fn * (-1 if reverse_z else 1))
        norm_ft = torch.norm(ft, dim=-1)
        ft_bar = torch.minimum(self.mu * fn_bar, norm_ft).unsqueeze(-1) * ft / (norm_ft.unsqueeze(-1) + 1e-8)
        fbar = torch.cat([ft_bar, fn_bar.unsqueeze(-1) * (-1 if reverse_z else 1)], dim=-1)

        fbar = torch.heaviside(-sdf, values=torch.tensor([0.0], device=self.device)).unsqueeze(-1) * fbar
        return fbar

    def get_marker_shear(self, tactile_pts_in_elastomer, indenter_pts_in_elastomer, reverse_z=False):
        sdf, _ = self.elastomer_sdf_sensor.get_sdf(indenter_pts_in_elastomer)
        fbar = self.step_hydrosoft_forces(sdf, indenter_pts_in_elastomer, reverse_z=reverse_z)
        self.hydrosoft_forces = fbar
        self.prev_indenter_pts_in_elastomer = indenter_pts_in_elastomer
        self.prev_sdf = sdf

        affected_marker_positions = indenter_pts_in_elastomer + fbar
        indenter_pts_height = fbar[:, :, self.elastomer_sdf_sensor.get_normal_axis()]

        num_envs, num_indenter_pts, _ = indenter_pts_in_elastomer.shape
        num_tactile_pts = tactile_pts_in_elastomer.shape[1]

        tactile = tactile_pts_in_elastomer.unsqueeze(1).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        affected = affected_marker_positions.unsqueeze(2).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        fbar_e = fbar.unsqueeze(2).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        norm_e2h = torch.norm(tactile - affected, dim=-1)
        h = indenter_pts_height.unsqueeze(-1).unsqueeze(-1).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)

        gaussian_exp = torch.exp(-self.lambda_s.unsqueeze(-1) * norm_e2h**2).unsqueeze(-1).expand(
            num_envs, num_indenter_pts, num_tactile_pts, 3)
        Mshear = torch.sum(
            self.shear_scale.unsqueeze(-1).unsqueeze(-1) * h * -1 * fbar_e * gaussian_exp * (-1 if reverse_z else 1),
            dim=1)
        return Mshear

    def get_marker_dilation(self, tactile_pts_in_elastomer, tactile_pts_in_indenter):
        tactile_pts_sdf, _ = self.indenter_sdf_sensor.get_sdf(tactile_pts_in_indenter)
        tactile_pts_height = F.relu(-tactile_pts_sdf)

        num_envs, num_tactile_pts = tactile_pts_height.shape
        dvec = tactile_pts_in_elastomer.unsqueeze(2) - tactile_pts_in_elastomer.unsqueeze(1)
        norm_dvec = torch.norm(dvec, dim=-1) ** 2

        gaussian_exp = torch.exp(-self.lambda_d.unsqueeze(-1) * norm_dvec).unsqueeze(-1).expand(
            num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3)
        Mdilate = (self.dilate_scale.unsqueeze(-1).unsqueeze(-1) * h * dvec * gaussian_exp).sum(dim=2)
        return Mdilate
