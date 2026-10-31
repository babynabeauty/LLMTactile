"""Shared, physics-clock object wrench pulses for manipulation teachers."""

import math
import torch
import genesis as gs
from eden.managers import EVENT_TERM_REGISTRY, OBSERVATION_TERM_REGISTRY, EventTerm


class WrenchPulseState:
    def __init__(
        self,
        num_envs,
        device,
        strength=0.25,
        clean_fraction=0.25,
        profile="weight_scaled",
        probe_wrench_scale=1.0,
        linear_force_scale=1.0,
    ):
        if strength not in (0.0, 0.25, 0.5, 1.0) or not 0 <= clean_fraction <= 1:
            raise ValueError("Invalid curriculum strength or pulse-free episode fraction")
        if not 0 < probe_wrench_scale <= 1:
            raise ValueError("probe_wrench_scale must be in (0, 1]")
        if not 0 <= linear_force_scale <= 1:
            raise ValueError("linear_force_scale must be in [0, 1]")
        self.probe_wrench_scale = probe_wrench_scale
        self.linear_force_scale = linear_force_scale
        self.strength, self.clean_fraction = strength, clean_fraction
        if profile not in ("weight_scaled", "probe"):
            raise ValueError("Unknown object perturbation profile")
        self.profile = profile
        self.duration = torch.zeros(num_envs, device=device)
        self.elapsed = torch.zeros_like(self.duration)
        self.age = torch.zeros_like(self.duration)
        self.next_allowed = torch.ones_like(self.duration)
        self.enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.peak_force = torch.zeros(num_envs, 3, device=device)
        self.peak_torque = torch.zeros_like(self.peak_force)
        self.lever = torch.zeros_like(self.peak_force)
        self.force = torch.zeros_like(self.peak_force)
        self.torque = torch.zeros_like(self.peak_force)
        self.active = torch.zeros(num_envs, 1, device=device)
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids):
        self.enabled[ids] = torch.rand_like(self.age[ids]) >= self.clean_fraction
        for value in (
            self.age,
            self.duration,
            self.elapsed,
            self.force,
            self.torque,
            self.peak_force,
            self.peak_torque,
            self.lever,
            self.active,
        ):
            value[ids] = 0
        self.next_allowed[ids] = 1.0

    def schedule(self, ids, mass, radius):
        selected = torch.zeros_like(self.enabled)
        selected[ids] = True
        selected &= self.enabled & (self.age >= self.next_allowed) & (self.elapsed >= self.duration)
        ids = selected
        n = int(ids.sum())
        if not n:
            return

        def uniform(low, high):
            return torch.empty(n, device=self.age.device).uniform_(low, high)

        def direction():
            x = torch.randn(n, 3, device=self.age.device)
            return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        mode = torch.randint(0, 3, (n,), device=self.age.device)
        self.mode[ids] = mode
        self.duration[ids] = uniform(0.1, 0.5)
        self.elapsed[ids] = 0
        self.next_allowed[ids] = self.age[ids] + self.duration[ids] + uniform(2, 4)
        weight = mass[ids] * 9.81
        force_direction = direction()
        if self.profile == "probe":
            # Curriculum stages have a physical 1 N floor: 1–2, 1–3, 1–5 N.
            magnitude = uniform(1.0, 1.0 + 4.0 * self.strength) * (self.strength > 0) * self.probe_wrench_scale
        else:
            magnitude = (weight * uniform(0.25, 1)).clamp_max(2) * self.strength
        force = force_direction * magnitude[:, None] * self.linear_force_scale
        force[mode == 2] = 0
        lever = direction() * (radius[ids] * 0.5 * uniform(0, 1))[:, None]
        lever[mode != 1] = 0
        torque_direction = direction()
        if self.profile == "probe":
            magnitude = uniform(0.02, 0.02 + 0.06 * self.strength) * (self.strength > 0) * self.probe_wrench_scale
        else:
            magnitude = (weight * radius[ids] * uniform(0.1, 0.5)).clamp_max(0.02) * self.strength
        torque = torque_direction * magnitude[:, None]
        torque[mode != 2] = 0
        self.peak_force[ids] = force
        self.lever[ids] = lever
        self.peak_torque[ids] = torque + torch.linalg.cross(lever, force)

    def step(self, dt, first_substep):
        if first_substep:
            self.active.zero_()
        # Exact average of half-sine over this physics step. Also handles a pulse
        # ending between two steps without adding a full step's extra impulse.
        duration = self.duration.clamp_min(1e-12)
        start = self.elapsed.minimum(duration)
        end = (self.elapsed + dt).minimum(duration)
        envelope = (
            duration / (math.pi * dt) * (torch.cos(math.pi * start / duration) - torch.cos(math.pi * end / duration))
        )
        envelope *= self.elapsed < self.duration
        self.force.copy_(self.peak_force * envelope[:, None])
        self.torque.copy_(self.peak_torque * envelope[:, None])
        nonzero = (self.force.abs().sum(-1) + self.torque.abs().sum(-1)) > 0
        self.active.copy_(torch.maximum(self.active, nonzero.float()[:, None]))
        self.elapsed += dt
        self.age += dt

    def arrow_data(self, positions, env_ids):
        """Actual last applied wrench, shared by physics and optional recording UI."""
        result = []
        for i in env_ids:
            force, torque = self.force[i], self.torque[i]
            if bool(force.any() | torque.any()):
                result.append(
                    dict(
                        env_id=i,
                        point=positions[i] + self.lever[i],
                        force=force,
                        torque=torque,
                        remaining=float((self.duration[i] - self.elapsed[i]).clamp_min(0)),
                        duration=float(self.duration[i]),
                    )
                )
        return result


@EVENT_TERM_REGISTRY.register()
class ObjectWrenchPulse(EventTerm):
    entity_name: str = "obj"
    strength: float = 0.25
    clean_fraction: float = 0.25
    profile: str = "probe"
    probe_wrench_scale: float = 1.0
    linear_force_scale: float = 1.0
    draw_envs: list[int] = []
    # Lift probes the carry: with this set, pulses start only once the object is this far above
    # its reset height. Unset, a 1 N pulse launches a free 30 g primitive at metres per second.
    only_lifted_above_m: float | None = None

    def build(self):
        from eden.managers.event_manager import EventManager

        if not hasattr(EventManager, "pre_physics_step"):
            raise RuntimeError("Run python scripts/patch_eden.py after uv sync")
        if self._env.env_options.sim_substeps != 1:
            raise ValueError("Object wrench pulses require sim_substeps=1")
        if any(i < 0 or i >= self.num_envs for i in self.draw_envs):
            raise ValueError("draw_envs contains an invalid environment index")
        self.obj = self._env.entities[self.entity_name]
        self.state = WrenchPulseState(
            self.num_envs,
            self.device,
            self.strength,
            self.clean_fraction,
            self.profile,
            self.probe_wrench_scale,
            self.linear_force_scale,
        )
        # These tasks use single rigid objects, including grouped primitive variants.
        names, _ = self.obj.find_named_links_idx_local("*")
        if len(names) != 1:
            raise ValueError("ObjectWrenchPulse requires a single-link object")
        self.links_idx = torch.tensor([self.obj._entity.get_link(names[0]).idx], device=self.device, dtype=torch.int32)
        aabb = self.obj.get_AABB()
        extents = aabb[..., 1, :] - aabb[..., 0, :]
        self.radius = (extents.norm(dim=-1) / 2).expand(self.num_envs).clone()
        self._arrows = []
        self._reset_z = self.obj.get_AABB()[:, 0, 2].clone()
        self._env.object_wrench = self

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        self.state.reset(ids)
        self._reset_z[ids] = self.obj.get_AABB()[ids, 0, 2]
        if self.draw_envs:
            self.draw()

    def compute(self, envs_idx=None):
        mass = torch.as_tensor(self.obj.get_mass(), device=self.device).expand(self.num_envs)
        ids = slice(None) if envs_idx is None else envs_idx
        if self.only_lifted_above_m is not None:
            lifted = self.obj.get_AABB()[:, 0, 2] - self._reset_z > self.only_lifted_above_m
            selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            selected[ids] = True
            ids = selected & lifted
        self.state.schedule(ids, mass, self.radius)

    def pre_physics_step(self, dt, first_substep):
        capture = getattr(self._env, "physics_state_callback", None)
        if capture is not None:
            capture(self._env)
        self.state.step(dt, first_substep)
        self._env.rigid_solver.apply_links_external_wrench(
            force=self.state.force[:, None, :],
            torque=self.state.torque[:, None, :],
            links_idx=self.links_idx,
            ref=gs.link_ref_frame.link_COM,
            local=False,
        )
        if (
            self.draw_envs
            and not self._env.training
            and self._env._sim_step_counter % self._env.env_options.decimation == 0
        ):
            self.draw()

    def draw(self):
        for arrow in self._arrows:
            self._env.scene.clear_debug_object(arrow)
        self._arrows.clear()
        for data in self.state.arrow_data(
            self._env.rigid_solver.get_links_root_COM(self.links_idx)[:, 0], self.draw_envs
        ):
            for key, scale, color in [("force", 0.8, (1, 0.2, 0.1, 1)), ("torque", 30.0, (0.2, 0.4, 1, 1))]:
                magnitude = float(data[key].norm())
                if magnitude > 1e-10:
                    # Keep small safe pulses visible. Direction and label are exact;
                    # displayed length is clipped to 6–12 cm for legibility.
                    vector = data[key] / magnitude * min(0.12, max(0.06, scale * magnitude))
                    # Scene's convenience method lacks env_idx in Genesis 1.4.
                    # The context API isolates arrows to the recorded environment.
                    scene = self._env.scene
                    context = scene._visualizer.context
                    rendered = list(context.rendered_envs_idx)
                    if data["env_id"] not in rendered:
                        continue
                    point = data["point"].cpu().numpy() + scene.envs_offset[data["env_id"]]
                    with scene._visualizer.viewer_lock:
                        arrow = context.draw_debug_arrow(
                            pos=point,
                            vec=vector.cpu().numpy(),
                            radius=0.002,
                            color=color,
                            env_idx=rendered.index(data["env_id"]),
                        )
                    # Sub-epsilon vectors legitimately produce no mesh.
                    if arrow is not None:
                        self._arrows.append(arrow)
        # Magnitude and remaining duration are exported in arrow_data for video overlays.


@OBSERVATION_TERM_REGISTRY.register()
def perturbation_active(env):
    pulse = getattr(env, "object_wrench", None)
    if pulse is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return pulse.state.active


@EVENT_TERM_REGISTRY.register()
class RandomizeObjectMass(EventTerm):
    """Genesis 1.4 mass shift, relative to the built object's mass on every reset."""

    entity_name: str = "obj"
    obs_name: str = "obj_mass"
    links_name: str = "*"
    mass_shift_range: tuple[float, float] = (0.0, 0.1)

    def build(self):
        self.obj = self._env.entities[self.entity_name]
        self.base_mass = torch.as_tensor(self.obj.get_mass(), device=self.device).expand(self.num_envs).clone()

    def compute(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        mass = self.base_mass[ids] + torch.empty_like(self.base_mass[ids]).uniform_(*self.mass_shift_range)
        self.obj._entity.set_mass(mass, envs_idx=ids)
        self._env.observation_manager.get_term(self.obs_name)._cached[ids] = mass[:, None]
