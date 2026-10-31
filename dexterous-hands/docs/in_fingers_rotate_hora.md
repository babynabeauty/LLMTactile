# HORA in-fingers rotation on the Allegro hand

`in_fingers_rotate` with `--robot=allegro` is a port of HORA's `AllegroHandHora`
task ([Qi et al. 2022](https://arxiv.org/abs/2210.04887),
[HaozhiQi/hora](https://github.com/HaozhiQi/hora)). This page records the numbers
that came from HORA, the ones that had to be re-derived, and why the deviations
are where they are.

## Where things are

| | |
| --- | --- |
| Task config | `src/tasks/in_fingers_rotate/config.py`, `custom_terms.py` |
| Grasp sampling | `conf/sample_grasps/in_fingers_rotate_{allegro,xhand1}.yaml` |
| Student configs | `conf/experiments/in_fingers_rotate_distill_rnn.yaml`, `..._rl_proprio_rnn.yaml` |
| Reward-sweep configs | Historical sweep results below; active `hold` and `grip` recipes remain, while obsolete one-off variants were pruned |
| Checkpoints | `checkpoints/in_fingers_rotate_manifest.yaml` — paths, load flags and measured scores |
| Videos | `docs/media/` |
| Evaluation | `scripts/eval_rotate.py` (`eval_policy.py` is hard-wired to `goal_rot` and will not run this task) |
| Diagnostics | `scripts/analysis/` — PD step response, grasp geometry, cage-pose search, run peeking |
| Sampling until full | `scripts/sample_grasps_until_full.sh` — reruns the sampler, which stops on a Genesis NaN |

## Scene

The Allegro base sits 0.15 m directly below the object with the palm facing up and
the fingers horizontal, so the object rests in the fingers just above the palm.
HORA spawns the hand at z = 0.5 with
`Quat.from_axis_angle(y, -pi/2) * Quat.from_axis_angle(x, pi/2)` and the object at
z = 0.65. That rotation is `HAND_EULERS["allegro"] = (90, -90, 0)` in extrinsic XYZ
degrees, which maps base-link `+z` (the direction the fingers extend) to world `-y`
and base-link `+x` (the palm normal) to world `+z`.

The target axis is world `-z`, matching HORA's `rot_axis_buf[:, -1] = -1`. The
existing `RotationAxisCommand` with `axis_mode="z_only"` already emits that sign.

## Control

| | HORA | here |
| --- | --- | --- |
| sim rate | 120 Hz | 200 Hz |
| control rate | 20 Hz (`controlFrequencyInv: 6`) | 20 Hz (`DECIMATION = 10`) |
| action | `target += a / 24`, clamped to joint limits | `ReferenceSource.ACCUMULATE`, `scale = 1/24` |
| gains | `pgain 3`, `dgain 0.1` | `default_dofs_kp 3.0`, `default_dofs_kd 0.6` |
| torque limit | `clip(tau, -0.5, 0.5)` | `default_dofs_force_limits 0.5` |
| armature | none (Isaac Gym) | `default_dofs_armature 0.03` |

Three of these carry most of the behaviour.

The action **integrates**: it is a per-step increment onto a persistent target, not
an absolute pose. At 1/24 rad per step and 20 Hz a joint cannot travel faster than
0.83 rad/s whatever the policy outputs, which is what makes the resulting motion a
gait rather than a flick. `ReferenceSource.ACCUMULATE` implements exactly HORA's
recurrence, so the control rate has to match too — the increment is per control
step, and at 40 Hz the same number would mean twice the joint speed.

The **gains are compliant**, not a stiff position servo. The registered
`AllegroHand` defaults (kp 120, kd 10) can command 100x the real Allegro motor's
0.5 Nm, which turns every contact into an impulse. HORA's kp 3 with the torque
clipped at 0.5 Nm means a 0.17 rad tracking error already saturates the motor: the
hand yields when it pushes on the object. The `allegro` entry in `HAND_GAINS` overrides the
registry per task rather than changing it, since `in_hand_repose` and the sysid
configs share that entity.

**`kd` is not HORA's, and the armature is why.** Damping ratio is
`kd / (2*sqrt(kp*I))`. Eden defaults every DOF to `default_armature = 0.1` kg m^2 — a
stabilization figure orders above an Allegro finger joint's real reflected inertia —
while Isaac Gym adds none. The same kd that is near-critical in HORA therefore leaves
`zeta = 0.09` here, and the joints ring at 5.5 rad/s.

Matching the real inertia is not an option: below about 0.02 the contact solve
diverges into NaN constraint forces against a 50 g block, and halving `sim_dt` does
not rescue it — the mass ratio is what the default regularizes. So the armature stays
at the largest value that still damps well, and `kd` is set to critical damping for
it, `2*sqrt(3*0.03) = 0.6`.

Measured per joint on a 0.3 rad step with the torque limit in place:

| config | overshoot, 15 real joints | ringing | contact-stable |
| --- | --- | --- | --- |
| armature 0.1, kd 0.1 (literal port) | 85% mean, 13 of 16 never settle | yes | yes |
| armature 0.001, kd 0.1 | 0.9% mean, settles in 77 ms | no | **no — NaN** |
| **armature 0.03, kd 0.6** | **under 3 mrad** | **no** | **yes** |

The 16th joint is `thumb_bend0` and it is not a controller artifact: its lower limit
is 0.263, so a step test starting from the zero pose begins outside its own range and
the limit constraint throws it. Percentage overshoot aggregated over all 16 joints is
dominated by that one and is not worth reading.

The failure this causes is silent, which is why it is worth checking before anything
else. The hand still articulates; it just cannot hold anything. Grasps sag out of the
fingers over a second or two, and the grasp sampler's stability filter then rejects
almost everything — which reads as a bad seed pose rather than a controller problem.

The grasp cache is dynamics-dependent, so it has to be regenerated after any change to
gains, armature or force limits.

## Reward

HORA's five terms at their published weights, from `configs/task/AllegroHandHora.yaml`:

| term | weight | quantity |
| --- | --- | --- |
| `rotation_reward` | `+1.0` | `clip(omega . axis, -0.5, 0.5)` |
| `obj_linvel_penalty` | `-0.3` | L1 norm of finite-difference object velocity |
| `pose_diff_penalty` | `-0.3` | `sum((q - q_grasp)^2)` |
| `torque_penalty` | `-0.1` | `sum(tau^2)` |
| `work_penalty` | `-2.0` | `(sum(tau * qdot))^2` |

**The rotation reward saturates.** Above 0.5 rad/s — about a twelfth of a
revolution per second — spinning faster earns nothing more while the velocity,
torque and work penalties all keep growing. A steady slow gait is the optimum of
this reward, not a compromise imposed on top of it, which is why the clip is the
term to reach for if the gait comes out too fast or too slow.

**The pose-diff penalty is what makes the motion repeat.** Fingers pay for every
radian they hold away from the grasp they started the episode in, so a stroke that
returns is cheaper than one that drifts. It has to be measured against the grasp
sampled for *this* episode, which is what `GraspPoseDiffPenalty` does by reading
`LoadGraspPose`'s per-env cache. The shared `shared_terms.pose_diff_penalty`
snapshots one pose on its first call and keeps it for the whole run, which is a
different quantity once grasps vary per episode.

Two terms are not HORA's:

- `joint_limits_penalty` at `-1.0`. Our joint limits are the URDF's rather than
  HORA's tightened ones, so the integrating target can park a joint against a limit
  and sit there.
- `is_dropped` at `-10.0`. HORA has no drop penalty because its termination fires
  5 mm below the start height, before a falling object earns anything. At the 2 cm
  threshold used here (Genesis's contact resolution needs the room) a spin-and-drop
  banks a few steps of rotation reward first, and this makes that trade losing. It
  is deliberately small: a large drop penalty buys the opposite failure, a policy
  that holds the object still because moving it risks the penalty.

The drop termination is `DroppedBelowGraspStart`, measured against the height the
episode's own grasp placed the object at. HORA can use an absolute 0.645 because every
grasp holds the same block in the same place; here the object set spans 5-8 cm and the
sampled start heights spread over about 2 cm, so a fixed threshold leaves the tallest
objects almost no margin and gives the shortest twice as much.

## Observations

The teacher is privileged but sensorless. Its groups are `goal`, `priv_proprio`,
`priv_obj_state` and `priv_obj_props` — joint position, joint velocity, its own last
action, and privileged object state. It has no tactile group, matching HORA, whose
policy sees three frames of joint positions and targets plus
privileged object position, scale, mass, centre of mass and friction.

Fingertip contact forces live in their own `priv_contact` group used only by the
critic. They are a tactile signal in all but name, and the point of the teacher is
to be the target a proprioception-only student distils from.

## Object size

HORA's block is 8 cm at `baseObjScale: 0.8`, randomized over
`randomizeScaleList: [0.7 ... 0.86]` — a 5.6-6.9 cm object. `OBJECT_SETS` gives the
Allegro `OBJECTS_HORA_MIXED_11`: five of HORA's block scales, three cylinders, two
spheres and one flattened box, all inside that envelope. The smaller hands keep the
4 cm set their grasp files were built against.

Shape variation is there deliberately. Scale on its own is something a proprioceptive
policy can read off how far the fingers close, so a scale-only set cannot show what
contact geometry is worth — which is the whole point of comparing a tactile student
against a blind one. Aspect ratios stay near 1: a palm-up cradle cannot hold a tall thin
cylinder at all, and the sampler will not finish while any variant is unfillable.

This is worth stating plainly because it is the whole task: HORA's canonical pose
closes the Allegro fingertips around a 6.4 cm block, and on a 4 cm object the tips
finish 2-3 cm clear of every face. Measured at the seed pose, an average of 0.6 of 4
fingertips came within 15 mm of a 4 cm object against 2.7 of 4 for a 6-8 cm one, and
no sampled grasp on the small set passed the sampler at all.

## Grasp cache

HORA collects 50k grasps per object scale by perturbing a canonical pose by
+-0.25 rad and keeping the ones where all four fingertips are within 0.1 m of the
object, at least two are in contact, and the object survives a **full 400-step
zero-action episode**. `conf/sample_grasps/in_fingers_rotate_allegro.yaml` mirrors
that with the surface-distance/penetration/stability checks the sampler provides, at
128 grasps per object variant.

The quota is per variant, so without a stopping rule the hardest shape decides when the
run ends. `--stall_iters` / `--min_grasps` let it stop once no variant has gained a grasp
for a while and every variant has cleared a floor, and it prints which variants came up
short rather than passing a thin cache off as a full one.

The duration of that last check is what makes the cache usable rather than nominal.
Under compliant gains a grasp that looks closed can sag out of the hand over a
second or two, and `stability_steps: 20` (0.1 s) accepts those: with the short filter,
every env dropped inside one episode with no action applied at all. At 1000 sim steps
— 5 s of holding — roughly 2% of sampled poses survive, and those are grasps a policy
can actually start from.

The seed pose is HORA's `canonical_pose` from `hora/tasks/allegro_hand_grasp.py`,
grouped as `[index, thumb, middle, ring]` x `[roll, bend0, bend1, bend2]`. HORA
clamps the pose to the joint limits after perturbing it, so its raw array need not be
feasible and the joint order cannot be recovered from the limits alone. This grouping
is the one that measures as a grasp in sim, closing 2.7 of 4 fingertips onto the
object against 0.6 for the alternatives.

## Judging a run

`Metrics/turns` is net revolutions about the target axis completed while the object
is still in the fingers, from `AxisTurnsMetric`. Reward alone does not separate a
continuous gait from a half-turn wobble repeated forever, and it does not notice an
object that left the hand and kept spinning on the way down; turns do both, because
the increment is gated on the object staying within `hold_radius` of where it
started.
