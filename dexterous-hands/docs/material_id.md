# Material discrimination from one fingertip slide

A single XHand1 index fingertip presses a fixed plate at ~2.5 N and drags across it
for 1.6 s. A per-link `ContactForce` sensor and a fingertip `TemperatureGrid` sensor
are logged, and an SVM classifies which of eight materials the plate is. No RL: the
motion is scripted, the trial is open loop, and the whole dataset is one batched
Genesis build.

| script | what it does |
| --- | --- |
| `scripts/material_id/materials.py` | The eight materials: friction against a silicone fingertip, plus handbook conductivity, density and specific heat. |
| `scripts/material_id/collect_slides.py` | Runs the slides and writes `data/material_id/slides.npz`. `--probe` prints pose, contact and cooling diagnostics. |
| `scripts/material_id/features.py` | Per-trial features, force and temperature kept strictly separate. |
| `scripts/material_id/fit_svm.py` | The force / temperature / both ablation, the cue-independence check, and the figure. |

```
python scripts/material_id/collect_slides.py --trials 64 --out data/material_id/slides.npz
python scripts/material_id/fit_svm.py --data data/material_id/slides.npz
```

512 trials take a few minutes on one GPU.

## The scene

Every material is its own plate entity, so it carries its own friction and its own
`TemperatureProperties`. All eight plates exist in every env; the env index picks
which one the hand is placed over, so a single build sweeps all of them. Plates are
25 mm wide against a 20 mm finger pitch and there is no floor, so the mid, ring and
pinky fingers hang in open air: the index fingertip is the only link that ever
touches anything.

The hand is placed from a measured fingertip pose, not from URDF arithmetic. The
root DOF frame is offset from the palm origin, and the per-trial pitch and flexion
jitter move the fingertip by more than the press depth, so the script reads the
fingertip AABB at a height clear of the plates and derives each env's start pose
from it.

The press is force-controlled with viscous damping on the same DOF. A stiff position
servo commanded below the surface does not rest on the plate, it hammers it: the
contact collapses into a 9 N spike every third step.

The solver runs an elliptic friction cone with Signorini contact resolution and a
0.02 s constraint time constant, which holds the fingertip in contact for 100% of the
slide at ~0.33 mm penetration. The cone is load-bearing for the result, not a
preference; see the traps section below.

## Materials

Friction is for a dry silicone fingertip on that surface; conductivity is the bulk
handbook value. `k_eff` is the series conductivity of skin and plate, which is what
the contact actually transports heat at.

| material | mu | k (W/m·K) | k_eff | measured cooling over the trial |
| --- | --- | --- | --- | --- |
| aluminum | 0.45 | 205 | 0.499 | -3.02 °C |
| steel | 0.35 | 16 | 0.492 | -3.03 °C |
| glass | 0.70 | 1.05 | 0.404 | -2.63 °C |
| ceramic | 0.55 | 1.50 | 0.429 | -2.69 °C |
| oak | 0.55 | 0.16 | 0.195 | -1.48 °C |
| abs | 0.40 | 0.17 | 0.202 | -1.46 °C |
| silicone | 1.20 | 0.20 | 0.222 | -1.54 °C |
| foam | 0.60 | 0.035 | 0.061 | -0.61 °C |

The set is chosen so neither cue is sufficient. A silicone fingertip is the
bottleneck in the series resistance, so every metal saturates at twice the skin's
conductivity and aluminum and steel are thermally identical; oak, ABS and silicone
sit within 15% of each other. Those same groups are far apart in friction.

## Sensing

The fingertip sits at 35 °C (a warm tactile board), plates and air at 22 °C, so
contact cools the tip at a rate set by `k_eff`. Convection is off and the emissivity
is low, so free-air drift is ~0.1 °C/s against a contact signal of 0.5-3 °C.

The force channel carries the noise model from `conf/sensor/tactile_params.yaml`
(`link_force`: 0.08 N white noise per axis, 0.01 N/step random walk, 5 mN
quantization); the temperature channel gets 0.05 °C noise, 0.1 °C quantization and a
0.15 s RC lag. The random walk is what limits the friction estimate: per-trial
`mu_hat` lands within 0.08 of the set value, so materials closer than that in
friction cannot be separated by force alone.

Feature extraction tares the force against the pre-touch window, as any real tactile
pipeline does. That is worth ~5 accuracy points.

## Results

512 trials, 64 per material, 5-fold cross-validated RBF SVM. Chance is 12.5%.

| channel | features | accuracy |
| --- | --- | --- |
| force | 8 | 42.2% |
| temperature | 7 | 34.6% |
| force + temperature | 15 | 73.6% |

The confusion structure is the result, not the headline number. Force alone nails
silicone (mu 1.20, far from everything) and confuses the metals with the plastics.
Temperature alone nails foam and collapses the metals into one block and the
plastics into another. Together they cover for each other, and what survives is the
pair that is close in both: aluminum against steel.

Figure: `figures/material_id/material_id.pdf`.

## Two simulator traps this experiment hit

**`TemperatureGrid` must be built with the link unrotated.** The sensor derives its
voxel volume by transforming the link's world AABB corners into the link frame,
which is only exact when the link is axis-aligned. Building the hand rotated
collapses that volume and inflates every heat-transfer rate by orders of magnitude:
the fingertip radiated from 35 °C to ambient in 0.3 s with nothing touching it. Build
at identity and apply the orientation through the root DOF afterwards.

**The default pyramidal friction cone couples friction into contact penetration.**
Under it, penetration comes out as a monotone function of the geom's friction: 0.03
mm on steel against 1.8 mm on silicone at the same 2.5 N press. The contact-heat
model sizes its conducting patch from penetration, so the temperature channel became
a disguised friction sensor. With `friction_cone=gs.friction_cone.elliptic` every
material penetrates identically and the cooling curve tracks `k_eff` alone. The
`fit_svm.py` cue-independence check guards this: a temperature-only classifier is at
chance inside each thermally matched group, whose members differ in friction by up to
3x.
