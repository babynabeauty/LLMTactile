Astra + frozen seed1 tactile PPO, cuboid125g mu0.5.
Source: direct_teacher_cuboid_tripod_clearance_20261031_v2/attempt01, 50g mu0.8.
Recorded three-finger closure/preload, live Astra registration and skill decisions. All three tactile support gates active.
PPO gains only thumb/index; middle stays nominal. New trajectory transfer, not retrained.
Run: bash experiments/astra_ppo_cuboid_tripod_125g_mu05/run.sh

Selected raised-contact v2 demonstration: minimum recorded middle clearance13.70mm. Original package demo retained in demo_original_20261030_v1.

Selectable modes (default: ppo):
  bash experiments/astra_ppo_cuboid_tripod_125g_mu05/run.sh --mode ppo
  bash experiments/astra_ppo_cuboid_tripod_125g_mu05/run.sh --mode no-ppo
Both use raised-contact v2 demo, cuboid125g, mu0.5, 300Hz physics/60Hz samples,
noslip5 deterministic solver, same cameras and table protection (default5N).
no-ppo loads no PPO checkpoint; retains Astra initial gain selection, recorded
three-finger close/preload, tactile support gates and original rule controller.
It is NOT the learned no-tactile PPO model and NOT an added fixed-gain controller.
The demonstration prompt requests initial gain2 in BOTH modes. With no PPO,
this gain remains in effect as in the ordinary skill executor.
Logs use astra_ppo_* versus astra_no_ppo_* labels; mode_readback.json records settings.
Run sequentially: both modes use port8775. No inference is launched by local preflight.
