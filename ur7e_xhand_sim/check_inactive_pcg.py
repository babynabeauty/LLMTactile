"""Verify inactive PCG kernels are true no-ops; no robot motion policy."""
import json
from pathlib import Path
import numpy as np
from environment import Sandbox
from fem_diagnostics import skip_inactive_pcg_iterations

sim=Sandbox(output_dir='outputs/silicone_audit_20261024/pcg_noop',render=False,
            scenario='small_clutter',silicone=True,silicone_substeps=20,silicone_skip_idle_pcg=False)
solver=sim.scene.fem_solver
stats=skip_inactive_pcg_iterations(solver)
for _ in range(12):sim.scene.step()
assert not solver.batch_pcg_active.to_numpy().any()
before=solver.pcg_state_v.x.to_numpy().copy()
positions=sim.array(sim.silicone.get_state().pos).copy()
for _ in range(500):solver.one_pcg_iter()
result={'extra_inactive_iterations':500,
        'position_increment_bitwise_identical':bool(np.array_equal(before,solver.pcg_state_v.x.to_numpy())),
        'physical_positions_bitwise_identical':bool(np.array_equal(positions,sim.array(sim.silicone.get_state().pos))),
        'pcg_stats':stats,'existing_pcg_threshold':solver._pcg_threshold,
        'existing_max_iterations':solver._n_pcg_iterations}
Path('outputs/silicone_audit_20261024/pcg_noop_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result['position_increment_bitwise_identical'] and result['physical_positions_bitwise_identical']
sim.scene.destroy()
