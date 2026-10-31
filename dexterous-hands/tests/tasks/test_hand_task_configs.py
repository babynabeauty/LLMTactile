from pathlib import Path

import pytest
import torch
import yaml
from eden.tasks import TASK_REGISTRY
from genesis.utils.geom import euler_to_quat, transform_by_quat

from registry import _load_tasks, get_task_config
from tasks.grasp_lift.config import HAND_EULERS


@pytest.fixture(scope="module", autouse=True)
def load_tasks():
    _load_tasks()


@pytest.mark.parametrize(
    ("robot", "expected_shape_sensors"),
    [("wuji", 21), ("sharpa", 17), ("allegro", 21)],
)
def test_grasp_lift_mesh_teacher_has_palm_down_shape_configuration(robot, expected_shape_sensors):
    config = get_task_config(
        task_name="grasp_lift",
        modifiers={
            "obj": "primitives",
            "robot": robot,
            "sensors": "none",
            "stage": 1,
        },
    )
    experiment = yaml.safe_load(Path(f"conf/experiments/grasp_lift_{robot}_objaverse_m1.yaml").read_text())

    shape_sensors = [name for name, _ in config.sensors_options if name.startswith("priv_shape_distance_")]
    assert len(shape_sensors) == expected_shape_sensors
    assert "priv_shape" in experiment["runner_options"]["obs_groups"]["actor"]
    assert experiment["event_options"]["object_pulse"]["strength"] == 0.25
    assert config.event_options.obj_mass.name == "randomize_mass_scale_with_obs"

    local_normal = torch.tensor(config.scene_options.robot.metadata.palm_normal)
    euler = HAND_EULERS.get(robot, HAND_EULERS["default"])
    world_normal = transform_by_quat(local_normal[None], torch.tensor(euler_to_quat(euler))[None])
    torch.testing.assert_close(world_normal, world_normal.new_tensor([[0.0, 0.0, -1.0]]), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "task",
    ["in_hand_repose", "in_fingers_rotate", "grasp_lift"],
)
def test_project_rigid_solver_defaults(task):
    config = get_task_config(
        task_name=task,
        modifiers={
            "obj": "primitives",
            "robot": "sharpa",
            "sensors": "none",
            "stage": 1,
        },
    )

    assert config.env_options.solver == "newton"
    assert config.env_options.friction_cone == "elliptic"
    assert config.env_options.contact_resolution == "signorini"
