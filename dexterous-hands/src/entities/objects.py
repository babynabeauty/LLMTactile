from __future__ import annotations

import json
import os
from pathlib import Path

from eden.options.entities import BoxOptions, CylinderOptions, EntityOptions, SphereOptions
from eden.options.materials import RigidMaterialOptions

from utils import get_asset_path

PASTEL_COLORS = (
    (1.0, 0.6, 0.7),
    (0.6, 1.0, 0.7),
    (0.7, 0.6, 1.0),
    (0.5, 1.0, 1.0),
    (1.0, 0.5, 1.0),
    (1.0, 1.0, 0.5),
)

OBJECT_SET_MANIFESTS = {
    "objaverse_16": "objaverse-train-16.json",
    "objaverse_128": "objaverse-train-128.json",
    "objaverse_stress_512": "objaverse-stress-512.json",
    "ycb_21": "ycb-heldout-21.json",
}


def get_object_set_cache() -> Path:
    return Path(os.environ.get("DEXTEROUS_OBJECT_CACHE", Path(__file__).parents[2] / "data" / "object_sets"))


def load_manifest_object_set(name: str) -> list[EntityOptions]:
    manifest_name = OBJECT_SET_MANIFESTS[name]
    cache = get_object_set_cache()
    manifest_path = cache / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Object-set manifest {manifest_path} is missing. "
            "Run scripts/prepare_grasp_object_corpus.py or set DEXTEROUS_OBJECT_CACHE."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path} has unsupported schema version {manifest.get('schema_version')!r}")
    objects = manifest.get("objects", [])
    if manifest.get("object_count") != len(objects) or len(objects) < 2:
        raise ValueError(f"{manifest_path} must contain its declared multiple-object set")

    options = []
    for item in objects:
        asset_path = cache / item["asset"]
        if not asset_path.is_file():
            raise FileNotFoundError(f"Object asset {asset_path} from {manifest_path} is missing")
        options.append(
            EntityOptions(
                file=str(asset_path),
                material=RigidMaterialOptions(rho=item["density_kg_m3"], friction=1.0),
                recompute_inertia=True,
                decimate=False,
                convexify=False,
                default_armature=0.0,
            )
        )
    return options


class BallyCube(EntityOptions):
    """A beveled cube. Base size is 1m^3, so use scale to set the size."""

    file: str = get_asset_path("objects/bally.obj")
    material: RigidMaterialOptions = RigidMaterialOptions(rho=500.0, friction=1.0)
    scale: float = 0.05
    recompute_inertia: bool = True  # compute inertia from material


class DexCube(EntityOptions):
    """A colored cube with default size 5cm^3."""

    file: str = get_asset_path("objects/dex_cube.urdf")
    material: RigidMaterialOptions = RigidMaterialOptions(rho=500.0, friction=1.0)
    scale: float = 0.05
    recompute_inertia: bool = True  # compute inertia from material


class Bin(EntityOptions):
    """A bin with default size 1m^3."""

    file: str = get_asset_path("objects/bin.obj")
    scale: float = 1.0


OBJECTS_6TO8CM_16 = [
    BoxOptions(size=(0.080, 0.055, 0.064)),
    BoxOptions(size=(0.052, 0.050, 0.060)),
    BoxOptions(size=(0.050, 0.072, 0.081)),
    BoxOptions(size=(0.054, 0.065, 0.045)),
    BoxOptions(size=(0.050, 0.054, 0.055)),
    BoxOptions(size=(0.060, 0.075, 0.070)),
    CylinderOptions(height=0.072, radius=0.040),
    CylinderOptions(height=0.060, radius=0.031),
    CylinderOptions(height=0.070, radius=0.045),
    CylinderOptions(height=0.080, radius=0.032),
    CylinderOptions(height=0.062, radius=0.045),
    CylinderOptions(height=0.064, radius=0.038),
    CylinderOptions(height=0.066, radius=0.034),
    CylinderOptions(height=0.077, radius=0.044),
    BallyCube(scale=0.060),
    BallyCube(scale=0.070),
    BallyCube(scale=0.080),
]

# HORA's rotation task uses one cube and gets its variation from scale: an 8 cm block over
# `randomizeScaleList: [0.7 ... 0.86]`, so 5.6 cm to 6.9 cm. These are those nine sizes.
OBJECTS_HORA_BLOCKS_9 = [
    BoxOptions(size=(s, s, s)) for s in (0.056, 0.0576, 0.0592, 0.0608, 0.0624, 0.064, 0.0656, 0.0672, 0.0688)
]

# HORA's block scales plus round shapes, all within its 5.6-6.9 cm envelope. Size alone is
# something a proprioceptive policy can infer from how far the fingers close; shape is not, so
# a set that varies only scale cannot show what contact geometry is worth. Aspect ratios stay
# near 1, because grasp sampling never finishes on a shape a palm-up cradle cannot hold.
OBJECTS_HORA_MIXED_11 = [BoxOptions(size=(s, s, s)) for s in (0.056, 0.0592, 0.0624, 0.0656, 0.0688)] + [
    CylinderOptions(height=0.060, radius=0.031),
    CylinderOptions(height=0.066, radius=0.030),
    CylinderOptions(height=0.058, radius=0.034),
    SphereOptions(radius=0.030),
    SphereOptions(radius=0.033),
    # One flattened box, so the set is not entirely equilateral.
    BoxOptions(size=(0.068, 0.062, 0.050)),
]

# The same mix shrunk by about 20% to 4.4-5.6 cm, giving the fingertips less to hold and less
# contact area to read. Size and hand placement are coupled, so `HAND_OFFSETS["allegro"]` in
# the in_fingers_rotate config moves with this.
OBJECTS_HORA_SMALL_11 = [BoxOptions(size=(s, s, s)) for s in (0.044, 0.047, 0.050, 0.053, 0.056)] + [
    CylinderOptions(height=0.048, radius=0.025),
    CylinderOptions(height=0.053, radius=0.024),
    CylinderOptions(height=0.046, radius=0.027),
    SphereOptions(radius=0.024),
    SphereOptions(radius=0.027),
    BoxOptions(size=(0.055, 0.050, 0.040)),
]

OBJECTS_3TO4CM_8 = [
    BoxOptions(size=(0.035, 0.035, 0.035)),
    BoxOptions(size=(0.03, 0.03, 0.03)),
    BoxOptions(size=(0.04, 0.04, 0.04)),
    SphereOptions(radius=0.018),
    SphereOptions(radius=0.020),
    CylinderOptions(height=0.04, radius=0.016),
    CylinderOptions(height=0.03, radius=0.018),
    CylinderOptions(height=0.03, radius=0.020),
]


OBJECTS_4CM_16 = [
    BoxOptions(size=(0.036, 0.040, 0.042)),
    BoxOptions(size=(0.042, 0.042, 0.042)),
    BoxOptions(size=(0.035, 0.035, 0.035)),
    BoxOptions(size=(0.040, 0.040, 0.040)),
    BoxOptions(size=(0.045, 0.045, 0.045)),
    BoxOptions(size=(0.042, 0.042, 0.042)),
    BoxOptions(size=(0.034, 0.035, 0.040)),
    BoxOptions(size=(0.031, 0.038, 0.042)),
    BoxOptions(size=(0.038, 0.045, 0.041)),
    BoxOptions(size=(0.039, 0.033, 0.044)),
    BoxOptions(size=(0.041, 0.037, 0.030)),
    BoxOptions(size=(0.042, 0.034, 0.043)),
    BoxOptions(size=(0.043, 0.039, 0.035)),
    BoxOptions(size=(0.040, 0.041, 0.036)),
    BoxOptions(size=(0.044, 0.039, 0.035)),
    CylinderOptions(height=0.042, radius=0.020),
    CylinderOptions(height=0.038, radius=0.017),
]

# The 4 cm lift set scaled 2x (7-9 cm, 8x the mass), so a 1-2 N probe pulse is a nudge rather
# than a launch.
OBJECTS_8CM_16 = [
    BoxOptions(size=tuple(2.0 * v for v in variant.size))
    if isinstance(variant, BoxOptions)
    else CylinderOptions(height=2.0 * variant.height, radius=2.0 * variant.radius)
    for variant in OBJECTS_4CM_16
]

# The 4 cm set scaled 2.5x (7.8-11 cm), the size of mjlab_hand's Grasp-Allegro objects and a full
# palm for the Allegro; at rho 500 a 10 cm cube is 0.5 kg.
OBJECTS_10CM_16 = [
    BoxOptions(size=tuple(2.5 * v for v in variant.size))
    if isinstance(variant, BoxOptions)
    else CylinderOptions(height=2.5 * variant.height, radius=2.5 * variant.radius)
    for variant in OBJECTS_4CM_16
]

# Pen-sized cylinders for palm-parallel spinning. Cylinder height is local Z.
OBJECTS_PEN_3 = [
    CylinderOptions(height=0.12, radius=0.006),
    CylinderOptions(height=0.14, radius=0.007),
    CylinderOptions(height=0.16, radius=0.008),
]
