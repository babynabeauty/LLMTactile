"""Portable, meter-scale insertion assets for the UR7e/XHand sandbox.

All paths resolve inside this repository.  Asset conversion is an offline step;
neither USD, Isaac Sim, nor the source projects are needed at runtime.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / 'sources' / 'insertion'
INSERTION_TASKS = ('tube_insert', 'usb_insert')


def load_insertion_assets(task, *, verify=True):
    """Return movable/fixed URDF records and original geometry metadata.

    Each asset provides ``urdf_path``, ``visual_mesh_path``, local visual bounds,
    separate ``collision_bounds_m`` / ``collision_bounds_size_m`` measured from
    every URDF collision part after its local origin transform,
    center of mass, unit-mass inertia, a default mass, and recommended Genesis
    ``morph_options`` / ``material_options``. Pass the latter explicitly: the
    sockets require non-convex collision and an appropriately resolved SDF.
    The ``fixed`` records use roles ``holder`` (tube only) and ``socket``.
    Positions are intentionally not assigned here; the sandbox owns placement.
    Quaternions in the metadata follow the Genesis wxyz convention.
    """
    if task not in INSERTION_TASKS:
        raise ValueError(f'Unknown insertion task {task!r}; expected {INSERTION_TASKS}')
    manifest = json.loads((SOURCE_DIR / 'manifest.json').read_text())
    data = deepcopy(manifest['tasks'][task])
    if verify:
        for relative, expected in manifest['packaged_sha256'].items():
            path = SOURCE_DIR / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Insertion asset hash mismatch: {relative}')
    for asset in [data['movable'], *data['fixed']]:
        for key in ('urdf_path', 'visual_mesh_path', 'collision_mesh_path'):
            if key in asset:
                asset[key] = str(SOURCE_DIR / asset[key])
    data['units'] = 'm'
    data['scale'] = 1.0
    data['manifest_path'] = str(SOURCE_DIR / 'manifest.json')
    data['source'] = deepcopy(manifest['sources'][data.pop('source_key')])
    return data
