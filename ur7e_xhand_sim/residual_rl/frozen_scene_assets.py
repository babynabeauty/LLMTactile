"""Read-only asset copies for concurrent experiments; no physical asset edits."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_assets(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha(directory/name) != expected:
            raise ValueError(f'Frozen asset hash mismatch: {name}')
    for name in ('ur7e_xhand.urdf', 'platform.urdf'):
        for mesh in ET.parse(directory/name).getroot().findall('.//mesh'):
            path = Path(mesh.get('filename'))
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f'Frozen asset requires an existing absolute mesh: {path}')
    layouts = json.loads((directory/'tactile_layout.json').read_text())
    return directory, layouts


@contextmanager
def frozen_scene_assets(directory):
    if directory is None:
        yield
        return
    directory, layouts = validate_assets(directory)
    from . import v1_env
    old_build, old_platform = v1_env.build, v1_env.build_platform
    try:
        v1_env.build = lambda: (directory/'ur7e_xhand.urdf', deepcopy(layouts))
        v1_env.build_platform = lambda: (directory/'platform.urdf', {})
        yield
    finally:
        v1_env.build, v1_env.build_platform = old_build, old_platform
