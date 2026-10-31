"""Immutable runtime identity for the fresh single-phase teacher campaign."""
import importlib.metadata as metadata
import json
from pathlib import Path

GENESIS_COMMIT = 'a68b97b9363aa0be9ccb0c8291b021f3afdd807d'
PROTOCOL = 'fresh_v141_single_phase_v1'

def runtime_identity(require_pin=False):
    import genesis
    dist = metadata.distribution('genesis-world')
    source = json.loads(dist.read_text('direct_url.json') or '{}')
    result = {'version': dist.version, 'commit':source.get('vcs_info',{}).get('commit_id'),
              'import_path':str(Path(genesis.__file__).resolve())}
    if require_pin:
        assert result['version']=='1.4.1' and result['commit']==GENESIS_COMMIT, result
        assert str(Path(dist.locate_file('genesis')).resolve()) == str(Path(genesis.__file__).resolve().parent), result
    return result
