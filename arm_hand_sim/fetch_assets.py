"""Verify vendored assets, or restore missing files from the pinned upstream commit."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    manifest = json.loads((HERE / 'asset_source.json').read_text())
    records = next(v for v in manifest.values() if isinstance(v, list) and v and isinstance(v[0], dict))
    for item in records:
        path = ROOT / item['path']
        if not path.exists():
            relative = path.relative_to(HERE / 'vendor').as_posix()
            url = f"https://raw.githubusercontent.com/{manifest['repo']}/{manifest['commit']}/{relative}"
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise RuntimeError(f'Download checksum mismatch: {relative}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Local checksum mismatch (not overwritten): {path}')
    print(f'OK: {len(records)} pinned asset files verified')


if __name__ == '__main__':
    main()
