#!/usr/bin/env python
"""Reduce the raw benchmark logs into the JSON the project page's charts fetch.

Reads dexterous-hands/data/sensor_benchmarks and writes docs/static/data/. Repeat runs at
one configuration are averaged and their spread kept, so a chart can show both. Only the
RTX 5090 sweep is emitted; the A6000 logs stay in the repository as the earlier run.
"""

import argparse
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "dexterous-hands/data/sensor_benchmarks"
OUT = ROOT / "docs/static/data"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GPU = "5090"
GPU_LABEL = "NVIDIA RTX 5090"

# Sensor ids as the benchmark harness writes them, in the order the paper's legend uses.
# Labels and colours are the ones scripts/plot/plot_benchmark_sensor.py uses, so the page and
# the paper's figure read as one plot.
SENSORS = {
    "no_sensors": ("No Sensor", "#767D8F"),
    "surface_distance_probe": ("Surface Distance Probe", "#F3AE54"),
    "contact_depth_probe": ("Contact Depth Probe", "#497FDB"),
    "kinematic_taxel": ("Kinematic Taxel", "#D65974"),
    "proximity_taxel": ("Proximity Taxel", "#57BF6A"),
    "elastomer_taxel": ("Elastomer Taxel", "#A96CD1"),
}


def coerce(value):
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def records(gpu, *stems):
    """Every ` | `-separated key=value line from one GPU's logs matching these filename stems."""
    rows = []
    for stem in stems:
        for path in sorted((BENCH / gpu).glob(f"{stem}*.txt")):
            for line in path.read_text().splitlines():
                line = ANSI.sub("", line).strip()
                if not line:
                    continue
                row = dict(
                    item.strip().split("=", 1) for item in line.split("|") if "=" in item
                )
                rows.append({k: coerce(v) for k, v in row.items()})
    return rows


def series(rows, x_field, keep, variant=lambda r: "one"):
    """Mean fps and memory per (sensor, variant, x) over the rows `keep` accepts."""
    buckets = {}
    for row in rows:
        if x_field not in row or not keep(row):
            continue
        buckets.setdefault((row["sensor"], variant(row), row[x_field]), []).append(row)
    grouped = {}
    for (sensor, var, x), group in buckets.items():
        fps = [r["runtime_fps"] for r in group]
        grouped.setdefault((sensor, var), []).append(
            {
                "x": x,
                "fps": round(statistics.mean(fps)),
                "fpsSd": round(statistics.stdev(fps)) if len(fps) > 1 else 0,
                "memMb": round(statistics.mean(r["peak_mem_mb"] for r in group)),
                "runs": len(group),
            }
        )
    out = []
    for sensor, (label, color) in SENSORS.items():
        for var in ("one", "five", "noisy"):
            points = grouped.get((sensor, var))
            if points:
                out.append({
                    "id": sensor, "variant": var, "label": label, "color": color,
                    "points": sorted(points, key=lambda p: p["x"]),
                })
    return out


def sensor_variant(row):
    if row.get("noise", False):
        return "noisy"
    return "five" if row.get("n_sensors") == 5 else "one"


def benchmark():
    envs = records(GPU, "no_sensor_scene_speed", "tactile_sensor_speed_full", "noised_tactile_sensor_speed")
    return {
        "gpu": GPU_LABEL,
        "variants": [
            {"id": "one", "label": "1 sensor", "hatch": None},
            {"id": "five", "label": "5 sensors", "hatch": "diagonal"},
            {"id": "noisy", "label": "1 sensor, with noise", "hatch": "dots"},
        ],
        "panels": [
            {
                "id": "envs",
                "title": "By parallel environments",
                "sub": "100 probes, 500 point cloud samples",
                "x": "Number of environments",
                "series": series(envs, "batch_size", lambda r: True, sensor_variant),
            },
            {
                "id": "probes",
                "title": "By number of probes",
                "sub": "1024 environments, 500 point cloud samples",
                "x": "Number of probes",
                "series": series(
                    records(GPU, "no_sensor_scene_speed", "probes_sensor_speed"),
                    "n_probes", lambda r: r.get("batch_size") == 1024, sensor_variant,
                ),
            },
            {
                "id": "points",
                "title": "By point cloud size",
                "sub": "1024 environments, 100 probes",
                "x": "Point cloud sample points",
                "series": series(
                    records(GPU, "no_sensor_scene_speed", "pointcloud_sensor_speed"),
                    "n_sample_points", lambda r: r.get("batch_size") == 1024, sensor_variant,
                ),
            },
        ],
    }


def read_csv(name):
    with (BENCH / name).open() as handle:
        return list(csv.DictReader(handle))


def ours(gpu, stems, keep):
    by_env = {}
    for row in records(gpu, *stems):
        if keep(row):
            by_env.setdefault(row["batch_size"], []).append(row)
    return {
        envs: {
            "fps": round(statistics.mean(r["runtime_fps"] for r in group)),
            "memMb": round(statistics.mean(r["peak_mem_mb"] for r in group)),
        }
        for envs, group in by_env.items()
    }


def comparison():
    """Ours against each prior simulator, on the numbers their own papers report."""
    # TacSL: its Table II, against our KinematicTaxel at the same two taxel resolutions.
    tacsl = {"100": {}, "10000": {}}
    for row in records("5090", "tacsl_comparison_speed"):
        key = str(row.get("n_probes"))
        if row["sensor"] != "kinematic_taxel" or key not in tacsl:
            continue
        tacsl[key].setdefault(row["batch_size"], {"envs": row["batch_size"], "ours": []})["ours"].append(
            row["runtime_fps"]
        )
    for resolution in tacsl:
        for point in tacsl[resolution].values():
            point["ours"] = round(statistics.mean(point["ours"]))
    for row in read_csv("tacsl.csv"):
        bucket = tacsl.get(row["resolution"])
        if bucket is None:
            continue
        envs = int(row["num_envs"])
        bucket.setdefault(envs, {"envs": envs, "ours": None})["tacsl"] = int(row["fps"])
    tacsl = {k: sorted(v.values(), key=lambda p: p["envs"]) for k, v in tacsl.items()}

    # HydroShear reports milliseconds per rollout step; convert to environment steps per second.
    hydro = {}
    for row in read_csv("hydroshear.csv"):
        envs = int(row["num_envs"])
        hydro.setdefault(envs, {"envs": envs})[row["source"]] = round(
            envs * 1000.0 / float(row["mean_ms"])
        )

    # TacMap: five ElastomerTaxel sensors of 1000+ taxels. Blank CSV cells are out of memory.
    mine = ours(
        GPU,
        ("tactile_sensor_speed_full",),
        lambda r: r["sensor"] == "elastomer_taxel"
        and r.get("n_sensors") == 5
        and not r.get("noise", False),
    )
    tacmap = []
    for row in read_csv("tacmap_benchmark.csv"):
        envs = int(row["num_envs"])
        if envs not in mine or envs < 512:
            continue
        tacmap.append(
            {
                "envs": envs,
                "oursFps": mine[envs]["fps"],
                "oursMemMb": mine[envs]["memMb"],
                "tacmapFps": int(row["sim_fps_w_5_tacmap"]) if row["sim_fps_w_5_tacmap"] else None,
                "tacmapMemMb": int(row["gpu_mb_w_5_tacmap"]) if row["gpu_mb_w_5_tacmap"] else None,
            }
        )
    tacmap = [row for row in tacmap if row["envs"] <= 2048]

    return {"tacsl": tacsl, "hydroshear": sorted(hydro.values(), key=lambda p: p["envs"]), "tacmap": tacmap}



LAYOUT_HANDS = {"wuji": "Wuji", "allegro": "Allegro", "sharpa": "Sharpa"}
LAYOUT_ORDER = ["low", "actual", "high"]
LAYOUT_LABEL = {
    "low": "Synthetic low",
    "actual": "Device layout",
    "high": "Synthetic high",
}


def layouts(images_dir: Path):
    """Probe layouts per hand, with the placement previews copied next to the page."""
    root = ROOT / "dexterous-hands/src/assets/sensors"
    images_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for hand, label in LAYOUT_HANDS.items():
        entries = []
        for name in LAYOUT_ORDER:
            meta_path = root / hand / name / "layout_metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            preview = root / hand / name / "placement_preview.png"
            image = None
            if preview.exists():
                image = f"{hand}-{name}.png"
                shutil.copyfile(preview, images_dir / image)
            entries.append(
                {
                    "id": name,
                    "label": LAYOUT_LABEL[name],
                    "count": meta.get("active_count"),
                    "regions": meta.get("region_counts", {}),
                    "geometry": meta.get("geometry", ""),
                    "note": meta.get("layout_note", ""),
                    "source": meta.get("source_url"),
                    "image": image,
                }
            )
        if entries:
            out.append({"id": hand, "label": label, "layouts": entries})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "benchmark.json": benchmark(),
        "comparison.json": comparison(),
        "layouts.json": layouts(args.out.parent / "images/layouts"),
    }
    for name, payload in payloads.items():
        (args.out / name).write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {args.out / name} ({(args.out / name).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
