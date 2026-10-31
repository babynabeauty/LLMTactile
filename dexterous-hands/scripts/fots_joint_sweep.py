"""Joint dilate+shear parameter search for any marker model (ours / FOTS / HydroShear).

Unlike run_fots_compare.sh (STAGED: calibrate dilate, then shear), this evaluates EACH parameter set
on BOTH the FOTS dilate frame and the shear frame, then reports the BEST set for dilate and the BEST
set for shear separately. Two methods:

    --method optuna  (default)  Bayesian optimization (Optuna TPE) over continuous bounds.
    --method grid               exhaustive grid over the comma-separated value lists.

Each model has its own search axes (dilate-relevant vs shear-only); dilate depends only on the
dilate axes, shear on both:
    elastomer : dilate = thickness, lambda_d, compressibility ; shear += lambda_s, shear_scale
    fots      : dilate = fots_lambda_d                        ; shear += fots_lambda_s
    hydroshear: dilate = hydroshear_lambda_d                  ; shear += hydroshear_lambda_s, hydroshear_mu

Each evaluation runs ``fots_marker_compare.py`` as a FRESH SUBPROCESS (one Genesis scene, then exit)
-- per-eval isolation keeps a long search stable (in-process scene churn segfaults Genesis). Up to
``--jobs`` run concurrently. Best dilate + best shear markers are written to --save-markers.

    python scripts/fots_joint_sweep.py --marker-model fots --method optuna --n-trials 50 --jobs 20 \
        --center-px 136 124 --shear-start 136 118 0 --shear-stop 133 135 -5 \
        --save-markers outputs/fots_compare/markers_joint.npz
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CMP = str(Path(__file__).resolve().parent / "fots_marker_compare.py")
_REL = re.compile(r"\brel=([0-9.eE+-]+)")
_FIT = re.compile(r"\bfit:.*")

# param name (== argparse dest) -> (CLI flag, scale). Bounds/values come from the matching CLI arg.
PARAMS = {
    "thickness": ("--elastomer-thickness", "lin"),
    "lambda_d": ("--lambda-d", "log"),
    "compressibility": ("--compressibility", "lin"),
    "lambda_s": ("--lambda-s", "log"),
    "shear_scale": ("--shear-scale", "lin"),
    "fots_lambda_d": ("--fots-lambda-d", "log"),
    "fots_lambda_s": ("--fots-lambda-s", "log"),
    "hydroshear_lambda_d": ("--hydroshear-lambda-d", "log"),
    "hydroshear_lambda_s": ("--hydroshear-lambda-s", "log"),
    "hydroshear_mu": ("--hydroshear-mu", "lin"),
}
DILATE_AXES = {"elastomer": ["thickness", "lambda_d", "compressibility"],
               "fots": ["fots_lambda_d"], "hydroshear": ["hydroshear_lambda_d"]}
SHEAR_AXES = {"elastomer": ["lambda_s", "shear_scale"],
              "fots": ["fots_lambda_s"], "hydroshear": ["hydroshear_lambda_s", "hydroshear_mu"]}


def _floats(s) -> list[float]:
    return [float(x) for x in str(s).split(",") if x != ""]


def _common(args) -> list[str]:
    cmd = ["--ball-radius", f"{args.ball_radius:g}", "--depth", f"{args.depth:g}",
           "--depth-query", args.depth_query, "--control-mode", args.control_mode,
           "--smooth-real", f"{args.smooth_real:g}", "--marker-model", args.marker_model]
    if args.no_boundary:
        cmd.append("--no-boundary")
    if args.gpu:
        cmd.append("--gpu")
    if args.depth_downsample != 1:
        cmd += ["--depth-downsample", str(args.depth_downsample)]
    return cmd


def eval_once(args, motion, params, save=None, out=None):
    """Run one fots_marker_compare subprocess; return ``(rel, fitline)``. ``rel`` is inf on failure.
    ``params`` maps param-name -> value (only the ones relevant to this motion)."""
    cmd = [sys.executable, CMP] + _common(args)
    if motion == "dilate":
        cmd += ["--frame", "0", "--motion", "dilate",
                "--center-px", f"{args.center_px[0]:g}", f"{args.center_px[1]:g}"]
    else:
        cmd += ["--frame", "1", "--motion", "shear",
                "--shear-start", *[f"{v:g}" for v in args.shear_start],
                "--shear-stop", *[f"{v:g}" for v in args.shear_stop]]
    for name, val in params.items():
        cmd += [PARAMS[name][0], f"{val:g}"]
    if save:
        cmd += ["--save-markers", save, "--force-markers"]
    if out:
        cmd += ["--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    txt = p.stdout + p.stderr
    if p.returncode != 0:
        sig = -p.returncode if p.returncode < 0 else p.returncode
        print(f"      WARN: eval failed (motion={motion}, exit/sig={sig}) -> rel=inf", file=sys.stderr)
        return float("inf"), ""
    rels = _REL.findall(txt)
    if not rels:
        print(f"      WARN: no rel parsed (motion={motion}) -> rel=inf", file=sys.stderr)
        return float("inf"), ""
    fit = _FIT.search(txt)
    return float(rels[-1]), (fit.group(0) if fit else "")


def _track(best, rel, fit, params):
    if best is None or rel < best["rel"]:
        return dict(rel=rel, fit=fit, params=dict(params))
    return best


def _p(params):
    return " ".join(f"{k}={v:g}" for k, v in params.items())


def run_grid(args):
    model = args.marker_model
    dnames, snames = DILATE_AXES[model], SHEAR_AXES[model]
    dvals = [_floats(getattr(args, n)) for n in dnames]
    svals = [_floats(getattr(args, n)) for n in snames]
    tasks = []
    for dcombo in itertools.product(*dvals):
        dp = dict(zip(dnames, dcombo))
        tasks.append(("dilate", dp))
        for scombo in itertools.product(*svals):
            tasks.append(("shear", {**dp, **dict(zip(snames, scombo))}))
    print(f"[grid] {len(tasks)} evals, {args.jobs} parallel")
    best_d = best_s = None
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(eval_once, args, mo, pr): (mo, pr) for mo, pr in tasks}
        for f in as_completed(futs):
            mo, pr = futs[f]
            rel, fit = f.result()
            done += 1
            print(f"  [{done}/{len(tasks)}] {mo.upper()} {_p(pr)}: rel={rel:.3f}")
            if mo == "dilate":
                best_d = _track(best_d, rel, fit, pr)
            else:
                best_s = _track(best_s, rel, fit, pr)
    return best_d, best_s


def run_optuna(args):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model = args.marker_model
    dnames, snames = DILATE_AXES[model], SHEAR_AXES[model]
    allnames = dnames + snames
    bounds = {n: (min(v), max(v)) for n in allnames for v in [_floats(getattr(args, n))]}
    print(f"[optuna] {model}: {args.n_trials} trials, {args.jobs} parallel  bounds: "
          + "  ".join(f"{n}[{lo:g},{hi:g}]" for n, (lo, hi) in bounds.items()))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    state = {"d": None, "s": None, "i": 0}
    lock = threading.Lock()

    def worker(_):
        trial = study.ask()
        vals = {n: trial.suggest_float(n, *bounds[n], log=(PARAMS[n][1] == "log")) for n in allnames}
        dparams = {n: vals[n] for n in dnames}
        sparams = {n: vals[n] for n in allnames}   # shear sees the dilate axes too
        dr, df = eval_once(args, "dilate", dparams)
        sr, sf = eval_once(args, "shear", sparams)
        study.tell(trial, (dr if dr != float("inf") else 9.99) + (sr if sr != float("inf") else 9.99))
        with lock:
            state["i"] += 1
            state["d"] = _track(state["d"], dr, df, dparams)
            state["s"] = _track(state["s"], sr, sf, sparams)
            print(f"  [{state['i']}/{args.n_trials}] {_p(vals)}: dilate={dr:.3f} shear={sr:.3f}  "
                  f"(best d={state['d']['rel']:.3f} s={state['s']['rel']:.3f})")

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(worker, range(args.n_trials)))
    return state["d"], state["s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-model", choices=["elastomer", "fots", "hydroshear"], default="elastomer")
    ap.add_argument("--method", choices=["optuna", "grid"], default="optuna")
    ap.add_argument("--n-trials", type=int, default=80, help="optuna trials")
    ap.add_argument("--jobs", type=int, default=4, help="max parallel eval subprocesses")
    ap.add_argument("--seed", type=int, default=0)
    # geometry / motion (forwarded to fots_marker_compare)
    ap.add_argument("--ball-radius", type=float, default=3.0)
    ap.add_argument("--depth", type=float, default=1.5, help="press penetration (mm)")
    ap.add_argument("--depth-query", choices=["sdf", "raycast"], default="sdf")
    ap.add_argument("--control-mode", choices=["position", "teleport"], default="position")
    ap.add_argument("--center-px", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--shear-start", type=float, nargs="+", required=True, metavar="V", help="X Y [ZROT_DEG]")
    ap.add_argument("--shear-stop", type=float, nargs="+", required=True, metavar="V", help="X Y [ZROT_DEG]")
    ap.add_argument("--smooth-real", type=float, default=0.5)
    ap.add_argument("--no-boundary", action="store_true")
    ap.add_argument("--depth-downsample", type=int, default=1)
    ap.add_argument("--gpu", action="store_true")
    # search ranges (comma list; optuna uses min,max as bounds, grid uses every value).
    ap.add_argument("--thickness", default="3,9")
    ap.add_argument("--lambda-d", default="3e5,1e6")
    ap.add_argument("--compressibility", default="0.5,0.95")
    ap.add_argument("--lambda-s", default="3e4,1e6")
    ap.add_argument("--shear-scale", default="1,4")
    ap.add_argument("--fots-lambda-d", default="0.0005,0.005")
    ap.add_argument("--fots-lambda-s", default="0.0005,0.05")
    ap.add_argument("--hydroshear-lambda-d", default="8000,30000")
    ap.add_argument("--hydroshear-lambda-s", default="5000,20000")
    ap.add_argument("--hydroshear-mu", default="0.3,1.0")
    # outputs
    ap.add_argument("--save-markers", default=None, metavar="NPZ", help="append best dilate+shear cells here")
    ap.add_argument("--out-dilate", default=None)
    ap.add_argument("--out-shear", default=None)
    args = ap.parse_args()

    best_d, best_s = (run_optuna if args.method == "optuna" else run_grid)(args)
    if best_d is None or best_s is None or best_d["rel"] == float("inf") or best_s["rel"] == float("inf"):
        print("ERROR: no successful evaluations (every subprocess failed).", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 88)
    print(f"[{args.marker_model}] BEST dilate: {_p(best_d['params'])}  rel={best_d['rel']:.3f}   [{best_d['fit']}]")
    print(f"[{args.marker_model}] BEST shear : {_p(best_s['params'])}  rel={best_s['rel']:.3f}   [{best_s['fit']}]")
    print("=" * 88)

    # re-run the two winners once more to write their markers/panels (fresh subprocess each), with a
    # couple of retries -- the eval subprocesses can crash transiently under load.
    def save_best(motion, params, out):
        for attempt in range(3):
            rel, _ = eval_once(args, motion, params, save=args.save_markers, out=out)
            if rel != float("inf"):
                return True
            print(f"      retry {attempt + 1}/3: best-{motion} save failed", file=sys.stderr)
        print(f"ERROR: could not save best {motion} (engine failing?) -> NOT in {args.save_markers}", file=sys.stderr)
        return False

    ok = True
    if args.save_markers or args.out_dilate:
        ok &= save_best("dilate", best_d["params"], args.out_dilate)
    if args.save_markers or args.out_shear:
        ok &= save_best("shear", best_s["params"], args.out_shear)
    if args.save_markers:
        print(f"saved best dilate+shear markers -> {args.save_markers}" if ok
              else f"WARNING: some best cells were NOT saved to {args.save_markers} (see errors above)")


if __name__ == "__main__":
    main()
