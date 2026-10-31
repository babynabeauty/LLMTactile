import argparse
import re
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402, F401 — registers Hanken Grotesk as the default font.

# Match hatch stroke thickness to the bar outline thickness.
matplotlib.rcParams["hatch.linewidth"] = 0.6

ANSI_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]")
SENSOR_LABELS = {
    "no_sensors": "No Sensor",
    "surface_distance_probe": "Surface Distance Probe",
    "contact_depth_probe": "Contact Depth Probe",
    "kinematic_taxel": "Kinematic Taxel",
    "proximity_taxel": "Proximity Taxel",
    "elastomer_taxel": "Elastomer Taxel",
    "temperature_grid": "Temperature Grid",
    # Comparison-plot pseudo-sensors used by plot_tacmap_comparison.py. Listed
    # here only to share the color/label dicts so the comparison plot inherits
    # the rest of the plotter's styling.
    "ours": "Ours",
    "tacmap_baseline": "Tacmap",
    "hydroshear_baseline": "HydroShear",
    "tacsl_baseline": "TacSL",
}
COLORS = {
    "no_sensors": "#767D8F",
    "surface_distance_probe": "#F3AE54",
    "contact_depth_probe": "#497FDB",
    "kinematic_taxel": "#D65974",
    "proximity_taxel": "#57BF6A",
    "elastomer_taxel": "#A96CD1",
    "temperature_grid": "#4ABFB3",
    # comparisons
    "ours": "#FF6060",
    "tacmap_baseline": "#75577D",
    "hydroshear_baseline": "#3F506D",
    "tacsl_baseline": "#3F506D",
}
X_AXIS_LABELS = {
    "batch_size": "Number of environments",
    "n_sample_points": "Point cloud sample points",
    "n_probes": "Number of probes",
}
Y_AXIS_LABELS = {
    "runtime_fps": "Environment steps per second",
    "peak_mem_mb": "Peak GPU memory (MB)",
    "mem_after_build_mb": "GPU memory after build (MB)",
}
# Total GPU memory capacity. The dual plot's memory panel y-axis spans 0..this,
# so each peak-usage bar reads as a fraction of the GPU's total capacity and
# the bottom axis line marks the full 32 GB.
GPU_TOTAL_MB = 32 * 1024
# Each bar is drawn this many inches wide regardless of figure size, so bar
# thickness reads as consistent across plots. The gap between groups fills
# whatever remains of the axis width — narrower figures get tighter groups,
# wider figures get more breathing room.
BAR_WIDTH_INCHES = 0.14


def parse_value(value):
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def parse_line(line):
    line = ANSI_RE.sub("", line).strip()
    if not line or "=" not in line:
        return None

    record = {}
    for item in line.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        record[key.strip()] = parse_value(value)
    return record or None


def load_records(paths):
    records = []
    for path in paths:
        with Path(path).open() as fd:
            for line in fd:
                record = parse_line(line)
                if record is not None:
                    record["source_file"] = str(path)
                    records.append(record)
    return records


def filter_batch_size(records, batch_size):
    """Keep only records at the given batch_size (the no-sensor baseline and
    n_probes/n_sample_points sweeps all carry a batch_size field). Drops records
    that have no batch_size field at all so a stray run can't slip through."""
    kept = [record for record in records if record.get("batch_size") == batch_size]
    dropped = len(records) - len(kept)
    if dropped:
        print(
            f"Filtered to batch_size={batch_size}: dropped {dropped}/{len(records)} record(s) at other batch sizes.",
            file=sys.stderr,
        )
    return kept


def require_fields(records, fields):
    missing = sorted({field for field in fields for record in records if field not in record})
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")


def sort_key(value):
    return (isinstance(value, str), value)


def darken(color, factor=0.5):
    """Return ``color`` scaled toward black by ``factor`` (smaller = darker)."""
    import matplotlib.colors as mcolors

    return tuple(channel * factor for channel in mcolors.to_rgb(color))


def lighten(color, factor=0.8):
    """Return ``color`` mixed toward white by ``factor`` (0 = no change, 1 = white)."""
    import matplotlib.colors as mcolors

    return tuple(c + (1 - c) * factor for c in mcolors.to_rgb(color))


def draw_bars(
    ax,
    records,
    *,
    x_field,
    y_field,
    args,
    label_specs,
    unit="",
    capacity=None,
    label_outside=False,
    group_gap_ratio=None,
):
    """Draw the grouped sensor bars for one metric onto ``ax``.

    ``unit`` is appended to the value label of only the first bar (e.g.
    " FPS"); the rest show the bare number.

    With ``capacity`` set, each bar shows the complement ``capacity - value``
    (free GPU memory) while the value label still reports the raw ``value``
    (memory used). ``label_outside`` forces every value label past the bar tip
    instead of inside.

    ``group_gap_ratio`` overrides the default bar-width rule (constant physical
    inches via ``BAR_WIDTH_INCHES``) and instead sizes bars so the inter-group
    gap equals this many bar widths: ``width = 1 / (n_series + group_gap_ratio)``.
    Use this for small comparison figures where the default rule leaves too
    much whitespace between groups.

    Appends ``(ax, bar, text, label_outside)`` tuples to ``label_specs`` so
    value labels can be rendered later, once the figure layout (and therefore
    each bar's pixel size) is final. Returns a dict describing what was drawn,
    or None when no record carries the requested y-axis metric."""
    from collections import defaultdict

    import numpy as np

    require_fields(records, ("sensor", "n_sensors"))

    # Skip (don't fail on) records lacking the y-axis metric — e.g. older
    # benchmark runs predate GPU-memory tracking and have no peak_mem_mb.
    missing_y = sum(1 for record in records if y_field not in record)
    if missing_y:
        print(
            f"WARNING: skipping {missing_y}/{len(records)} record(s) with no {y_field!r} field "
            f"(likely benchmark runs predating that metric).",
            file=sys.stderr,
        )
    records = [record for record in records if y_field in record]
    if not records:
        print(f"WARNING: no records contain {y_field!r}.", file=sys.stderr)
        return None

    records = [record for record in records if x_field in record]
    if not records:
        raise ValueError(f"No records contain x-axis field {x_field!r}.")

    # Bars are bucketed only by x_field, so every other swept param is silently
    # averaged. For non-batch_size plots that means a stray batch_size sweep in
    # the inputs would corrupt the means; warn loudly instead of failing quietly.
    if x_field != "batch_size":
        batch_sizes = sorted({record["batch_size"] for record in records if "batch_size" in record})
        if len(batch_sizes) > 1:
            print(
                f"WARNING: plotting x={x_field!r} but the inputs span multiple batch_size values "
                f"{batch_sizes}; bars will average across them. Each input file should be a single "
                f"sweep at one fixed batch_size.",
                file=sys.stderr,
            )

    present_sensors = {record["sensor"] for record in records}
    sensors = [sensor for sensor in SENSOR_LABELS if sensor in present_sensors]
    sensors.extend(sorted(present_sensors - set(SENSOR_LABELS)))

    all_x_values = sorted({record[x_field] for record in records if x_field in record}, key=sort_key)
    x = np.arange(len(all_x_values))

    # Each requested n_sensors value becomes its own bar per sensor. The lowest
    # clean count keeps the plain sensor colour; higher clean counts add a slash
    # hatch, and any noisy count adds a dotted hatch — so e.g. (1 clean, 5 clean,
    # 1 noisy) reads as (solid, slashes, dots) per sensor at a glance.
    clean_counts = sorted(args.num_sensors)
    noisy_counts = sorted(getattr(args, "noisy_num_sensors", []) or [])
    # Flat ordered list of (n_sensors, noisy) entries; clean variants first.
    count_specs = [(c, False) for c in clean_counts] + [(c, True) for c in noisy_counts]
    grouped = len(count_specs) > 1
    CLEAN_HATCHES = [None, "//", "xx"]
    NOISY_HATCHES = ["..", "oo", "**"]
    hatch_for: dict = {}
    for i, c in enumerate(clean_counts):
        hatch_for[(c, False)] = CLEAN_HATCHES[min(i, len(CLEAN_HATCHES) - 1)]
    for i, c in enumerate(noisy_counts):
        hatch_for[(c, True)] = NOISY_HATCHES[min(i, len(NOISY_HATCHES) - 1)]

    # Flat list of (sensor, n_sensors, noisy) bars. Real sensors get one bar per
    # (count, noisy) entry; the no-sensor baseline has no count axis, so it gets
    # a single solid bar.
    series_specs = []
    for sensor in sensors:
        if sensor == "no_sensors":
            series_specs.append((sensor, None, False))
        else:
            series_specs.extend((sensor, c, n) for (c, n) in count_specs)
    n_series = max(len(series_specs), 1)
    if group_gap_ratio is not None:
        # Size bars so the gap between adjacent groups equals
        # ``group_gap_ratio`` bar widths: n_series*w + ratio*w = 1 data unit.
        # Bars scale with the figure here (the constant-physical-inches rule
        # is skipped) — used for small comparison figures.
        width = 1.0 / (n_series + group_gap_ratio)
    else:
        # Convert the constant physical bar width (inches) into data units using
        # the axis's current pixel extent. The x-axis spans len(all_x_values) data
        # units (one per group), so inches-per-data-unit = panel_width_in / N. For
        # dual plots the axis position has been pinned by subplots_adjust; for the
        # single-plot path it's the default subplot rect, which is close enough
        # that bars still read as consistent across plots.
        panel_width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
        width = BAR_WIDTH_INCHES * len(all_x_values) / panel_width_in

    # Only the first labelled bar carries the unit suffix; the rest show the
    # bare number so the plot stays uncluttered.
    first_labeled = True

    for i, (sensor, count, noisy) in enumerate(series_specs):
        series = [
            record
            for record in records
            if record["sensor"] == sensor
            and (record["n_sensors"] == 0 if count is None else record["n_sensors"] == count)
            and bool(record.get("noise", False)) == noisy
        ]
        if not series:
            continue

        x_fps = defaultdict(list)
        for record in series:
            if x_field not in record or y_field not in record:
                continue
            x_fps[record[x_field]].append(record[y_field])

        means = []
        stds = []
        present = []
        for xv in all_x_values:
            if xv in x_fps:
                means.append(np.mean(x_fps[xv]))
                stds.append(np.std(x_fps[xv]))
                present.append(True)
            else:
                means.append(0)
                stds.append(0)
                present.append(False)

        means = np.array(means)
        stds = np.array(stds)

        offset = width * (i - (n_series - 1) / 2)
        # Per-sensor legend entries only in single-count mode; grouped mode
        # builds a custom pattern + color legend instead.
        legend_label = None if grouped else SENSOR_LABELS.get(sensor, sensor)
        base_color = COLORS[sensor]
        # Uniform thin darkened outline on every bar. Hatch line thickness is
        # tied to `hatch.linewidth` (set globally below) so the inside pattern
        # matches the outline weight.
        hatch = hatch_for.get((count, noisy)) if count is not None else None
        bar_style = dict(
            width=width,
            color=base_color,
            hatch=hatch,
            edgecolor=darken(base_color),
            linewidth=0.6,
        )
        # With `capacity` set the lower bar shows the complement (free GPU
        # memory) and a lighter stacked segment fills the rest up to capacity,
        # so every column reaches the same height. The value label still
        # reports the raw value (MB used).
        if capacity is None:
            heights = means
        else:
            heights = np.array([max(0.0, capacity - m) if p else 0.0 for m, p in zip(means, present)])
        value_bars = ax.bar(x + offset, heights, yerr=stds, label=legend_label, capsize=3, **bar_style)
        if capacity is not None:
            used_heights = np.array([m if p else 0.0 for m, p in zip(means, present)])
            light_style = dict(
                width=width,
                color=lighten(base_color),
                hatch=hatch,
                edgecolor=darken(base_color),
                linewidth=0.6,
            )
            ax.bar(x + offset, used_heights, bottom=heights, **light_style)
        # Labels always anchor to the saturated bar's tip (the free/used seam
        # in capacity mode), so the number sits where the *actual* bar ends
        # rather than at the capacity ceiling.
        for bar, mean, p in zip(value_bars, means, present):
            if p and mean > 0:
                text = f"{mean:.0f}{unit}" if first_labeled else f"{mean:.0f}"
                first_labeled = False
                label_specs.append((ax, bar, text, label_outside))

    ax.set_xticks(x, [str(xv) for xv in all_x_values])
    # Tight x-limits so the first/last groups sit near the panel edges.
    ax.set_xlim(-0.5, len(all_x_values) - 0.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return {
        "all_x_values": all_x_values,
        "x": x,
        "sensors": sensors,
        "grouped": grouped,
        "count_specs": count_specs,
        "hatch_for": hatch_for,
    }


def _count_spec_label(count, noisy):
    """Legend text for one (count, noisy) entry — kept in one place so the
    standalone legend PNGs and any inline legend stay in sync."""
    base = f"{count} sensor{'' if count == 1 else 's'}"
    return f"{base} with noise options" if noisy else base


def _pattern_handles(info):
    from matplotlib.patches import Patch

    return [
        Patch(
            facecolor="white",
            hatch=info["hatch_for"][(count, noisy)],
            edgecolor="0.4",
            linewidth=1.0,
            label=_count_spec_label(count, noisy),
        )
        for (count, noisy) in info["count_specs"]
    ]


def _sensor_handles(info):
    from matplotlib.patches import Patch

    return [Patch(facecolor=COLORS[sensor], label=SENSOR_LABELS.get(sensor, sensor)) for sensor in info["sensors"]]


def add_legend(ax, info, **legend_kwargs):
    """Attach the sensor legend to ``ax`` from a draw_bars() info dict. Grouped
    plots get a two-part legend: the (n_sensors, noise) patterns first, then the
    per-sensor colors."""
    if info["grouped"]:
        ax.legend(handles=_pattern_handles(info) + _sensor_handles(info), **legend_kwargs)
    else:
        ax.legend(**legend_kwargs)


def save_standalone_legend(handles, output_path, *, ncol=1):
    """Render ``handles`` as a standalone legend PNG, tightly cropped."""
    import matplotlib.pyplot as plt

    labels = [h.get_label() for h in handles]
    rows = -(-len(labels) // ncol)
    fig = plt.figure(figsize=(2.8 * ncol, max(0.8, 0.28 * rows + 0.4)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    legend = ax.legend(handles, labels, loc="center", fontsize=10, frameon=True, framealpha=1.0, ncol=ncol)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches=bbox.expanded(1.05, 1.05))
    plt.close(fig)


def render_bar_labels(fig, label_specs):
    """Draw each bar's value once the layout (and thus bar pixel sizes) is
    settled. Labels that fit inside the bar are white; labels that overflow a
    too-short bar — or whose spec set ``label_outside`` — are black just past
    the tip (honoring inverted y-axes)."""
    import matplotlib.patheffects as pe

    # Rotated text's geometric centre sits left of its visual centre because
    # the descender lengthens the un-rotated bbox on one side; nudge every
    # label a hair right so it reads as centred over the bar.
    X_NUDGE_PT = 1.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, bar, text, outside in label_specs:
        # Rough vertical extent of the rotated text, with a little padding.
        text_extent_pt = 3.0 * len(text) + 14.0
        bar_height_pt = bar.get_window_extent(renderer).height * 72.0 / fig.dpi
        fits = bar_height_pt >= text_extent_pt and not outside
        xc = bar.get_x() + bar.get_width() / 2
        y1 = bar.get_y() + bar.get_height()
        # `rotation_mode="anchor"` rotates around the precise anchor point
        # implied by ha/va, which keeps the rotated text centered on the bar
        # instead of biased by the un-rotated bounding box.
        common = dict(ha="center", fontsize="x-small", fontweight=600, rotation=90, rotation_mode="anchor")
        inverted = ax.yaxis_inverted()
        if fits:
            # White, inside the bar, centered a little in from the tip. The
            # path_effects stroke (same colour as the fill) thickens the glyphs
            # so the labels read as genuinely bold even when the variable-font
            # weight axis isn't honoured by the current matplotlib backend.
            inset = (text_extent_pt / 2 + 3.0) * (1 if inverted else -1)
            ax.annotate(
                text,
                xy=(xc, y1),
                xytext=(X_NUDGE_PT, inset),
                textcoords="offset points",
                va="center",
                color="white",
                path_effects=[pe.withStroke(linewidth=0.3, foreground="white")],
                **common,
            )
        else:
            # Too short to fit the number inside: place it in black just past
            # the bar tip (above for upward bars, below for a mirrored panel).
            # With rotation=90, va="bottom" would put the un-rotated bbox's
            # bottom edge at the anchor — which becomes the rotated bbox's
            # *left* edge, shifting the text leftward by its full height. Use
            # va="center" + a half-width y-offset so the rotated bbox sits
            # symmetrically above the bar tip, matching the inside branch.
            outset = (text_extent_pt / 2 + 0.0) * (-1 if inverted else 1)
            ax.annotate(
                text,
                xy=(xc, y1),
                xytext=(X_NUDGE_PT, outset),
                textcoords="offset points",
                va="center",
                color="black",
                path_effects=[pe.withStroke(linewidth=0.3, foreground="black")],
                **common,
            )


def plot_records(records, *, args):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tuple(args.figsize), constrained_layout=True)
    label_specs = []
    info = draw_bars(ax, records, x_field=args.x_axis, y_field=args.y_axis, args=args, label_specs=label_specs)
    if info is None:
        print(f"WARNING: skipping plot {args.output}.", file=sys.stderr)
        plt.close(fig)
        return None

    ax.set_xlabel(X_AXIS_LABELS.get(args.x_axis, args.x_axis))
    ax.set_ylabel(Y_AXIS_LABELS.get(args.y_axis, args.y_axis))
    ax.set_title(args.title)
    add_legend(ax, info, ncols=1, fontsize="small")

    render_bar_labels(fig, label_specs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def draw_dual_panel(
    fig,
    ax_top,
    ax_bot,
    records,
    *,
    x_field,
    num_sensors,
    title,
    sublabel,
    label_specs,
    ymax_fps,
    ymax_mem_mb,
    show_y_labels,
    show_legend=True,
    noisy_num_sensors=(),
    baseline_batch_size=None,
    baseline_label="",
    baseline_label_axis=None,
):
    """Draw one (FPS top, free-memory bottom) panel pair onto pre-built axes.
    Used by both the single-x ``plot_dual`` and the multi-column
    ``plot_multi_dual``. Figure-level layout (gridspec, subplots_adjust) and
    label rendering happen in the caller; here we only draw bars, set y-limits,
    and place the seam text for this column."""
    from types import SimpleNamespace

    panel_args = SimpleNamespace(num_sensors=num_sensors, noisy_num_sensors=noisy_num_sensors)
    top_field = "runtime_fps"
    bottom_field = "peak_mem_mb"

    info_top = draw_bars(
        ax_top, records, x_field=x_field, y_field=top_field, args=panel_args, label_specs=label_specs, unit=" FPS"
    )
    # The memory panel draws free GPU memory (capacity minus peak usage) as a
    # plain sensor-coloured bar; the value labels report the MB actually used
    # and sit above the bar (label_outside) in the headroom below the seam.
    info_bot = draw_bars(
        ax_bot,
        records,
        x_field=x_field,
        y_field=bottom_field,
        args=panel_args,
        label_specs=label_specs,
        unit=" MB used",
        capacity=ymax_mem_mb,
        label_outside=True,
    )
    if info_top is None or info_bot is None:
        raise ValueError(f"Dual panel x={x_field!r} needs records carrying both 'runtime_fps' and 'peak_mem_mb'.")

    if ymax_fps is not None:
        ax_top.set_ylim(0, ymax_fps)

    # Optional no-sensor FPS baseline: a horizontal dashed line at the mean
    # runtime_fps of the no-sensor records at `baseline_batch_size`. Drawn only
    # on the FPS axis (the memory panel has its own baseline structure via the
    # no-sensor bar). Labeled when `baseline_label` is set.
    if baseline_batch_size is not None:
        import numpy as np

        baseline_vals = [
            r["runtime_fps"]
            for r in records
            if r.get("sensor") == "no_sensors" and r.get("batch_size") == baseline_batch_size and "runtime_fps" in r
        ]
        if baseline_vals:
            baseline_fps = float(np.mean(baseline_vals))
            ax_top.axhline(baseline_fps, ls="--", color="0.25", lw=1.2, zorder=4)
            if baseline_label:
                # Anchor on a different axis (e.g. the rightmost panel) when
                # the caller wants the text right-aligned past this panel's
                # right edge. sharey keeps the data-y coordinate consistent.
                text_ax = baseline_label_axis if baseline_label_axis is not None else ax_top
                text_ax.text(
                    1.0,
                    baseline_fps,
                    f"{baseline_label}, {baseline_batch_size} envs: {baseline_fps:.0f} FPS",
                    transform=text_ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    color="0.25",
                    zorder=5,
                )
        else:
            print(
                f"WARNING: baseline requested at batch_size={baseline_batch_size} but no matching "
                f"no_sensors record found.",
                file=sys.stderr,
            )
    # Memory panel y-axis: free memory from 0 at the bottom edge, with extra
    # headroom above the capacity ceiling for the always-outside value labels.
    ax_bot.set_ylim(0, ymax_mem_mb * 1.4)
    # Clear the panel background so the speed panel's 0 shows through the seam.
    ax_bot.patch.set_visible(False)
    # GB ticks: aim for ~4-5 labels across the range. For the default 48 GB
    # capacity this gives [0, 12, 24, 36, 48].
    cap_gb = int(ymax_mem_mb // 1024)
    gb_step = 12 if cap_gb >= 36 else 6 if cap_gb >= 18 else max(1, cap_gb // 4)
    gb_ticks = list(range(0, cap_gb + 1, gb_step))
    ax_bot.set_yticks([g * 1024 for g in gb_ticks])
    ax_bot.set_yticklabels([str(g) for g in gb_ticks])

    if show_y_labels:
        ax_top.set_ylabel(Y_AXIS_LABELS[top_field])
        ax_bot.set_ylabel("GPU memory (GB)")
    else:
        # Hide tick labels on non-leftmost columns; sharey keeps the ticks
        # aligned with the leftmost column.
        ax_top.tick_params(axis="y", left=False, labelleft=False)
        ax_bot.tick_params(axis="y", left=False, labelleft=False)
        ax_top.spines["left"].set_visible(False)
        ax_bot.spines["left"].set_visible(False)

    # Drop the default x tick labels on both panels — figure-level text replaces
    # them. The FPS x values hang from just under the seam (top of the memory
    # panel); the GPU x values sit on the memory panel's x-axis (bottom of the
    # memory panel). ``get_xaxis_transform`` puts x in data coords and y in axes
    # coords, so both placements are independent of ``ymax_mem``.
    for ax in (ax_top, ax_bot):
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False, labeltop=False)
    box = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
    bot_pos = ax_bot.get_position()
    # FPS x value labels: hang from the seam (top of memory panel) so they sit
    # right under the FPS panel's x-axis.
    for xv, xpos in zip(info_top["all_x_values"], info_top["x"]):
        ax_bot.text(
            xpos,
            0.96,
            str(xv),
            transform=ax_bot.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            bbox=box,
            zorder=6,
        )
    # GPU x value labels: sit on the memory panel's x-axis at the bottom.
    for xv, xpos in zip(info_bot["all_x_values"], info_bot["x"]):
        ax_bot.text(
            xpos,
            0.04,
            str(xv),
            transform=ax_bot.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=box,
            zorder=6,
        )
    # X-axis title and sublabel sit in the figure margin below the memory panel.
    cx = (bot_pos.x0 + bot_pos.x1) / 2
    fig.text(
        cx,
        bot_pos.y0 - 0.022,
        X_AXIS_LABELS.get(x_field, x_field),
        ha="center",
        va="center",
        fontsize=12,
        zorder=6,
    )
    if sublabel:
        fig.text(
            cx,
            bot_pos.y0 - 0.045,
            sublabel,
            ha="center",
            va="center",
            fontsize=10,
            zorder=6,
        )

    # Legend pinned to the top-left for the dual plot. In multi-dual mode only
    # the leftmost panel carries a legend (its sensor set is a superset of the
    # narrow panels' subsets).
    if show_legend:
        add_legend(ax_top, info_top, loc="upper left", ncols=1, fontsize="small")
    # Title sits above each panel — placed in-panel it collides with the legend
    # in narrow columns of the multi-dual layout.
    ax_top.set_title(title, fontsize="large", pad=2)
    return info_top


def plot_dual(records, *, args):
    """Two-panel plot: environment steps/s on top, free GPU memory on the
    bottom. Thin wrapper around ``draw_dual_panel`` that builds the figure."""
    import matplotlib.pyplot as plt

    # Memory panel height defaults to half the speed panel (override via --height-ratios).
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=tuple(args.figsize), gridspec_kw={"hspace": 0.0, "height_ratios": args.height_ratios}
    )
    # Manual margins (no constrained layout) so the panels meet flush at the
    # shared x-axis seam instead of being pushed apart to fit tick labels.
    fig.subplots_adjust(left=0.055, right=0.992, top=0.94, bottom=0.08, hspace=0.0)

    ymax_mem_mb = int(args.ymax_mem_gb * 1024) if args.ymax_mem_gb else GPU_TOTAL_MB
    label_specs = []
    draw_dual_panel(
        fig,
        ax_top,
        ax_bot,
        records,
        x_field=args.x_axis,
        num_sensors=args.num_sensors,
        noisy_num_sensors=args.noisy_num_sensors,
        title=args.title,
        sublabel=args.sublabel,
        label_specs=label_specs,
        ymax_fps=args.ymax_fps,
        ymax_mem_mb=ymax_mem_mb,
        show_y_labels=True,
        baseline_batch_size=args.baseline_batch_size,
        baseline_label=args.baseline_label,
    )

    render_bar_labels(fig, label_specs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def parse_panel_spec(spec_str):
    """Parse a ``--panel`` argument string into a dict.

    Format: ``key=value;key=value;...``. Keys: ``x`` (x-axis field, required),
    ``n`` (comma-separated num_sensors, required), ``nn`` (comma-separated
    *noisy* num_sensors, optional — adds a dotted-hatch bar per sensor matched
    against records with noise=True), ``b`` (batch_size filter, optional),
    ``bl`` (any truthy value enables a horizontal no-sensor baseline line on
    the FPS panel at the panel's ``b`` batch_size), ``bllabel`` (optional text
    annotation rendered on the baseline line), ``title`` (plot title), ``sub``
    (sublabel under the x-axis title). Splits on the *first* ``=`` per item so
    values may contain ``=``.
    """
    spec = {}
    for item in spec_str.split(";"):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Bad panel spec item {item!r} (expected key=value)")
        key, value = item.split("=", 1)
        spec[key.strip()] = value.strip()
    if "x" not in spec or "n" not in spec:
        raise ValueError(f"Panel spec {spec_str!r} missing required keys 'x' and 'n'.")
    spec["num_sensors"] = [int(c) for c in spec["n"].split(",") if c.strip()]
    spec["noisy_num_sensors"] = [int(c) for c in spec["nn"].split(",") if c.strip()] if spec.get("nn") else []
    spec["batch_size"] = int(spec["b"]) if spec.get("b") else None
    spec["baseline"] = bool(spec.get("bl"))
    spec["baseline_label"] = spec.get("bllabel", "")
    spec.setdefault("title", "")
    spec.setdefault("sub", "")
    return spec


def plot_multi_dual(panel_records, panel_specs, *, args):
    """One figure with N (FPS + memory) column pairs, sharing y across columns.
    Only the leftmost column shows y-axis labels; the others suppress them so
    the bars read against the shared scale set by ``--ymax-fps`` /
    ``--ymax-mem-gb``."""
    import matplotlib.pyplot as plt

    n = len(panel_specs)
    width_ratios = args.width_ratios if args.width_ratios else [1.0] * n
    if len(width_ratios) != n:
        raise ValueError(f"--width-ratios needs {n} values (one per panel), got {len(width_ratios)}.")

    fig = plt.figure(figsize=tuple(args.figsize))
    gs = fig.add_gridspec(
        2,
        n,
        height_ratios=args.height_ratios,
        width_ratios=width_ratios,
        hspace=0.0,
        wspace=0.02,
    )

    # Build axes column by column, sharing y per row so all columns honor the
    # same y-limits set on the leftmost axis.
    axes_top = []
    axes_bot = []
    for col in range(n):
        if col == 0:
            ax_t = fig.add_subplot(gs[0, col])
            ax_b = fig.add_subplot(gs[1, col], sharex=ax_t)
        else:
            ax_t = fig.add_subplot(gs[0, col], sharey=axes_top[0])
            ax_b = fig.add_subplot(gs[1, col], sharex=ax_t, sharey=axes_bot[0])
        axes_top.append(ax_t)
        axes_bot.append(ax_b)

    # Pin margins before drawing so each panel's pixel width is final when
    # draw_bars converts BAR_WIDTH_INCHES into data units. Left margin is sized
    # in inches (not a figure fraction) so the y-axis label + tick numbers fit
    # regardless of figure width.
    fig_w = fig.get_size_inches()[0]
    fig.subplots_adjust(left=1.1 / fig_w, right=1 - 0.05 / fig_w, top=0.94, bottom=0.08, hspace=0.0, wspace=0.02)

    ymax_mem_mb = int(args.ymax_mem_gb * 1024) if args.ymax_mem_gb else GPU_TOTAL_MB

    label_specs = []
    leftmost_info = None
    for col, (records, spec) in enumerate(zip(panel_records, panel_specs)):
        col_records = records
        if spec["batch_size"] is not None:
            col_records = filter_batch_size(col_records, spec["batch_size"])
            if not col_records:
                raise ValueError(f"Panel x={spec['x']!r} has no records at batch_size={spec['batch_size']}.")

        info_top = draw_dual_panel(
            fig,
            axes_top[col],
            axes_bot[col],
            col_records,
            x_field=spec["x"],
            num_sensors=spec["num_sensors"],
            noisy_num_sensors=spec.get("noisy_num_sensors", []),
            title=spec["title"],
            sublabel=spec["sub"],
            label_specs=label_specs,
            ymax_fps=args.ymax_fps,
            ymax_mem_mb=ymax_mem_mb,
            show_y_labels=(col == 0),
            show_legend=False,  # multi-dual emits standalone legend PNGs instead
            baseline_batch_size=spec["batch_size"] if spec.get("baseline") else None,
            baseline_label=spec.get("baseline_label", ""),
            baseline_label_axis=axes_top[-1],
        )
        if col == 0:
            leftmost_info = info_top

    render_bar_labels(fig, label_specs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    # Standalone legend PNGs alongside the figure: one for sensor-type colors,
    # one for the (n_sensors, noise) pattern entries. Built from the leftmost
    # panel's info — its sensor set and count_specs are the superset.
    if leftmost_info is not None:
        save_standalone_legend(_sensor_handles(leftmost_info), output.parent / "legend_sensors.png")
        if leftmost_info["grouped"]:
            save_standalone_legend(_pattern_handles(leftmost_info), output.parent / "legend_counts.png")

    return output


def main():
    parser = argparse.ArgumentParser(description="Plot tactile sensor benchmark results.")
    parser.add_argument("files", nargs="*", type=Path, help="Benchmark result text files (ignored with --multi-dual).")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tactile_sensor_speed.png"),
        help="Output image path. Defaults to tactile_sensor_speed.png.",
    )
    parser.add_argument(
        "--num-sensors",
        "-n",
        type=int,
        nargs="+",
        default=[5],
        help="n_sensors value(s) to plot. Pass several (e.g. -n 1 5) to draw them as grouped bars "
        "per sensor, with lower counts shown lighter. Defaults to 5.",
    )
    parser.add_argument(
        "--noisy-num-sensors",
        type=int,
        nargs="*",
        default=[],
        help="n_sensors value(s) for the *noisy* variant. Each gets an extra dotted-hatch bar per "
        "sensor matched against records with noise=True. e.g. `--noisy-num-sensors 1` adds the "
        "1-sensor noisy bar alongside the clean 1- and 5-sensor bars.",
    )
    parser.add_argument(
        "--x-axis", "-x", choices=sorted(X_AXIS_LABELS), default="batch_size", help="Field to use for the x axis."
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="If set, keep only records at this batch_size. Use for the n_probes / "
        "n_sample_points sweeps so bars are not averaged across environment counts.",
    )
    parser.add_argument(
        "--y-axis",
        "-y",
        choices=sorted(Y_AXIS_LABELS),
        default="runtime_fps",
        help="Metric to plot on the y axis. Defaults to runtime_fps. Ignored with --dual.",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Two-panel plot: env steps/s on top, GPU memory below, sharing one x-axis. Ignores --y-axis.",
    )
    parser.add_argument(
        "--multi-dual",
        action="store_true",
        help="Multi-column dual plot: one (FPS+memory) pair per panel, sharing y across columns. "
        "Provide each panel via matched --panel / --panel-files arguments in the same order.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        default=[],
        help="Panel spec for --multi-dual: 'x=FIELD;n=N1,N2;b=BATCH;title=TEXT;sub=TEXT'. "
        "Repeat once per panel; pair each with a --panel-files in the same order.",
    )
    parser.add_argument(
        "--panel-files",
        action="append",
        nargs="+",
        type=Path,
        default=[],
        help="Files for the corresponding --panel. Repeat once per panel.",
    )
    parser.add_argument(
        "--width-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Column width ratios for --multi-dual (one per panel).",
    )
    parser.add_argument(
        "--height-ratios",
        type=float,
        nargs=2,
        metavar=("TOP", "BOTTOM"),
        default=[2.0, 1.0],
        help="Row height ratios for the (FPS, GPU memory) panels in --dual / --multi-dual. "
        "Defaults to 2 1 (memory panel half the FPS panel's height).",
    )
    parser.add_argument(
        "--ymax-fps",
        type=float,
        default=None,
        help="Top of the FPS y-axis. Use with --multi-dual to give all panels the same FPS scale.",
    )
    parser.add_argument(
        "--ymax-mem-gb",
        type=float,
        default=None,
        help="Top of the GPU memory y-axis in GB. Defaults to 32 (an RTX 5090).",
    )
    parser.add_argument(
        "--sublabel",
        type=str,
        default="",
        help="Optional second line below the x-axis title (single dual mode). At value-label size.",
    )
    parser.add_argument(
        "--baseline-batch-size",
        type=int,
        default=None,
        help="Draw a dashed no-sensor FPS baseline line on the FPS panel at this batch_size "
        "(single-dual mode). For multi-dual, use the panel-spec 'bl' key instead.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="",
        help="Optional text annotation rendered on the baseline line.",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(9.0, 4.0), help="Figure size in inches."
    )
    parser.add_argument("--title", "-t", type=str, default="Tactile Sensor Benchmark", help="Plot title.")
    args = parser.parse_args()

    if args.multi_dual:
        if len(args.panel) != len(args.panel_files):
            raise ValueError(
                f"--panel and --panel-files must come in matched pairs "
                f"(got {len(args.panel)} configs and {len(args.panel_files)} file groups)."
            )
        if not args.panel:
            raise ValueError("--multi-dual requires at least one --panel / --panel-files pair.")
        panel_specs = [parse_panel_spec(spec) for spec in args.panel]
        panel_records = []
        for files in args.panel_files:
            records = load_records(files)
            if not records:
                raise ValueError(f"No benchmark records in {files}.")
            panel_records.append(records)
        output = plot_multi_dual(panel_records, panel_specs, args=args)
    else:
        if not args.files:
            raise ValueError("Provide benchmark files (or use --multi-dual).")
        records = load_records(args.files)
        if not records:
            raise ValueError("No benchmark records found.")
        if args.batch_size is not None:
            records = filter_batch_size(records, args.batch_size)
            if not records:
                raise ValueError(f"No benchmark records at batch_size={args.batch_size}.")
        output = plot_dual(records, args=args) if args.dual else plot_records(records, args=args)
    if output is not None:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
