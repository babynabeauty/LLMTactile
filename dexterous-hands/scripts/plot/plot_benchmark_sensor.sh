# One directory per GPU, since the globs below bucket by filename alone and would otherwise
# average runs from different hardware into one bar:
#   DATA_DIR=data/sensor_benchmarks/5090 bash scripts/plot/plot_benchmark_sensor.sh
DATA_DIR=${DATA_DIR:-data/sensor_benchmarks/A6000}
# Published baselines from other papers, which are not per-GPU and so do not move with DATA_DIR.
REF_DIR=data/sensor_benchmarks

# Each glob below corresponds to ONE benchmark sweep. Keep the sweeps in
# separate files — the plotter buckets only by the x-axis field, so mixing a
# batch_size sweep into an n_probes/n_sample_points plot silently averages
# across batch sizes (see pointcloud_batch_size_speed* / probes_batch_size_speed*,
# which are deliberately named so these globs do NOT pick them up).

# no-sensor baseline, batch_size sweep (n_sensors=0)
NO_SENSOR_DATA=$DATA_DIR/no_sensor_scene_speed*.txt
# tactile sensors, batch_size sweep. Holds n_sensors=1 and n_sensors=5 runs;
# the plotter filters by the n_sensors field per --num-sensors, so both coexist.
TACTILE_SENSOR_DATA=$DATA_DIR/tactile_sensor_speed_full*.txt
# tactile sensors, batch_size sweep with noise=True. Surface as an extra
# dotted-hatch bar via `nn=1` on the batch_size panel.
NOISED_SENSOR_DATA=$DATA_DIR/noised_tactile_sensor_speed*.txt
# pointcloud sensors, n_sample_points sweep at fixed batch_size=1024 (n_sensors=1)
POINTCLOUD_SENSOR_DATA=$DATA_DIR/pointcloud_sensor_speed*.txt
# probe sensors, n_probes sweep at fixed batch_size=1024 (n_sensors=1)
PROBES_SENSOR_DATA=$DATA_DIR/probes_sensor_speed*.txt
# temperature_grid sensors, batch_size sweep (n_sensors=1 and n_sensors=5)
TEMPERATURE_SENSOR_DATA=$DATA_DIR/temperature*.txt
# kinematic_taxel, batch_size sweep at TacSL's two taxel resolutions (100 and 10000), from
# test_tacsl_comparison_speed. The n_probes sweep cannot stand in: it is fixed at 1024 envs,
# so plot_tacsl_comparison would draw a single point per resolution.
TACSL_COMPARISON_DATA=$DATA_DIR/tacsl_comparison_speed*.txt
OUT_DIR=scripts/plot/output/tactile_benchmark

# # # --- Combined side-by-side plot (shared FPS + GPU memory scales) ---
# python scripts/plot/plot_benchmark_sensor.py --multi-dual \
#   --output $OUT_DIR/benchmark_combined_split.png \
#   --figsize 5.8 6.5 \
#   --ymax-fps 450000 --ymax-mem-gb 32 \
#   --width-ratios 2 4 \
#   --height-ratios 6 5 \
#   --panel "x=n_sample_points;n=1;b=1024;bl=1;bllabel=No Sensor;title=by Point Cloud Size;sub=(fixed: 1024 envs, 100 probes)" \
#   --panel-files $NO_SENSOR_DATA $POINTCLOUD_SENSOR_DATA \
#   --panel "x=n_probes;n=1;b=1024;bl=1;title=by Number of Probes;sub=(fixed: 1024 envs, 500 point cloud samples)" \
#   --panel-files $NO_SENSOR_DATA $PROBES_SENSOR_DATA

# --- Temperature grid sensor: batch_size sweep, n=1 and n=5 ---
python scripts/plot/plot_benchmark_sensor.py --multi-dual \
  --output $OUT_DIR/benchmark_temperature.png \
  --figsize 4 6.5 \
  --ymax-mem-gb 32 \
  --height-ratios 6 5 \
  --panel "x=batch_size;n=1,5;title=Temperature Grid Sensor;sub=(grid size 2x2x2)" \
  --panel-files $NO_SENSOR_DATA $TEMPERATURE_SENSOR_DATA

# --- Combined side-by-side plot (shared FPS + GPU memory scales) ---
python scripts/plot/plot_benchmark_sensor.py --multi-dual \
  --output $OUT_DIR/benchmark_combined.png \
  --figsize 20 8.5 \
  --ymax-fps 200000 --ymax-mem-gb 32 \
  --width-ratios 18 2 4 \
  --panel "x=batch_size;n=1,5;nn=1;title=Performance Benchmark by Tactile Sensor Type (using NVIDIA RTX 5090);sub=(fixed: 100 probes, 500 point cloud samples)" \
  --panel-files $NO_SENSOR_DATA $TACTILE_SENSOR_DATA $NOISED_SENSOR_DATA \
  --panel "x=n_sample_points;n=1;b=1024;bl=1;bllabel=No Sensor;title=by Point Cloud Size;sub=(fixed: 1024 envs, 100 probes)" \
  --panel-files $NO_SENSOR_DATA $POINTCLOUD_SENSOR_DATA \
  --panel "x=n_probes;n=1;b=1024;bl=1;title=by Number of Probes;sub=(fixed: 1024 envs, 500 point cloud samples)" \
  --panel-files $NO_SENSOR_DATA $PROBES_SENSOR_DATA

# # --- Ours-vs-TacMap (FPS + GPU memory, ours=elastomer_taxel @ n_sensors=5) ---
# python scripts/plot/plot_tacmap_comparison.py \
#   --ours-files $TACTILE_SENSOR_DATA \
#   --tacmap-csv $REF_DIR/tacmap_benchmark.csv \
#   --sensor elastomer_taxel --n-sensors 5 \
#   --figsize 2 4.6 \
#   --output $OUT_DIR/tacmap_comparison.png

# # --- Ours-vs-HydroShear (FPS only — CSV carries no GPU memory column) ---
# python scripts/plot/plot_hydroshear_comparison.py \
#   --hydroshear-csv $REF_DIR/hydroshear.csv \
#   --n-sensors 5 \
#   --figsize 2 3 \
#   --output $OUT_DIR/hydroshear_comparison.png
# --- Ours-vs-HydroShear (FPS only — CSV carries no GPU memory column) ---
python scripts/plot/plot_hydroshear_comparison.py \
  --hydroshear-csv $REF_DIR/hydroshear.csv \
  --n-sensors 5 \
  --figsize 2 3 \
  --output $OUT_DIR/hydroshear_comparison.png

# --- Ours-vs-TacSL (FPS vs #envs, log-log; ours=kinematic_taxel from the TacSL
# comparison sweep, lined up against TacSL's 100/10000-taxel resolutions) ---
# Only the 5090 set has this sweep so far, hence the guard rather than an unconditional call.
if compgen -G "$TACSL_COMPARISON_DATA" >/dev/null; then
  python scripts/plot/plot_tacsl_comparison.py \
    --tacsl-csv $REF_DIR/tacsl.csv \
    --ours-logs $TACSL_COMPARISON_DATA \
    --sensor kinematic_taxel \
    --figsize 6 4.5 \
    --output $OUT_DIR/tacsl_comparison.png
else
  echo "skipping TacSL comparison: no files match $TACSL_COMPARISON_DATA"
fi
