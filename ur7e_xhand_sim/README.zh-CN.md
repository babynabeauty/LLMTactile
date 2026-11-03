# v1 全流程命令

在仿真机的仓库根目录执行。方案见 `PLAN_v1.md`。

## 0. 准备

```bash
uv pip install --python .venv/bin/python --no-deps -r ur7e_xhand_sim/residual_rl/requirements.txt
```

```bash
.venv/bin/python -m pytest -q ur7e_xhand_sim/residual_rl
```

## 1. 示范：每个物体 × 手型一条

参数文件，如 `configs/v1/demos/cuboid_pinch.json`：

```json
{
  "object_id": "cuboid",
  "object_spec": {"shape": "cuboid"},
  "hand_shape": "pinch",
  "palm_offset_m": [0.0, 0.0, 0.0],
  "palm_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
  "open_hand_rad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "closed_hand_rad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "grip_gain": 1.0,
  "close_seconds": 1.0
}
```

`hand_shape`：`pinch` / `tripod` / `wrap`。`palm_offset_m`：对准时手掌相对物体中心的偏移（世界系）。手部 12 个关节按 `build_assets.HAND_JOINTS` 的顺序。调参数直到退出码为 0；输出目录须不存在，每个组合只留一个成功目录。

```bash
.venv/bin/python ur7e_xhand_sim/v1_demo.py --spec configs/v1/demos/cuboid_pinch.json --output outputs/v1/demos/cuboid_pinch
```

## 2. 冒烟测试：用示范本身当 base

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_bases --libraries outputs/v1/demos/*/bases.json --output outputs/v1/smoke/experiment.json
```

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_train --experiment-json outputs/v1/smoke/experiment.json --output outputs/v1/smoke/train --smoke
```

## 3. 采集 base：BasePolicyV1 + Astra

`configs/v1/collect.json`。`unseen` 每个没见过的物体 × 手型写一行，`demo` 用最接近的见过物体、同手型的示范：

```json
{
  "demos": ["outputs/v1/demos/cuboid_pinch/demo.json"],
  "unseen": [{"demo": "outputs/v1/demos/cuboid_pinch/demo.json", "object_id": "cube45",
              "object_spec": {"shape": "cube", "side_m": 0.045}}],
  "train_placements": 30, "test_placements": 10, "unseen_placements": 10,
  "xy_range_m": 0.10, "yaw_range_deg": 10, "seed": 1
}
```

先试跑 5 次（耗时见输出的 `wall_s`），再并行跑完。同一输出目录会跳过已完成的任务，加 `--retry-errors` 重跑出错的。

```bash
.venv/bin/python ur7e_xhand_sim/v1_collect.py --config configs/v1/collect.json --output outputs/v1/collect --limit 5
```

```bash
.venv/bin/python ur7e_xhand_sim/v1_collect.py --config configs/v1/collect.json --output outputs/v1/collect --workers 4
```

## 4. 标注 S/R/U

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_label --collection outputs/v1/collect --output outputs/v1/labels
```

输出 `label_counts.json`、每个物体 × 手型的 base 库、`experiment_train.json`、`experiment_test.json`。

## 5. 训练：3 种子 × 3 方法 × 2 阶段

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_train --experiment-json outputs/v1/labels/experiment_train.json --dry-run
```

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_train --experiment-json outputs/v1/labels/experiment_train.json --output outputs/v1/train --suite
```

已完成的阶段会跳过；中断后先删掉没跑完的 `seed*_*_stage*` 目录，再跑同一命令。

## 6. 评估：测试 base（含没见过的物体）

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_eval --experiment-json outputs/v1/labels/experiment_test.json --output outputs/v1/eval/base_only --base-only
```

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_eval --experiment-json outputs/v1/labels/experiment_test.json --output outputs/v1/eval/fixed_gains --fixed-gains 1 2 3
```

```bash
for s in 1 2 3; do for m in tactile no_tactile tactile_no_level; do
  .venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_eval --experiment-json outputs/v1/labels/experiment_test.json \
    --output outputs/v1/eval/seed${s}_${m} --method $m \
    --checkpoint outputs/v1/train/seed${s}_${m}_stage2/final_model.zip
done; done
```

结果在各输出目录的 `summary.json`。
