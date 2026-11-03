# v1 全流程命令

在仿真机的仓库根目录执行。方案见 `PLAN_v1.md`。

## 0. 准备

```bash
uv pip install --python .venv/bin/python --no-deps -r ur7e_xhand_sim/residual_rl/requirements.txt
```

```bash
.venv/bin/python -m pytest -q ur7e_xhand_sim/residual_rl
```

示范位置改到了 (−0.50, −0.133) m。先渲染一次，在 `outputs/platform_preview/contact_sheet.png` 里确认三个相机都拍得到物体：

```bash
.venv/bin/python ur7e_xhand_sim/preview_environment.py --output outputs/platform_preview
```

## 1. 示范：每个物体（1×）× 手型一条

参数文件命名为 `物体_手型.json`，如 `configs/v1/demos/cuboid_pinch.json`：

```json
{
  "object_id": "cuboid",
  "object_spec": {"shape": "cuboid", "size_m": [0.045, 0.06, 0.045]},
  "hand_shape": "pinch",
  "palm_offset_m": [0.0, 0.0, 0.0],
  "palm_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
  "open_hand_rad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "closed_hand_rad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "grip_gain": 1.0
}
```

四种物体：

| `object_id` | `object_spec` |
|---|---|
| `cuboid` | `{"shape": "cuboid", "size_m": [0.045, 0.06, 0.045]}` |
| `hex_prism` | `{"shape": "hex_prism", "across_flats_m": 0.045, "height_m": 0.045}` |
| `cylinder` | `{"shape": "cylinder", "radius_m": 0.0225, "height_m": 0.09}` |
| `sphere` | `{"shape": "sphere", "radius_m": 0.0225}` |

`hand_shape`：`pinch` / `tripod` / `wrap`。`palm_offset_m`：对准时手掌相对物体中心的偏移（世界系）。手部 12 个关节按 `build_assets.HAND_JOINTS` 的顺序。调参数直到退出码为 0；输出目录须不存在，每个组合只留一个成功目录。每种手型至少要有 2 种物体成功。

```bash
.venv/bin/python ur7e_xhand_sim/v1_demo.py --spec configs/v1/demos/cuboid_pinch.json --output outputs/v1/demos/cuboid_pinch
```

尺寸检查：全部示范成功后，每个参数文件在 0.85× 和 1.15× 上各跑一次，输出放在 `size_check/`，不当示范。

```bash
for f in configs/v1/demos/*.json; do b=$(basename $f .json); for s in 0.85 1.15; do
  .venv/bin/python ur7e_xhand_sim/v1_demo.py --spec $f --scale $s --output outputs/v1/size_check/${b%_*}_x${s/./}_${b##*_}
done; done
```

```bash
grep -l 'before close' outputs/v1/size_check/*/score.json
```

列出的目录名（如 `cuboid_x115_wrap`）写进第 3 步配置的 `skip`；其他失败不用管。

## 2. 冒烟测试：用示范本身当 base

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_bases --libraries outputs/v1/demos/*/bases.json --output outputs/v1/smoke/experiment.json
```

示范验收（零残差、50 g、摩擦 0.8 重放，每条都应成功）：

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_eval --experiment-json outputs/v1/smoke/experiment.json --output outputs/v1/smoke/replay --split train --base-only --materials '[[0.05, 0.8]]' --levels 4
```

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_train --experiment-json outputs/v1/smoke/experiment.json --output outputs/v1/smoke/train --smoke
```

## 3. 采集 base：BasePolicyV1 + Astra

`configs/v1/collect.json`：

```json
{
  "demos": ["outputs/v1/demos/cuboid_pinch/demo.json", "outputs/v1/demos/cuboid_tripod/demo.json"],
  "scales": [0.85, 1.15],
  "skip": [],
  "unseen": [
    {"object_id": "cube45", "near": "cuboid", "object_spec": {"shape": "cube", "side_m": 0.045}},
    {"object_id": "cylinder35", "near": "cylinder", "object_spec": {"shape": "cylinder", "radius_m": 0.0175, "height_m": 0.07}},
    {"object_id": "cuboid120", "near": "cuboid", "object_spec": {"shape": "cuboid", "size_m": [0.054, 0.072, 0.054]}}
  ],
  "train_placements": 10, "test_placements": 4, "edge_placements": 3, "unseen_placements": 10,
  "xy_range_m": 0.10, "edge_range_m": 0.15, "yaw_range_deg": 10, "seed": 1
}
```

- `demos`：列出所有成功示范。
- `scales`：每个示范物体缩放后也采训练、测试位置，不采边缘位置。
- `skip`：第 1 步尺寸检查中 close 前已接触的组合；其他抓取失败不跳过。
- `unseen`：`near` 填最接近的见过物体，该物体的每种手型都会测。

物体离桌边不到 2 cm 时，规划会直接报错。

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

输出 `label_counts.json`（含回放与采集不一致而剔除的数目）、每个物体 × 手型的 base 库、`experiment_train.json`、`experiment_test.json`。

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

## 协调接近的两指示范（2026-11-03）

按更新后的用户要求，`v1_demo.py` 不再先在原地完成手型：从初始张开姿态出发，
接近路径上逐步插值到轻微闭合的 `open_hand_rad`，到预抓取位置后才进入 `close`。
`close_translation_m` 可指定正式闭合时的手掌位移（世界坐标系，最长 0.1 m）；
缺省为零。`palm_offset_m` 指的是 **close 开始前的预抓取位置**。
当前长方体两指参数在正式抓取时下降 0.05 m；这一阶段手指完成主要闭合。
BasePolicyV1 同步读取此位移；小模型仍从 close 开始接管手指残差。

```bash
.venv/bin/python ur7e_xhand_sim/v1_demo.py \
  --spec configs/v1/demos/cuboid_pinch.json \
  --output outputs/v1/demos/cuboid_pinch
.venv/bin/python ur7e_xhand_sim/replay_demo_video.py \
  --demo outputs/v1/demos/cuboid_pinch \
  --output outputs/v1/demos/cuboid_pinch/video
```

输出目录必须不存在。视频由独立的关节指令回放生成，不重新运行 IK，也不移动物体。
示范导出新增滑移验收：test_lift 起物体在手掌系中的最大位移不得超过 5 mm。

## 已采集 base 的零残差回放视频

使用实际 `V1VecEnv` 评估并录制单条 base，原始数据的训练/测试划分不变：

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.v1_replay_video \
  --collection outputs/v1/BASE_cuboid_v1_batch01 \
  --run-id cuboid_pinch_train_000 \
  --output outputs/v1/BASE_cuboid_v1_batch01/zero_residual_replay/cuboid_pinch_train_000
```

每条使用一个独立进程，输出目录必须不存在。按原始物体姿态重放接近前缀，随后采用
V1 标准的闭合前状态恢复（速度清零），以 60 Hz 控制、300 Hz 物理步执行零残差。
质量固定 50 g、摩擦 0.8、不可碎，保留轨迹记录的增益。视频含三相机及手部近景，
15 fps、原速播放，在自动重置前记录终止画面；评估失败会按标准提前终止。

输出 `replay.mp4`、`validation.json`、`frames.jsonl`、来源哈希和关键截图。
`validation.json` 中的 `episode_metrics` 是同一次回放的 V1 指标；成功率不能与仅检查
抬起保持的采集评分混用。此命令不执行残差修复搜索，也不写入正式 S/R/U 库。
