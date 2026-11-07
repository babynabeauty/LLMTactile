# UR7e + XHand1 场景

使用同一台 UR7e、XHand1、导入的桌子及原有前视、左视、腕部三个相机。
`--task` 选择独立场景；`grasp` 是原有默认值。两种插入场景以 300 Hz 物理、
60 Hz 控制执行，不改现有抓取示范、base 数据或训练配置。

| task | 可抓取物体 | 初始状态 | 固定目标 |
|---|---|---|---|
| `grasp` | 原有 cube/cuboid/hex_prism/cylinder/sphere | 桌上 | 抓起并保持 |
| `tube_insert` | UniVTAC 试管，直径 20 mm、长 130 mm | 原始 50 mm 支架中竖直放置 | 100 mm 倾斜孔座，轴倾斜 30° |
| `usb_insert` | ContactWorld USB，约 12.17 × 30 × 120.96 mm | 平放桌上，需先抓起 | 原始插座，保留矩形孔和内部舌片 |

尺寸为源资产的米制尺寸，未放大。初始抓取区中心 XY 为 (-0.50, -0.1333) m，
目标座原点 XY 为 (-0.50, -0.36) m；桌面 Z=0.20 m。USB 的坐标原点在插入端，
因此平放时原点不等于物体中心。试管沿目标孔轴插入，USB 需相对插座绕 Z 转 180°，
以错开插头内部塑料片和插座舌片。

## RGB 预览与静置检查

在仓库根目录执行：

```bash
.venv/bin/python ur7e_xhand_sim/preview_environment.py \
  --task tube_insert --output outputs/insertion_scenes/tube_insert

.venv/bin/python ur7e_xhand_sim/preview_environment.py \
  --task usb_insert --output outputs/insertion_scenes/usb_insert
```

每个目录生成：

- `scene_preview.png`：机器人和桌子的全景，以及任务资产近景。
- `task_rgb.png`：任务资产近景；仅用于检查，不替换三个策略相机。
- `scene_front.png`、`scene_left.png`、`wrist.png`：原有相机 RGB。
- `overview.png`、`contact_sheet.png`：环境全景与多视图。
- `validation.json`：末段静置稳定性、材料读数、几何目标与状态，以及引擎实际 SDF 的插入通道采样检查。

预览会真实运行 300 个物理步（1 秒），检查物体仍在桌面或支架上，末段约 0.18 秒的位移小于 0.5 mm、转角小于 1°。
同时用真实碰撞网格在实际 SDF 中检查完整插入深度范围，USB 另检查错误朝向受到阻挡。
有限采样检查不等于连续轨迹或机器人完整任务的成功证明。
这些图片是环境检查结果，不是机器人成功执行抓取插入的示范。

## 启动场景桥接

```bash
.venv/bin/python ur7e_xhand_sim/server.py \
  --task tube_insert --output outputs/insertion_scenes/tube_live --port 8767

# 停止上一个服务后切换 USB，或使用另一个端口。
.venv/bin/python ur7e_xhand_sim/server.py \
  --task usb_insert --output outputs/insertion_scenes/usb_live --port 8767
```

服务器绑定 127.0.0.1。插入场景不传 `--object-spec-json`；默认质量为试管 30 g、
USB 50 g，这是仿真设定，可用原有 `--cube-mass-g` 参数覆盖。
`POST /describe` 返回场景、目标位姿和控制协议；`POST /observe` 返回现有观测；
`POST /evaluate` 的 `insertion` 字段返回插入深度、横向偏差、姿态偏差、几何状态、
物体与孔座的接触。`/reset`、`/step`、`/move`、`/hand` 提供同步场景控制。

```bash
curl -s -X POST http://127.0.0.1:8767/evaluate \
  -H 'Content-Type: application/json' -d '{}'
```

`step` 仍接收现有归一化关节动作；`move` 接收世界系 TCP 位移 `delta`（单次不超过
5 cm）与控制步数 `steps`（1–10）；`hand` 接收 `closure`（0–1）与 `steps`。
同步服务在请求推进时仿真才前进。

## 接入范围与几何判据

本次接入资产、物理碰撞、摆放、重置、相机、场景控制和插入几何评估。
原有 `v1_demo`、`BasePolicyV1`、S/R/U 标注、残差训练的任务仍是抓取，
不能直接用于这两个插入场景；桥接会拒绝插入场景调用原有 `trial_start` 或 `--online`。
尚需针对新任务设计和采集抓取→搬运→对准→插入示范，再连接对应的策略技能。

`geometric_success` 是瞬时几何状态：试管进入 35–45 mm 且杆身留在孔内；
USB 进入 16–18 mm，方向正确，壳体与内部舌片不相交。它不代表机器人已经完成
抓取、稳定插入或释放，不作为现有抓取 S/R/U 成功标签。

固定支架和孔座保留源网格的非凸碰撞，关闭凸包化、网格简化和自动包覆。
USB 原始插头网格有开放表面，因此碰撞使用主体凸壳、四片金属壳壁和内部塑料片；
外观保持原始网格，卡扣、柔性片和主体内部空腔被简化。这些接触细节尚未对实物标定。
具体尺寸、碰撞参数、来源、哈希及许可说明见 `sources/insertion/manifest.json`。
运行时资产全部位于仓库内，不依赖外部 UniVTAC/ContactWorld 目录或 USD/Isaac Sim。
