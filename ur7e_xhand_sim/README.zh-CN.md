# UR7e（850 mm）＋XHAND1 触觉仿真

沿用当前 Tactile Genesis / Genesis 1.4.1 环境的独立场景。机器人包含 UR7e 六关节、XHAND1 右手十二关节和五指各 120 个三轴触觉测点。保留原 Panda＋Wuji 场景，启动端口为 8767。

## 启动

```bash
cd /home/sai/Documents/ChatGPT/Shiqi/tactile-genesis
bash ur7e_xhand_sim/launch.sh
```

看到 `UR7E_XHAND_READY` 后打开 <http://127.0.0.1:8767/>。页面可单步、暂停、移动手腕和调整手指关节。手指开合是调试姿态，不是已验证的抓取方案。控制步长 25 ms，物理步长 5 ms；这两项不是实机采样率结论。

手部近景同时展示掌心、手背、拇指侧和小指侧。四台观察相机每次渲染都根据 `right_hand_link` 实际位置与朝向更新，移动与转腕后保持相对手掌的观察方向；手靠近桌面时，低于桌面的视点自动抬高并保持观察距离。整体场景相机保持固定。这些是仿真观察视角，不代表四台真实硬件相机。`observe.hand_rgb_paths` 返回四张图片路径，原 `hand_rgb_path` 和 `/hand.jpg` 保留为拇指侧视角。

外部及腕部相机使用用户提供的 `sources/cameras/cfg_reg_xhand.yaml` 中 `environment.realtime_obj_pc` 配置，由 `cameras.py` 仅读取相机字段。前方为 `external`（`scene_front`，兼容 `scene`），第二台为 `external_2`（`scene_left`，文件定义为左侧而非后方；`scene_rear.jpg` 仅为兼容别名）。原 RaiSim URDF 底座位于 Z=0.9 m，当前底座为 Z=0.2 m，因此外部相机平移 `[0,0,-0.7]` m，保留已转换过的 XY 和四元数，不能再重复翻转 XY。光学坐标 +Z 向前、+Y 向下，转换成 Genesis 的 lookat/up。腕部平移与 xyz Euler 从该 YAML 读取，映射到 `right_hand_link`，随手移动和旋转；此映射尚未独立进行实机复标定。

图像输出为外部各 1000×750、腕部 640×362。保留 YAML 的垂直 FOV，按水平/垂直 FOV 的比例选择输出尺寸，水平 FOV 仅有整像素取整误差。YAML 的 96×72、80×60 是点云射线网格，不用作 VLM 图像分辨率。其 near/far 用于深度点云范围，前方 ROI 重建会按该范围过滤；RGB 远裁剪为 20 m，避免把机械臂显示成截断的几段。界面仍保留一排四个手部近景。`describe.cameras` 和 `camera_source` 提供实际参数、源文件哈希及坐标转换；`observe.overview_rgb_paths`、`wrist_rgb_path` 返回图片路径。

默认六关节角抄录自用户于 2026-10-24 提供的 UR 示教器“关节位置”照片：`[0.00, -89.95, +89.96, -89.96, -89.95, -44.98]` 度，顺序为底座、肩、肘、腕 1、腕 2、腕 3。代码转换成弧度后直接设置初始位置和控制目标，重置也恢复这组数值。上臂接近竖直、前臂接近水平、腕部向外延伸、法兰和掌面朝下；源模型坐标中手指朝 -X。桌面、方块和交互移动范围放到 -X 工作侧，观察视角仍为机械臂在右、手指向左。

用户已确认实机使用同一套转接件。手掌安装变换及传感器坐标定义保持原模型设置；当前使用标称 URDF 运动学，相机、底座世界位置和实机 TCP 尚未标定。示教器上方 TCP 的 XYZ 和 RX/RY/RZ 未用作关节角或手掌位姿，以免混淆 TCP 偏置与参考系。旧的照片估计记录保留在 `outputs/ur7e_photo_direction_20261024/` 和 `outputs/ur7e_photo_posture_20261024/`；示教器角度版本检查记录在 `outputs/ur7e_measured_home_20261024/`。

```bash
.venv/bin/python astra_bridge/client.py --url http://127.0.0.1:8767 describe
.venv/bin/python astra_bridge/client.py --url http://127.0.0.1:8767 observe --json '{"full":true}'
.venv/bin/python astra_bridge/client.py --url http://127.0.0.1:8767 close
```

`step` 接收归一化关节位置目标：`q = midpoint + action * half_range`。`describe` 给出关节顺序、限位和单位；手指排列严格使用硬件 SDK 的 ID 0..11。当前没有实现残差策略、训练奖励或随机化。

## 可选硅胶近似软块

运行 `bash ur7e_xhand_sim/launch.sh --scenario small_clutter --silicone` 可在原场景旁新增粉色 45 mm FEM 弹性块，原橙色抓取目标和六个彩色干扰物保留。不带 `--silicone` 时仍是原刚体场景。

该原型使用隐式 FEM、`linear_corotated` 模型和 Legacy 刚柔耦合，E=100 kPa、nu=0.45、rho=1100 kg/m³（名义质量约 100.24 g）。参数未经真实硅胶标定，仅适合先验证弹性形变；不能据此宣称精确模拟硅胶的大变形或黏弹性。控制周期仍为 25 ms，启用软块时每个 5 ms 物理步默认细分为 20 个 0.25 ms 子步（`--silicone-substeps` 可指定）。这取代了初版的 5 个 1 ms 子步，不能与原单子步刚体实验直接视为同一动力学设置。

`describe.silicone` 返回配置，`evaluate.silicone` 返回实际顶点包围盒、速度和有限性检查；这些诊断数据不加入策略观测。当前 XHand 的 600 测点仍只覆盖原刚体触觉通道，**没有接入 FEM 接触力**。既有 S8V4 任务目标仍是橙色刚体木块，不能直接将其成功判据用于粉色软块。新增软块不自动启动抓取。

落桌及重置验证记录位于 `outputs/silicone_block_20261024/`，其中 `check_scene.py` 从项目根目录运行，`validation.json` 为独立场景结果，`live_validation.json` 为当前 HTTP 服务的运行检查。首次尝试的显式求解器不稳定，已改为隐式求解；失败日志保留用于追溯。

## 独立 VLM agent → bridge → 仿真

`run_s8v4.py` 调用原项目 `/home/sai/zsq/vlm-policy/codex_policy.py` 的 `CodexAppServerPolicy`，通过 app-server JSON-RPC API 创建独立策略会话，使用 `gpt-6-astra`。当前对话只启动 runner、检查结果和维护实现；抓取 ROI、候选偏移、技能选择及完成/中止判断来自独立 agent 的结构化响应。网络失败即停止，不回退到当前对话手动选择动作。

链路为：Genesis 图像与关节/触觉观察 → 独立 VLM agent → `SkillAct` → HTTP bridge → 技能执行器 → Genesis → 新观察。物理在模型思考期间暂停；技能执行期间由 PD 与 `RuleSupportController` 驱动。仿真执行器复用项目的 `XHandContactPrior`、`SkillAct`、规则支持控制器，但名义手腕位姿由可见 RGB-D 与手指几何构造，是 **S8V4 的 Genesis 适配**，不是已验证的实机抓取示范。未使用训练后的触觉残差模型。

每次模型调用仅输入三张图：前方 `scene_front`、左侧外部 `scene_left`、手腕 `wrist`。四个手部近景保留在网页及仿真记录中，不发送给 Astra。`locate` 的 ROI 必须属于 `scene_front`；尺寸从每轮 `stack.image_sizes_px` 读取，RGB-D 重建和边界校验同样使用实际相机分辨率。每次观测的 `view_*/` 保存图片与三台前/左/腕相机的 `camera_params.json`，agent 的 `codex_inputs/` 保存实际发送的三张图。旧试验的相机和输入记录保持原样。

先按上面的启动命令运行 bridge。另一个终端执行：

```bash
cd /home/sai/Documents/ChatGPT/Shiqi/tactile-genesis
# 只检查 agent API，不重置或驱动仿真；每次换一个新 label。
.venv/bin/python ur7e_xhand_sim/run_s8v4.py --label api_check_NEW --check-connection
# API 初始化成功后，runner 才创建并重置新的仿真试验。
.venv/bin/python -u ur7e_xhand_sim/run_s8v4.py --label agent_grasp_NEW
```

默认 `--connection direct`，与原项目 `B2D_CODEX_DIRECT=1` 一致。`--connection inherit` 使用环境代理，`--connection project-proxy` 显式使用项目配置的代理；不会自动启用可能失效的旧代理。账号沿用当前 `~/.codex` 登录，模型调用不改成 API-key 计费。

bridge 绑定 `127.0.0.1:8767`，POST JSON 返回 `{ok,result}` 或 `{ok:false,error}`：

| 接口 | 用途 |
| --- | --- |
| `/s8v4_start` | runner 用唯一 `label` 创建试验；已有活动试验时拒绝 |
| `/s8v4_observe` | 返回带 `observation_id` 的多视角、状态和 GenDex 候选 |
| `/s8v4_act` | 接收 agent 返回的 `SkillAct`；校验观察版本、技能阶段与候选偏移后执行 |
| `/s8v4_score` | 控制结束后读取独立评价，物体真值不作为策略输入 |

试验活动期间拒绝网页的 `step/move/hand/reset` 手动运动。机器人自碰撞或碰桌时停止。抓取成功要求方块底面高于桌面至少 5 cm，并连续保持 3 s；触觉非零或模型说“成功”都不能替代物理验收。

审计记录分开保存：`outputs/s8v4_agent/<label>/` 包含独立模型每轮响应、API thread/turn ID、usage 和输入图像；`outputs/s8v4_genesis/<label>/` 包含执行动作、物理过程和 `score.json`。历史 `dialogue_*`、`thumb_preshape_*` 试验由当前对话选择动作，不属于独立 agent 验证，保留作失败排查记录。所有质量、摩擦与触觉阈值仍未经过实机标定。

## 规则采集成功示范

`rule_demo.py` 是独立的确定性仿真教师，经 bridge 执行。它用前方 RGB-D 提取木块可见表面，用关节反馈补偿实际手掌位置；预成形后下降，拇指和食指连续接触才增加预紧，再试抬 2 cm、继续抬升并保持 3 s。动作通过 IK 和原有 PD 执行，不附着物体、不直接改写物体位姿，保留质量、摩擦、力矩限幅及碰撞设置。物体真值仅供独立评价，求解器接触用于自碰撞／碰桌停止。

启动 bridge 后运行（每次使用新 label）：

```bash
.venv/bin/python -u ur7e_xhand_sim/collect_rule_demo.py --label rule_demo_NEW --options '{"preload_rad":0.15}'
```

`/rule_demo_start` 创建试验，`/rule_demo_advance` 每次推进最多 12 个控制步；活动期间拒绝其他试验及手动运动。采集器失败返回非零退出码。结果保存在 `outputs/rule_demos/<label>/`：

- `control.jsonl`：25 ms 步长的动作前后观测、18 关节目标、实际位置、五指三轴触觉与期望手掌位姿。
- `evaluation.jsonl`、`score.json`：独立物理验收；旋转后木块最低点离桌至少 5 cm，连续保持 3 s 才算成功。
- `stages.json`、`view_*/`：阶段关键帧、手掌及 tool0 位姿、RGB-D、相机参数；`frames/` 每 0.3 s 保存前方／左侧／手腕图像。
- `manifest.json`：教师参数、相机来源和源代码哈希。仅成功时生成 `reference.json`，包含观测锚点及 open/light/closed 手形，标记 `simulation_verified`。

2026-10-24 的 `teacher_pinch_v2_20261024` 已通过：1275 个控制步，保持 3 s 的最低离桌高度 71.80 mm，最高离桌高度 80.83 mm。该目录另有三视角视频 `demonstration_three_views.mp4`。第一次 `teacher_pinch_v1_20261024` 因保持阶段滑落失败，保留记录且不生成成功参考。成功版本只将接触后的预紧增量从 0.05 rad 改为 0.15 rad。

这条数据是仿真规则教师示范，不是原项目 `xhand-icl-v1` 实机数据格式。现已通过显式 `--reference` 接入；不传该参数仍保留旧实验路径，不能认为旧试验收到了这条示范。

## 带成功示范的四场景 S8V4 测试

```bash
# 自动依次启动四个场景；运行前 8767 端口应空闲。
.venv/bin/python -u ur7e_xhand_sim/run_four_s8v4.py \
  --label demo_context_four_NEW \
  --reference outputs/rule_demos/teacher_pinch_v2_20261024/reference.json
```

场景为原位、小幅平移 `[0.03,0.02,0]` m（3.61 cm）、较大平移 `[0.15,0.10,0]` m（18.03 cm）及原位周围三个彩色方块、三个彩色球。所有位移均为仿真世界坐标。物体只在场景构建时设置初始位置；控制中没有物体位姿写入。每组只运行一次，使用相同模型、示范、质量、摩擦、摄像头和控制参数。每组独立 Astra 会话，不传入上一组结果。最后一个场景保持在网页上。

`success_context.py` 校验成功记录哈希，将 open/aligned/light_contact/closed/test_lifted/lifted/held 七个关键帧及其三视角、实际关节、发送目标、稀疏三轴触觉载入上下文。首轮为 21 张历史图与 3 张实时图，后续仅追加 3 张实时图，历史保留在同一会话。完整 1275 步记录不全部铺入模型文本；执行器读取完整 close/preload 段。`api_input_*.json` 与 `codex_inputs/` 可核验真实发送的文本和图像，模型 metadata 的 demonstration 不再为 null。

`verified_s8v4.py` 使用示范锚点到期望手掌的相对位姿，按当前 RGB-D 配准平移，保留示范手掌朝向。当前移植版的完整链路为：示范上下文 → Astra ROI → RGB-D → GenDex CMap-CVAE 候选评分 → Astra 选择技能／偏移 → 示范闭合轨迹与物理 IK/PD → 原项目 RuleSupportController。提供 GenDex 排序候选及明确标注的零偏移示范基线，选择由 Astra 完成。局部控制使用规则教师已验证的平滑路径与有界位置积分补偿。

该链路是 UR/XHand 移植版：GenDex 对固定手形的小范围位移评分，未实现原生 Allegro 的全手形优化。平移配准范围固定为 25 cm，仅用于本次仿真；不修改实机配置的范围。抬升前要求最近五个采样均有拇指、食指触觉；关节偏差不能单独放行。碰桌、自碰撞或触碰干扰物停止。物体与干扰物真值单独写入 evaluation，不发给模型。S8V4 本身使用支撑规则，不包含训练后的 GRU。

批次目录 `outputs/s8v4_four_tests/<label>/` 保存冻结协议、源代码快照、各组日志及汇总；动作和模型记录仍分置于 `outputs/s8v4_genesis/` 与 `outputs/s8v4_agent/`。四次单例只说明这四个条件的观察结果，不能估计总体成功率。

## 资产来源与适配

- UR7e 型号由用户确认为标准 850 mm 版本。UR 官方说明 UR5e 与这款 UR7e 共用机械设计：<https://www.universal-robots.com/download/mechanical-e-series/ur5eur7e/robot-step-file-ur5eur7e-e-series/>。
- 机械臂、转接件和 D405 安装几何来自 `/home/sai/zsq/TactileTTT/ur5_xhand/ur5_xhand.urdf`。移除该文件的实验台与 0.9 m 支架，让机器人固定在本场景桌面；保留源模型的底座方向及手掌安装变换。用户于 2026-10-24 确认真实转接件与这套模型相同，并提供安装近照。
- 六个关节的运动学变换、臂部质量和惯性张量已与 UR 官方 ROS2 Description 的 `config/ur7e` 对照。固定提交与原始配置保存在 `sources/ur7e_official/`，包含许可证；这不是这台真机的个体运动学校准数据。
- 手部替换为用户交付包 `urdf/Xhand-urdf.zip` 内的右手模型，保留厂家关节限位、惯量和网格。厂家变更日志标注 v1.3。
- 触点采用交付包的 transformed JSON，直接属于每根手指 **link2** 坐标系；按 point 1..120 排序，从毫米转米，不再重复加平移。拇指采用 right_hand 文件。
- 所有导入资产及 SHA-256 在 `sources/asset_manifest.json`；`build_assets.py` 从冻结源文件生成模型，并校验来源。生成文件包含本机绝对路径，移动目录后重新运行构建脚本。
- 转接件原碰撞 OBJ 已是填满空隙的单凸包，会与拇指根部产生虚假碰撞。构建器保留详细 STL 的三个独立部件，每个部件单独使用凸包，避免填满部件之间的空隙；部件内部小孔仍是近似。自碰撞开启，不静默过滤零位碰撞。生成记录在 `generated/adapter_components_*/manifest.json`。

## 触觉与物理含义

`observe(full=True)` 输出五根手指各 `(120,3)`，顺序为拇指、食指、中指、无名指、小指：

- `force_link_N`：Genesis 在附着 link2 坐标系下的三轴估算。
- `force_sensor_N`：仅旋转到厂家传感器本体 XYZ，不加平移。T16 的位置转换为 `x'=z+5.35, y'=x, z'=y+15.86`（mm），对应向量旋转；右手 T30 只有平移，向量旋转为单位阵。
- 每点保留三轴，不向策略提供仿真的额外力矩通道。历史索引 0 为最新物理子步。
- 读数是未标定的几何／运动学近似，单位标记为名义 N；**不是硬件 raw_pressure、LSB 或标定后的真实力**。没有将资料中的标称 10 LSB/N 直接用于制造硬件读数。未实现真实传感器的低力死区、迟滞或刷新延迟。
- 探针半径暂设 1.5 mm，刚度、阻尼、剪切参数为开发初值，均可在 `describe` 检查。它们需要实测标定。切向项基于速度，尚不支持对静态剪切累积／临界滑移的真实性承诺。
- 当前场景是刚体接触；厂家提供的软胶材料参数没有被冒充为已实现的 FEM 模型。

`evaluate` 另外返回物体真值和物理求解器接触数据（接触双方、位置、力），供验收和后续奖励设计；这些数据不混入 `observe`。控制器的 PD 增益是仿真设置，力矩限幅取自源模型。`joint_force_Nm` 是内部关节受力，不能当作硬件电机测量值。

厂家“指尖触觉仿真（初版）”示例可以辅助理解点位布置，但没有照搬它的 `get_tac`：该函数直接累计接触坐标系的分量，并对 XYZ 求和；这不等于传感器本体三轴力或合力模长。

### 2026-10-24 接触存在但触觉间歇归零的修复

本次查到的是 Genesis `kinematic_tactile.py` 两个接触预筛选 kernel 漏用了
`collider_state.contact_sort_idx`。求解器裁剪接触后仅压缩逻辑索引，并不搬移
`contact_data` 的物理存储；直接遍历前 `n_contacts` 个存储槽会漏掉有效接触。
新增平移干扰场景在第 848 步有 21 条有效接触，拇指两条与食指一条分别位于物理槽
23、22、21；旧预筛选只读 0..20，导致两指候选几何都为空。现已在 geom 和 contact
两条预筛选路径中使用逻辑到物理映射，后续保存的接触编号仍为物理存储编号。

点位覆盖不是这次缺测的直接原因：旧四场景测试的原位与原位干扰场景第 855 步，
拇指均有 4 个半径 1.5 mm 的测点与木块表面相交，位置和穿入量几乎相同。
求解器给出的离散接触点不必与测点中心重合；SDF 触觉查询的是木块整个表面。
未修改厂家测点、半径、刚度、阻尼、摩擦、策略手形或抬升门槛。

独立冻结状态验证恢复新增试验第 848 步的关节和木块姿态及接触索引，不进行任何物理步进：
旧预筛选拇指／食指均为零；新预筛选恢复 4／22 个测点，静态峰值约 0.0548／0.3277 名义 N。
该验证将速度置零，所以这些数值不能冒充原始动态采样。接触／几何两条路径、恒等索引、
乱序索引、空接触、对端过滤和几何去重均通过。证据与可复现脚本：

- `outputs/tactile_contact_audit_20261024/validation.json`、`live_debug.json`、`coverage_step855.json`、`diagnosis.png`。
- `.venv/bin/python ur7e_xhand_sim/validate_tactile_contact_mapping.py`（独立离线场景，不连接 bridge）。

用户要求的单次新增试验 `demo_context_small_clutter_20261024_v1` 在修复前完成：
目标平移 `[0.03,0.02,0]` m（3.6056 cm），另加 3 个彩色方块和 3 个彩色球，
干扰物相对目标的布置与前次 clutter 相同，随目标一起平移。仍使用同一成功示范、
独立 Astra API 和前方／左侧／腕部输入。5 次模型调用、848 个控制步后，由于缺少持续
双指触觉，Astra 中止，未执行抬升，无碰桌／自碰撞／触碰干扰物记录。结果保留为失败，
不是修复后抓取验证。完整记录在 `outputs/s8v4_genesis/demo_context_small_clutter_20261024_v1/`，
含 `replay_three_views.mp4`；审计在 `outputs/s8v4_small_clutter_20261024_v1/audit.json`。
修复后的冻结状态验证不计入抓取次数；当时尚未执行修复后的抓取试验。

用户随后要求复测，同日完成一次独立 Astra 试验
`demo_context_small_clutter_fixed_20261024_v1`，同场景、同成功示范、同策略与控制参数。
8 次模型调用、1268 个控制步，完成试抬、抬升及 3 秒保持，独立物理验收成功：
保持期最低离桌 72.11 mm，最高离桌 81.34 mm；保持期 120/120 个采样均有双指触觉
超过 0.01 名义 N，没有碰桌、自碰撞或触碰干扰物记录。支撑控制器运行 420 个采样，
本次未触发额外闭合修正。与修复前的 RGB-D 锚点完全相同；第 848 步手掌位置相差
约 0.08 微米，但双指峰值从 `[0,0]` 恢复为 `[0.05482,0.32769]` 名义 N。
这进一步支持接触索引错误的诊断，单次通过不代表总体成功率。
协议、源码快照、审计与前后曲线保存在 `outputs/s8v4_small_clutter_fixed_20261024_v1/`；
过程和三视角视频在 `outputs/s8v4_genesis/demo_context_small_clutter_fixed_20261024_v1/`；
独立模型输入和响应在 `outputs/s8v4_agent/demo_context_small_clutter_fixed_20261024_v1/`。

单独启动该场景使用 `bash ur7e_xhand_sim/launch.sh --scenario small_clutter`。
`POST /tactile_debug` 为只读诊断，返回原始／映射接触索引和测点坐标，不发送给策略。

## 验收

### 摩擦／质量参数与固定轨迹回放

`Sandbox` 支持 `cube_density`（kg/m³）或 `cube_mass_kg`（kg，二者互斥），以及
`cube_friction`、`robot_friction`。默认仍为 45 mm 立方体、500 kg/m³、45.5625 g、
摩擦 0.8。质量输入先换算为密度，再创建刚体，使质量与惯性通过相同几何构建；
不在运动期间改写质量或物体位姿。`describe.materials` 同时提供配置和引擎质量读数。

```bash
# 以下均为启动新仿真服务；端口须空闲。
bash ur7e_xhand_sim/launch.sh --cube-density 500 --cube-friction 0.2 --robot-friction 0.2
bash ur7e_xhand_sim/launch.sh --cube-mass-g 100 --cube-friction 0.8 --robot-friction 0.8

# 六组独立进程、每组一次，输出目录必须不存在。
.venv/bin/python -u ur7e_xhand_sim/replay_material_sweep.py --output outputs/material_replay_NEW
.venv/bin/python ur7e_xhand_sim/analyze_material_replay.py outputs/material_replay_NEW
# 可在测试期间启动只读实时预览 http://127.0.0.1:8768/。
.venv/bin/python ur7e_xhand_sim/material_replay_preview.py --output outputs/material_replay_NEW
```

摩擦组为 `0.2 / 0.5 / 0.8`，质量均为 45.5625 g；质量组为 `50 / 100 / 200 g`，
摩擦均为 0.8。Genesis 刚体接触取双方系数的最大值，因此摩擦组同步设置机器人和木块，
桌面固定为 0.8；手指—木块分别得到目标系数，木块—桌面始终为 0.8。
每组从 `cube.get_mass()`、材质密度、几何摩擦读回参数，并按 `contact_sort_idx`
映射实际接触记录，逐步断言手指—木块／木块—桌面的真实接触系数。

场景为示范采集时的原位、无干扰物。回放 `teacher_pinch_v2_20261024/control.jsonl`
中完整 1275 步、18 关节的 `sent_target_rad`，25 ms 一步，保留原关节 PD；
不重新计算 IK、不调用 Astra，不使用触觉门槛或支撑规则修改动作。
触觉只记录，因此可观察固定闭合轨迹在不同物理参数下的表现。若发生机器人碰桌或自碰撞则停止。
成功标准仍为最终 120 个保持采样全部离桌至少 5 cm，而非仅凭触觉非零。

每组保存 `material_verification.json`、`control.jsonl`、`evaluation.jsonl`、`score.json`、
三视角图片和视频；`trajectory_tactile.npz` 含完整发送／实际关节序列及
`(1275,5,120,3)` 三轴触觉。批次 `protocol.json`、源码快照、`audit.json` 和
`comparison.png` 保存参数、同轨迹核验、独立物理验收及曲线。

2026-10-24 批次 `outputs/material_replay_20261024_v1/` 六组全部完成参数与同轨迹核验：

| 接触摩擦 | 引擎质量（g，取整显示） | 密度（kg/m³） | 3 秒保持结果 | 保持最低离桌（cm） |
| --- | --- | --- | --- | --- |
| 0.2 | 45.5625 | 500.00 | 通过 | 7.62 |
| 0.5 | 45.5625 | 500.00 | 通过 | 7.68 |
| 0.8 | 45.5625 | 500.00 | 通过 | 7.18 |
| 0.8 | 50 | 548.70 | 通过 | 7.16 |
| 0.8 | 100 | 1097.39 | 通过 | 7.11 |
| 0.8 | 200 | 2194.79 | 未通过：抬起后掉落 | 回到桌面 |

200 g 最高离桌 6.94 cm，连续超过 5 cm 约 2.025 秒；28.7 秒双指触觉归零，
28.75 秒已无手—木块物理接触，随后落回桌面。五个成功组的保持期均有 120/120 个
双指触觉有效采样。六组均无机器人碰桌／自碰撞，全部 1275 步发送目标与示范相符，
最大数值误差约 2.22e-16 rad。每组一次，只验证本次固定轨迹，未运行在线触觉调整。
`replays.html` 可在实时预览服务中打开以播放六组录像。

当前 `Rigid.friction` 允许范围为 `[0.01,5.0]`；求解器混合系数也有 `0.01` 下限。
若测试更低摩擦，可同时设置 `--cube-friction 0.01 --robot-friction 0.01`；
物理材质参数会生效，颜色／贴图／材质名称不会自动映射成摩擦系数。

同日新增两组，批次 `outputs/low_friction_stronger_20261024_v1/`：

- `friction_001`：45.5625 g、接触摩擦 0.01、原增益和完整原目标轨迹。未有效抬起，
  最大最低点离桌仅 0.064 mm，保持失败；实际接触摩擦已核验。
- `mass_200g_stronger`：200 g、摩擦 0.8，完整原目标轨迹；在第 796～855 步预紧阶段
  将拇指和食指六关节 Kp 从 3 平滑增至 6、Kv 从 0.08 增至 0.08×√2，之后保持。
  其余关节增益及所有力矩上限不变。成功保持 3 秒，最低离桌 69.99 mm、最高 79.36 mm，
  保持期 120/120 个采样有双指触觉。没有机器人碰桌或自碰撞。

用预紧末尾 20 个控制采样比较同为 200 g 的原／加强夹持，求解器逐接触力按手指
先向量相加再取模长：拇指平均 4.49→8.43 N，食指平均 4.86→8.50 N，增力已核验。
这是物理接触合力，不是触觉测点峰值，也不意味着触觉幅值会同比增加。
两组都没有 VLM、IK 重规划或触觉反馈修正，均只运行一次。
源代码、增益计划、材料读回与同轨迹检查见协议、`manifest.json`、`audit.json`；
接触力核验见 `force_audit.json`、`force_comparison.png`；`replays.html` 提供录像。

```bash
.venv/bin/python -u ur7e_xhand_sim/replay_low_friction_stronger.py --output outputs/low_friction_stronger_NEW
.venv/bin/python ur7e_xhand_sim/analyze_material_replay.py outputs/low_friction_stronger_NEW
.venv/bin/python ur7e_xhand_sim/analyze_grip_force.py outputs/low_friction_stronger_NEW
```

### 基础功能验收

```bash
.venv/bin/python ur7e_xhand_sim/check_sim.py --output outputs/ur7e_xhand_check_NEW
```

输出目录必须不存在，避免覆盖历史证据。检查关节顺序、官方臂部参数、测点与网格位置、200 步运行、运动响应、五指独立按压与卸载、坐标变换和相机。按压使用独立测试球；不在正常交互场景中添加测试球。每次保留完整测试数据与失败信息。

这些检查不等同于抓稳、防滑、过度施力控制或 sim-to-real 验证。进入残差训练前仍需实机响应核验、独立滑移／载荷指标、可重复名义抓取和公平的无触觉基线。

2026-10-23 验收通过，证据位于 `outputs/ur7e_xhand_acceptance_20261023_final/validation.json`：初始稳定状态无机器人接触，200 个控制步正常，手腕上移约 12.7 mm，手指最大实际变化约 0.184 rad。五根手指均在测试球接触对应 link2 时产生非零读数，移开后归零；保存了每点三轴数组、接触双方及求解器载荷。浏览器单步和重置已核验。此前转接件单凸包造成的失败保留在独立旧测试目录，没有覆盖。

## 硅胶软块捏压与松开实验

启用 `--silicone` 的服务提供 `silicone_start`、`silicone_observe` 和 `silicone_act`，由独立 Astra API 选择 `locate → approach → align → close → hold → release → retreat → done`。启动：

```bash
.venv/bin/python ur7e_xhand_sim/run_silicone_squeeze.py --label pink_firm_release_unique
```

这是软材料诊断实验，不是完整 S8V4 抬升任务。Astra 使用原三视角 RGB 和公开标明的仿真 FEM 诊断值；粉色目标由 Astra 的 ROI 加实测深度定位。示范仅提供掌位相对偏移和初始手形，不提供软块成功结论。桥接器渐进闭合食指（目标最多 0.85 rad，原 PD 与力矩限幅）。已按用户要求删除局部体积比、局部压缩率和包围盒尺寸三个形变阈值停止条件；这些数值仅记录诊断。全过程不附着、不修改物体位姿，不把软块形变伪装成 XHand 触觉读数，也不执行抬升。

`outputs/silicone_squeeze/<label>/` 保存全量控制、物理评估、局部主伸长推导的压缩指标和三视角图像；`outputs/silicone_agent/<label>/` 保存模型输入、输出和 API 线程编号。`principal_compression_p05` 是每个四面体最小主伸长的第五百分位对应压缩率，不是整个方块宽度的缩短百分比。初始落桌也会产生局部应变，分析必须与捏压前基线比较。可使用 `analyze_silicone_squeeze.py <run_dir>` 生成回放和曲线。

2026-10-24 网格与接触审计记录在 `outputs/silicone_audit_20261024/`。初始物理网格为 605 节点、2347 四面体，表面为 304 节点、604 三角面；未发现零体积单元，最小 mean-ratio 质量约 0.215。原 primitive 显示网格只有 8 个角点和 12 个面，会漏掉角点之间的局部凹陷。`fem_diagnostics.use_physics_surface` 现改为直接渲染实际物理边界，未改变任何四面体；检查时渲染顶点与物理顶点位置差为零。

固定机器人、同一软块落桌的对照中，物理子步 1 / 0.5 / 0.25 ms 的采样最小 `J=det(F)` 分别约 -0.963 / 0.413 / 0.745。负值表示局部四面体翻转，不能解释成正常材料压缩；整体体积比仍可接近 1。1 ms 下只把接触 softness 从 2 mm 缩到 0.2 mm，采样最小 J 反而下降到 -2.163。因此采用较小子步，保留 2 mm softness；这些是采样对照，不是所有子步均已收敛的证明。

Legacy FEM 刚柔接触在表面节点上用刚体 SDF 修正速度，并非面/边连续接触检测或整体隐式接触求解。2 mm softness 下，指数影响范围约为 4.6 mm；它不是材料厚度。按三角面遍历时，共享节点存在重复读写，是需要进一步隔离验证的数值敏感点。当前材料求解使用默认单次 Newton 更新，无线搜索；不能仅凭停止条件删除就宣称大变形可信。每个阶段新增 `fem_positions.npz`、`fem_contacts.json`；后者是几何 SDF 接近/穿透诊断，不是实际接触力或触觉读数。逐帧网格位置另存于 `frames/*_fem.npz`。

较小子步会增加运行时间，API 客户端技能等待默认延长至 3600 s。若仅客户端中断，可用新的 agent 日志标签加 `--resume` 观察并继续当前试验，不重新调用 `silicone_start` 或重置场景；同一试验只能有一个活动客户端。

性能审计发现 Genesis 原 PCG 循环在所有批次已收敛后仍会调用剩余空 kernel 直到 500 次。软块场景现仅跳过这些空迭代，仍保留原 `1e-6` 残差判据和 500 次预算；这不是新的形变停止条件。`pcg_noop_result.json` 验证已收敛后额外执行 500 次，位置增量与物理顶点逐位不变。落桌测试 2400 次线性求解均在 32 次以内清除原收敛标志。每个捏压阶段另存 `fem_pcg_stats.json`，记录是否用尽预算仍未收敛。

`analyze_silicone_mesh.py <run_dir>` 生成实际物理表面的阶段图。图中去除了最佳拟合的整体平移和旋转，形变不放大，颜色是相对静止网格的剩余位移（毫米）；它不代表接触力。`source_diagnostics.json` 还记录了表面边长（中位数约 6.73 mm、最大约 11.37 mm）和当前材料公式的反射示例：`F=diag(-1,1,1)` 时 J=-1，但现有 `R=U Vᵀ` 得到的弹性能为零。这是公式层面的反例，说明不能把翻转后仍有限的数值当作有效硅胶响应。

无形变阈值复测 `outputs/silicone_squeeze/pink_no_threshold_20261024_v3/` 已由独立 Astra agent 完成（API 记录在同名 `outputs/silicone_agent/`）。食指目标达到 0.85 rad；首次记录到翻转在第 959 步、目标约 0.72 rad，全程最多 12 个翻转单元，最小 J=-0.913。闭合终点翻转的 11 个单元初始质量为 0.516–0.991、中位数 0.808，因此不能归咎于初始劣质细长单元。全部 139100 次 PCG 求解最多用了 40 次 kernel 调用，无预算耗尽仍未收敛的记录；这仅说明线性系统达到既有判据，不表示非线性接触物理解正确。

松开后最终翻转数归零、总体积比 0.99905，去除整体平移旋转后的最大剩余节点位移约 0.18 mm。但现有 release 仅打开食指近端关节，其余仍为预抓手形；退回时软块曾被短暂带高约 8.8 cm，之后脱离并落桌。因此 `completed_sequence=true` 只表示技能序列完成，既不是纯原地松开测试通过，也不是大变形模型通过验证。完整指标见 `audit_summary.json`，图像见 `physical_mesh_stages.png`，视频见 `replay_three_views.mp4`。此前 v2 在手指接触前为应用空迭代优化而中断，原因记录于该目录的 `interrupted.json`，不得计作完整试验。

### 200 Hz 滑移监测与报警后增力（2026-10-24）

证据：`outputs/slip_recovery_20261024_v1/`。使用同一示范的 1275 条关节目标，
每条 25 ms；每个物理步 5 ms 记录一次五指 120 点三轴触觉，共每组 6375 帧。
没有调用 VLM、修改木块位姿、附着物体或重算 IK。五组各一次，结果属于探索性验证。

| 配置 | 触觉报警时间 | 干预 | 结果 |
|---|---:|---|---|
| μ=0.8，200 g | 23.315 s | 仅监测 | 抬起后滑落，28.760 s 最终失去指尖接触 |
| μ=0.8，200 g | 23.310 s | 报警后增力 | 离桌 ≥5 cm 保持3秒，保持最低6.47 cm |
| μ=0.01，45.56 g | 21.410 s | 仅监测 | 无有效抬升，25.350 s 最终失去指尖接触 |
| μ=0.01，45.56 g | 21.520 s | 报警后增力 | 仍无有效抬升，25.390 s 最终失去指尖接触 |
| μ=0.8，100 g | 无 | 成功对照 | 保持3秒，保持最低7.10 cm |

检测器只读当前和历史触觉，以及固定测点坐标；预紧结束后启用。
在 50 ms 窗口内，拇指或食指信号加权中心移动速度 >20 mm/s，或测点信号总量下降 >25%，
且前后信号量都 ≥0.01 名义 N，连续3帧满足时报警。阈值参考了历史40Hz日志，不是盲测结果。
这里的信号总量为各测点三轴向量模长之和，不是物理夹持力。

闭环组报警后下一物理步（5 ms）开始增力，50 ms 内将拇指、食指6个关节 kp 从3升到6，
kv 乘以 √2；关节目标及力矩上限不变。200 g 闭环组报警前0.5秒／增益完成后0.5秒的
物理接触合力均值分别为：拇指4.35→8.49 N，食指4.42→8.52 N。
该力由求解器接触力逐指矢量求和后取模，不是法向力，也不是测点读数。
低摩擦组同等增力没有救回，不能据此推断任意增力都无效。

**采样频率不等于检测延迟。** 独立物理标签采用接触处材料点相对切向速度的后向差分，
按接触力模长加权；>5 mm/s 且持续3个采样作为“微滑代理”。200 g 监测组在21.385 s
已出现该代理，23.315 s 才报警，晚约1.93秒，但比最终脱离接触早5.445秒。
低摩擦监测组对应延迟约30 ms，闭环组约130 ms。不同运行存在细微仿真差异。
100 g 成功组同样存在早期微滑代理，因此微滑、危险滑落和失去接触必须分开标注。
本轮验证的是触觉前兆与一次成功干预，尚未证明精确起滑检测、200Hz优于40Hz或长期抓稳。

历史40Hz失败日志复查见 `historical_failure_review.json`：200 g 旧试验中食指测点峰值在
26.8→28.0秒几乎维持0.269 N，但信号总量从1.925降到1.282，激活点从16降到12。
只看单个峰值会遗漏接触分布变化；应保留完整空间阵列与时间窗口。
历史40Hz数据不能补回5ms级起滑瞬间。此前接触索引错误导致的触觉漏报不作为真实滑移样本。

训练文件分层保存：

- `inputs_200hz.npz`：五指三轴触觉 `[6375,5,120,3]`、实际关节角、目标角、已应用增益和时间；不含物体真值。
- `labels_200hz.npz`：仅供评价/监督的物体姿态、离桌高度、接触相对速度、接触载荷和数量。
- `training_labels.npz`：微滑代理、接触状态、未来250/500/1000ms最终接触丢失标签及各自有效掩码。
- `highfreq_events.jsonl`：逐5ms特征、报警、增益与明确隔离的评价量。
- `dataset_manifest.json`：字段、有效掩码、数据划分与局限说明；`audit.json` 为独立核验。

未来接触丢失标签必须乘以对应 `valid_loss_target_*` 掩码：预测窗口遇到增力干预时无效，
避免把被救回的危险状态误标成安全。低摩擦物体仍在桌上，接触丢失不等同于空中掉落。
持续微滑标签是离线标签，使用至多10ms后续样本确认；模型输入必须只取过去窗口。
五条轨迹共31875帧不等于31875个独立样本，应按完整episode/物体/材料/轨迹划分数据，
后续加入多次重复、不同延迟、不同摩擦/质量、正常触碰及可恢复微滑。
当前 KinematicTaxel 剪切项直接由仿真相对速度构造，存在仿真捷径，必须实机校准后验证。

实验期间 `environment.py` 的可选硅胶 FEM 分支发生同期修改；本组均未启用硅胶，
核对差异未改变本组刚体参数与控制路径，但没有逐子进程记录源码版本，不能声称全部运行
源码字节一致。保留 `source_snapshot/`、`environment_at_audit.py`、
`environment_source_drift.diff` 和 `source_drift_review.json`；其余3个协议源码哈希一致。
后续正式采集应在隔离、固定版本的工作区运行。

```bash
.venv/bin/python -u ur7e_xhand_sim/slip_recovery_experiment.py --output outputs/slip_recovery_NEW
.venv/bin/python ur7e_xhand_sim/analyze_slip_recovery.py outputs/slip_recovery_NEW
.venv/bin/python ur7e_xhand_sim/material_replay_preview.py --output outputs/slip_recovery_NEW --port 8770
```

`replays.html` 提供五组录像、恢复曲线和起滑时间分析；录像为稀疏渲染，精确时序以200Hz日志为准。

### 软块的仿真内部验证与 IPC 配置（2026-10-25）

没有实物也可以提高**给定材料模型下的数值可信度**。本轮增加了 IPC 接触配置、
`stable_neohookean` 非线性弹性模型，以及独立压板的时间步和网格敏感性试验。
仍用 E=100 kPa、ν=0.45、ρ=1100 kg/m³、摩擦系数0.8；这些是可复现的模型假设。
不要求实物标定，不再添加基于形变大小的停止阈值。此模型没有加入黏弹性、损伤或塑性。

安装可选依赖、启动当前验证配置：

```bash
uv pip install --python .venv/bin/python -r ur7e_xhand_sim/requirements-silicone-ipc.txt
bash ur7e_xhand_sim/launch.sh --scenario small_clutter --silicone --silicone-solver ipc --silicone-substeps 2 --output outputs/silicone_ipc_NEW/runtime
```

场景物理步5 ms，2个子步对应2.5 ms；IPC 接触距离尺度0.5 mm。
保留 `--silicone-solver legacy --silicone-substeps 20` 供对照。
机器人使用双向软约束耦合，Genesis 原有刚体碰撞继续启用；IPC 负责软体接触。
阶段记录 `ipc_coupling.json` 比较 IPC 碰撞体与 Genesis link 位姿，避免只看软块稳定而漏掉手指代理漂移。

`ipc_support.py` 只在场景导入时合并重复顶点，并补齐凸手指碰撞网格的小孔。
发现8个手指碰撞网格各缺一个三角面，修复后闭合；没有放大碰撞体、改成包围盒或关闭软体碰撞。
源 STL 与 Genesis 源码不变，具体修复计数写入场景参数。
软块显示直接使用物理网格表面。

独立压板不控制 UR7e，按同一轨迹压缩、保持、释放；不直接改软块位置或节点。
压板 PD 执行器限幅500 N，这是执行器参数，不是形变停止条件。目标压缩44.4%，
受力限幅后实际最大压缩约28%，应以实际结果为准。`servo_force_N` 是 PD 估计，不能当作接触力标定值。

证据目录：`outputs/silicone_verified_20261025/`，结果为：

| 时间步 | 四面体数 | 保持末端高度 | 释放后高度 | 全程翻转单元最大数 |
|---|---:|---:|---:|---:|
| 5 ms | 2131 | 32.422 mm | 44.945 mm | 0 |
| 2.5 ms | 2131 | 32.479 mm | 44.945 mm | 0 |
| 2.5 ms | 4149 | 32.501 mm | 44.941 mm | 0 |

初始高度45 mm。三组共4000个物理步逐步记录，所有数值有限，最小 J 分别0.770、0.770、0.762。
时间步减半：高度全程 RMS 差0.091 mm、最大瞬时差0.433 mm；细化网格：RMS差0.040 mm、最大差0.115 mm。
保持末端差分别0.057 mm和0.022 mm；释放后总体积比均约0.99975。
相对 Genesis 压板平面的最大穿入量分别0.069、0、0 mm；该指标不是完整的 IPC 三角面距离审计。
两档对照支持此工况的数值稳定性，不能声称已证明任意工况或局部应力完全收敛。

复现（输出目录必须尚不存在）：

```bash
.venv/bin/python ur7e_xhand_sim/validate_soft_press.py --dt .005 --maxvolume 1e-7 --output outputs/press_5ms_NEW
.venv/bin/python ur7e_xhand_sim/validate_soft_press.py --dt .0025 --maxvolume 1e-7 --output outputs/press_2p5ms_NEW
.venv/bin/python ur7e_xhand_sim/validate_soft_press.py --dt .0025 --maxvolume 5e-8 --output outputs/press_fine_NEW
```

`press_comparison.json`、`press_convergence.png`、`press_before_after.png` 保存对照摘要、曲线和可视化。
`press_5ms_coarse` 为早期相机保存类型错误中断的记录，不计入完成试验；有效5 ms组为 `press_5ms_coarse_v2`。
FEM 形变和接触诊断仍未接入 XHand 的600点触觉数组，不应把软块网格读数标为触觉力。

算法参考：[Genesis IPC 耦合说明](https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/theory/coupling/ipc_coupler.html)、
[IPC 原始工作](https://ipc-sim.github.io/)。具体可信范围以本仓库实际测量为准。

完整机械手复测：`outputs/silicone_squeeze/pink_ipc_verified_20261025_v1/`，
独立 Astra API 经 bridge 完成定位、靠近、对齐、捏压、保持、全手松开、退回，未直接设置软块位姿。
1597条25 ms控制周期诊断中翻转数均为0、最小 J=0.376；局部压缩 p05 峰值22.1%，
单个单元最小主伸长对应的最大压缩67.9%，这些局部应变不是整块缩短比例。
保持阶段去除刚体运动后最大节点形变6.01 mm；松开退回后最大剩余形变0.074 mm，
与初始重力平衡形变一致，总体积比0.99975。软块发生平移、转动，并非位置保持试验。
释放与退回阶段底面最高离桌2.34 mm，最终0.484 mm；未重现旧版被带高8.8 cm的现象。
该手部诊断按25 ms采样，不应声称检查了每个 IPC 子步。

阶段性 IPC 手指代理与 Genesis link 平移差最大0.323 mm（保持阶段）；它是双向软约束的残差，
不能声称两个求解器位姿完全相同。压板的低敏感性结果尚不等于复杂手指接触已完成同样的网格收敛验证。
原始 IPC `get_state().vel` 没有同步，旧记录中的速度0不是静止证据；
`finite_difference_speed.json` 从保存节点位置估算区间平均速度，当前源码对 IPC 速度返回 null 并标为不可用。

当前 `--silicone` 默认选择 IPC / 2子步，Legacy 仍可显式选择。
此次试验源码在 `source_snapshot/` 留存，之后源码只调整默认配置、速度字段可用性和来源哈希记录。
当前服务已重启加载新版诊断并通过只读接口检查；历史 JSON 不改写。
