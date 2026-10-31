# Tactile Genesis × 当前 Codex Astra

这是一个**暂停式视觉—触觉控制闭环**：当前对话中的 Astra 读取图片和传感器数据，通过命令行发送关节动作。本地 Python 进程负责物理仿真、控制器和记录，不包含独立模型 API，不会自己生成策略，也不要求教师 checkpoint。

默认环境：Wuji 20 自由度右手、手内立方体姿态调整、`low-hand/force_torque`（实际 140 个 taxel）、clean 传感器。物理步长 0.005 秒，控制步长 0.025 秒。每次动作默认执行 4 个控制步；等待模型期间，仿真完全暂停。

## 安装与快照兼容修复

上游固定在 `4777fd5da08244acbc677dca6f6f9acfb2d69d2b`。环境使用 Python 3.12、PyTorch 2.10.0+cu128；第三方依赖固定在 `requirements.lock`，需要重新安装时运行 `bash astra_bridge/install.sh`。

这个上游快照不能直接启动所选任务，本地补充了以下兼容修复，保留在 `codex/astra-tactile-bridge` 分支工作区中：

- 补齐 Eden 缺失的 `soft_dof_pos_violation`，按给定上下界计算逐关节越界距离。
- 补齐任务需要的四元数乘法、逆、最短旋转向量和角距离接口；已测试恒等旋转、90° 旋转、乘法顺序及 q/-q 等价性。
- 将 `friction_cone`、`contact_resolution` 传给 Genesis 的 `RigidOptions`，而非错误地传给 `Scene`；保留上游的 elliptic/signorini 选择。
- Eden 的查看器构造改用 Genesis 1.4.1 支持的 `refresh_rate`、`split_envs`，避免弃用参数在配置复制时发生冲突。
- 为终止管理器补齐任务所调用的 `get_term_dones` 访问器，读取已计算的终止标记，保留掉落惩罚。

Wuji 的 MJCF 本身没有 freejoint，因此底座固定；启动时关于 MJCF 不支持 `is_fixed_base` 的通用警告不代表本模型底座会自由落下。

## 启动与控制

在仓库根目录运行：

```bash
bash astra_bridge/launch.sh --label astra_live
```

首次运行需编译 GPU 内核。出现 `BRIDGE_READY` 后，在另一终端或 Codex 中调用：

```bash
python3 astra_bridge/client.py describe
python3 astra_bridge/client.py reset --json '{"seed":42}' --compact
python3 astra_bridge/client.py observe --compact
python3 astra_bridge/client.py step --json '{"actions":{"r_index_finger_mcp_flex":0.2},"steps":4}' --compact
python3 astra_bridge/client.py observe --json '{"full":true}'
python3 astra_bridge/client.py status
python3 astra_bridge/client.py close
```

`rgb_path` 是本机 PNG 的绝对路径；Astra 必须实际查看该图片，再结合触觉决定下一步。`--compact` 只简化终端显示，完整数据仍保存在服务端日志。`--output FILE` 可额外保存完整响应。

服务只监听 `127.0.0.1:8765`。可用服务端 `--port` 与客户端 `--url` 指定其他端口。可通过 `--output` 指定新的输出目录；已有目录会拒绝覆盖。

## 动作和观测约定

- `actions` 是关节名称到 `[-1,1]` 归一化位置动作的映射。缺省关节维持上一次目标。它不是弧度、速度或力矩。
- 上游映射为 `目标角度 = offset_rad + action × scale_rad`；`describe` 返回每个关节的 offset、scale、物理限位和顺序。随后仍由上游动作修饰器和显式 PD 控制器执行；超出物理限位的目标不会使关节绕过物理限位。
- `steps` 必须为 1–10 的整数，默认 4。无效关节、越界动作、NaN/Inf 或布尔动作会被拒绝，仿真不会前进。
- 可加入 `expected_episode` 与 `expected_step` 防止基于旧观测执行动作；不匹配会拒绝请求。
- 终止后禁止 `step`，保留最后场景与触觉，直到显式 `reset`。reset 会结束并保存上一个回合。
- `reset(seed=...)` 重设随机数生成器，保留上游目标生成规则。该任务的 `sample_relative=true` 会基于上一目标继续采样，因此同一进程中重复使用相同种子并不保证目标姿态相同。要复现本报告的首个回合，请启动新服务后首次 `reset(seed=42)`；做独立种子 benchmark 时需要明确是否沿用这项上游状态约定。
- 模型获得 RGB 图、真实无噪声关节本体感觉、触觉与目标四元数 `[w,x,y,z]`；不获得物体姿态真值、姿态误差、奖励或成功评分。这里只是按接口隔离，并非阻止拥有文件访问权限的智能体读取评测文件的安全沙箱。
- 触觉向量采用各附着 link 的局部坐标系，前三维单位 N，后三维单位 N·m。上游实际使用 `KinematicTaxel`：力来自接触深度与相对速度的弹簧阻尼估计，并非物理解算器的真实接触冲量。力矩关于附着 link 原点，已包含力臂项及扭转阻尼项；多个探针的估计可能重叠，不能把摘要之和当成标定后的净接触 wrench。保留上游接触过滤设置，不将所有激活 taxel 都解释为物体接触。
- 摘要使用最近一个物理子步的数据（历史索引 0）；完整记录保存每个控制步内的 5 帧历史，按新到旧排列。数组形状以实际输出为准，最后一维固定为 6。首轮 `astra_demo` 沿用了上游 `last` 的索引 -1，事后核对发现它对应最旧子步，因此当轮摘要相对当前场景滞后 0.020 秒；完整五帧记录无缺失。交付版已经改为索引 0，并验证摘要与完整数组一致。
- `active_taxels` 使用力模长大于 `1e-5 N` 的可视化阈值，不是经过真实硬件标定的接触阈值。

## 调试配置与评测边界

第一版只保留对象初始放置事件，关闭质量、摩擦、执行器随机化和外力扰动，并关闭本体感觉噪声。原任务的目标生成、奖励和掉落/超时终止规则保留。完整配置写入每次运行的 `resolved_config.yaml`。

这是用于接通接口的调试配置，不能直接对照论文结果。暂停式控制也不能证明实时控制能力；报告仿真时间与模型等待时间时应分别列出。没有观察反馈就预先生成整段动作的脚本只能算测试脚本，不能标为 Astra 实际闭环。

正式演示时，只通过 `describe/reset/observe/step/status` 获取信息，不读取 `evaluation.jsonl` 或 `summary.json`；结束控制后才分析这些文件。

## 输出与验证

默认输出到 `outputs/astra_bridge/<时间>/`：

- `manifest.json`、`resolved_config.yaml`、`description.json`：版本、配置、动作映射和传感器布局。
- `interaction.jsonl`：每轮请求、动作前后完整观测、决策等待时间。
- `episode_*/frames/`、`rollout.mp4`：逐控制步场景帧与按 40 FPS 编码的视频；视频反映仿真时间，不含等待模型的停顿。
- `episode_*/tactile.npz`：完整六维触觉及时间戳。
- `episode_*/tactile_stream.npy`：每个控制步追加一个 `(5,140,6)` 数组（含第 0 步），传感器按 `description.json` 的排列拼接。可以在打开的二进制文件上反复调用 `numpy.load(file, allow_pickle=False)`，读到文件结尾；帧序号乘以 0.025 即记录时间。单个数组内 5 个子步由新到旧排列，第 0 步为重置历史。
- `episode_*/evaluation.jsonl`、`summary.json`：仅供事后分析的真值、姿态误差、目标达成事件和结束情况。

录像与压缩触觉在显式 reset/close 时保存，终止时也保存压缩触觉。建议使用 `close` 正常退出；进程被强制终止时，已写入的帧、JSONL 和触觉流仍在，但未完成的压缩数据/视频可能尚未生成。触觉流是演示结束后增加的落盘加固；首轮 `astra_demo` 的完整触觉保存在 `tactile.npz`。

决策等待时间包括模型推理、图片查看、工具调度与对话停顿，不是单独的模型 API 推理延迟。

运行测试前单独启动一个服务，避免把测试动作混入 Astra 演示：

```bash
.venv/bin/python -m unittest discover -s astra_bridge -p 'test_*.py'
bash astra_bridge/launch.sh --label scripted_smoke
# 在另一个终端执行；完成后会关闭此服务。
.venv/bin/python astra_bridge/smoke_test.py --steps 1000
```

然后重新启动 `--label astra_live`，由 Astra 逐轮观察并控制。

## 本次交付

结果见 [接入验证报告](RESULTS.zh-CN.md)。演示、最终测试、可继续操作的暂停环境分别位于 `outputs/astra_bridge/astra_demo/`、`smoke_verified/`、`ready/`，互不混用。交付时服务监听 `127.0.0.1:8765`，已用种子 42 初始化并暂停；先调用 `status` 或 `observe` 即可继续。进程信息在 `outputs/astra_bridge/active_server.json`，重启电脑后按上方启动命令启动新服务。
