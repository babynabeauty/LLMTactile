# 手臂 + Wuji 灵巧手交互仿真

已在本机 RTX 5090 上运行。默认使用 **Franka Panda 7 关节机械臂 + Wuji 20 关节右手**，包含桌面、可移动立方体、重力、碰撞、关节位置控制和 140 个触觉探针。机器人是同一个 MJCF 关节链，手腕固定连接在机械臂末端，手部会随机械臂实际运动。

初始姿态为手掌与桌面平行、掌心朝下、手指沿世界 −Y 方向水平伸出。重置会恢复这一姿态；移动手腕时保持当前朝向。

这是独立的交互场景，不是上游手内姿态任务，也不运行训练或数据采集。

## 启动与操作

```bash
cd /home/sai/Documents/ChatGPT/Shiqi/tactile-genesis
bash arm_hand_sim/launch.sh
```

首次编译可能需要约一分钟。终端出现 `ARM_HAND_READY` 后，打开 <http://127.0.0.1:8766>。服务已启动时直接打开网页，不要重复启动占用同一端口。

- **运行 / 暂停**：持续推进物理仿真，保持当前关节目标；暂停后物理时间不前进。
- **单步 0.1 s**：执行 4 个控制步，每控制步 0.025 s。
- **移动手腕**：世界坐标 XYZ 各方向移动目标 2 cm；通过 IK 和关节控制器驱动机械臂，保持当前手腕朝向。实际位移受重力、接触和控制器跟踪误差影响。
- **张开 / 半握 / 开合滑块**：改变手指姿态，不包含自动抓取策略。
- **关节控制**：展开面板，可以分别调整 7 个机械臂关节和 20 个手指关节，滑块单位为弧度。
- **重置**：恢复初始姿态及立方体位置。

画面包含固定全景相机、固定手部近景相机；右侧显示仿真时间，下面显示触觉反馈。先暂停再调整关节比较方便。连续运行速度取决于渲染和请求速度，不保证等同真实时间。

停止服务：

```bash
python3 arm_hand_sim/client.py close
```

## 命令行接口

```bash
python3 arm_hand_sim/client.py describe
python3 arm_hand_sim/client.py observe --compact
python3 arm_hand_sim/client.py move --json '{"delta":[0,0,-0.02],"steps":10}' --compact
python3 arm_hand_sim/client.py hand --json '{"closure":0.55,"steps":10}' --compact
python3 arm_hand_sim/client.py step --json '{"actions":{"joint1":0.0},"steps":4}' --compact
python3 arm_hand_sim/client.py reset --compact
```

`step` 的动作是关节限位内归一化目标：`目标弧度 = (上限+下限)/2 + action*(上限-下限)/2`，`action ∈ [-1,1]`。未指定关节保持原目标。**这里的动作映射与先前固定手环境不同，不能直接复用旧动作数值。** `describe` 返回名称、顺序、限位和映射。每次 1–10 个控制步，非法动作拒绝执行。`move` 每次位移向量不超过 5 cm；它是局部 IK，不提供避障规划。

`observe --json '{"full":true}'` 可按需读取当前完整触觉数据，每点顺序为 `[Fx,Fy,Fz,Tx,Ty,Tz]`，坐标系是所属传感器附着连杆的局部坐标系。16 个传感区域共 140 个探针，来自上游 Wuji `low/probes_hand.json`。Genesis KinematicTaxel 用接触运动学和弹簧阻尼模型估计力/力矩；读数可能包含自接触，不代表经过硬件标定的真实载荷。页面显示区域内单点最大力，不是区域合力。

## 保存内容与限制

不会保存逐步轨迹、动作数据集或视频。`runtime/scene.png`、`runtime/hand.png` 仅覆盖保存最新画面，方便预览。进程日志用于排错。`validation.json` 是一次功能检查的汇总结果，不是采集的训练数据。

机械臂模型取自官方 MuJoCo Menagerie，移除了平行夹爪；Wuji 使用项目原模型。新增固定安装件的质量、惯量及安装方向是仿真配置，尚未按真实转接件标定。手指沿用原模型控制参数，机械臂使用较高增益位置控制，存在跟踪误差。碰撞模型采用仿真器支持的几何近似。当前用于查看、操作和验证臂手联合仿真，不宣称已经完成稳定抓取或真实硬件动力学标定。

机器人底座和桌面固定；重置为同一个初始场景，无随机化。机械臂所有零位不满足原始 Panda 第 4 关节限位，加载时可能出现 neutral position 警告；执行物理步前已切换到 IK 求出的合法初始姿态。

## 版本与验证

- 沿用根目录 `.venv`：Python 3.12.3、Genesis 1.4.1、PyTorch 2.10.0+cu128。
- Tactile Genesis 上游：`4777fd5da08244acbc677dca6f6f9acfb2d69d2b`。
- Panda 资产：`google-deepmind/mujoco_menagerie`，提交 `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`，许可证保存在 `vendor/franka_emika_panda/LICENSE`。
- 详细版本见 `versions.json`，原环境依赖锁定记录保留在根目录已有交付中。
- `asset_source.json` 记录 79 个 Panda 文件及 SHA-256；运行 `python3 arm_hand_sim/fetch_assets.py` 校验，缺失文件会从固定上游提交恢复，已有但不匹配的文件不会被覆盖。

服务运行时可执行 `python3 arm_hand_sim/check_sim.py`。该检查会重置场景并操作机器人，完成后恢复初始暂停状态。已验证：200 个连续控制步状态有限、暂停时仿真时间不变、非法输入不推进仿真、手腕与手指实际运动、无接触与接触时触觉变化、140×6 数据布局、两路相机 JPEG 输出。

原固定手实验仍使用端口 8765；本场景独立使用 8766。
