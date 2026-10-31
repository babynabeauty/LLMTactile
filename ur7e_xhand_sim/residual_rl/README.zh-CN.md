# 触觉局部残差 PPO：第一阶段

本目录只连接 Genesis 仿真。机械臂和手指关节目标来自已有成功示范；策略仅调整拇指、食指的 PD 增益。尚未开放手形残差、抬升暂停、重新抓取或真实硬件。

## Astra 命令轨迹导入对照（2026-10-26）

`outputs/astra_trajectory_seed3_20261026_v1/` 保存了原位 Astra 执行日志导入、原材料预检及冻结 seed3 的四组对照。转换器 `import_astra_trajectory.py` 使用发送目标及其生效区间，不把实测关节位置作为目标，也不复制历史触觉。`adaptive_benchmark --trajectory <converted/control.jsonl>` 通过 metadata 核对场景、关节顺序、初态和基础 PD，冻结输入副本后执行；本接口目前仅支持已核验的无杂物原位与大平移场景，不自动迁移任意场景。大平移场景的方块位置在建场景时设置并读取核验，不在回合内瞬移。原默认示范和训练配方保持不变。

九组已评估材料为 75/125/175 g × μ=.3/.5/.7；fixed2、fixed3、tactile seed3 的 task/strict 均为9/9，no_tactile seed3为3/9。前三者均接触载荷分别16.9492、25.0271、12.4643 N；触觉相对fixed2降低26.46%。无触觉全组均值含失败，不能直接用于节力结论；双方共同成功的三个条件上，触觉载荷反而高13.35%。所有模型权重未训练/修改，初始历史一致，fixed2重复回放全部保存通道逐位一致；独立重算保存在该目录 `independent_review.json`，图表和汇总为 `comparison.png`、`experiment_report.json`。

**解释限制：该 Astra 原位轨迹与旧示范最大关节差仅2.62e-6 rad，时间与阶段相同。这是实际日志导入和回放对照，不能作为实质新轨迹泛化证据。** 触觉本次三秒保持及记录尾段均有双指支持，不代表长期稳定。未在回放时调用 Astra 或叠加规则触觉控制器。

### 大平移轨迹对照（2026-10-26）

`outputs/astra_large_trajectory_seed3_20261026_v1/` 使用历史 Astra 执行 `demo_context_four_20261024_v1_large`，方块平移 `[0.15, 0.10, 0] m`（18.03 cm）。按阶段对齐后的最大手臂关节差约 26.68°、腕部位置差约 18.96 cm，手指闭合模式沿用示范。实际评估输入为 `imported_trajectory_v2/control.jsonl`；初版转换文件保留追溯，但未用于评估。模型仍为冻结 seed3，无续训、无实时 Astra 调用、无额外规则反馈。

75/125/175 g × μ=0.3/0.5/0.7 共九条件、四控制器，完成 36 个正式回合。固定增益 2、3 均 task/strict 9/9；触觉 PPO 均 8/9；无触觉 PPO task 3/9、strict 2/9。八个共同成功条件中，触觉平均接触载荷 13.0687 N，分别比固定 2 的 16.9655 N、固定 3 的 25.0471 N 低 22.97% 和 47.82%。不能把包含失败回合的低载荷当作节力证据。

触觉唯一失败条件为 175 g、μ=0.7：在保持末尾失去支持并滑落；策略此前已逐步增力至上限 3，仍未恢复。无触觉 75 g、μ=0.3 虽满足原高度判据，但发生接触中断且末帧已低于 5 cm，因此不计 strict 成功。不能将本轮概括为触觉控制器可靠性优于固定增益 2。

本轮保持区间依据新轨迹设为 31.425–34.425 s，触觉 60 Hz、动作 15 Hz。原材料预检及重复回放通过；九条件固定 2 重置核对的全部记录通道逐点一致，源文件及模型哈希不变，初始观测和动态阶段时序均经独立复核。`--audit-prefix` 对策略开始前 1,344 个 60 Hz 时刻单独记录碰撞检查，九条件均未检出碰撞；这不是连续 300 Hz 的碰撞保证。详见 `experiment_report.json`、`independent_review.json` 和 `comparison.png`。

初轮提供单个新手臂轨迹上的单训练种子迁移证据，使用已用过的材料条件和三秒保持；不代表新手形、长时间保持或真实机器人上的泛化验证。

### 大平移场景补测种子 1、2（2026-10-26）

`outputs/astra_large_trajectory_seeds12_20261026_v1/` 保留两颗新增种子的评估及三个种子的合并报告。设置完全沿用上轮，仿真 seed 固定为 7000，模型使用各训练 seed 的第三阶段 786432 步 checkpoint，没有重新训练。每颗种子补跑 fixed2、fixed3、tactile、no_tactile 九条件，再重复 fixed2 作重置核验；共新增 36 个模型条件回合、36 个固定基线回合与 18 个重置核验回合。

| 训练种子 | 触觉 task / strict 成功数 | 无触觉 task / strict 成功数 |
| --- | --- | --- |
| 1 | 9 / 9（共9条件） | 9 / 9（共9条件） |
| 2 | 9 / 9（共9条件） | 9 / 9（共9条件） |
| 3（沿用上轮） | 8 / 8（共9条件） | 3 / 2（共9条件） |
| 合计 | 26 / 26（共27次模型条件评估） | 21 / 20（共27次模型条件评估） |

种子1触觉/无触觉的全组平均接触载荷分别为 13.0126/15.6365 N，触觉降低 16.78%；种子2为 14.1590/24.9966 N，降低 43.36%，两组均在九条件共同成功的前提下比较。种子2无触觉在 23.0 s 达到增益3并保持至结束，其执行增益及所有物理、状态、触觉记录与固定3逐点相同。三种子相对固定2的共同成功条件上载荷降低 20.76%，相对固定3降低 46.32%；固定基线仍在全部条件成功，触觉仍有种子3的一个失败。

种子1还有保持判据之外的尾段失接触：触觉175 g、μ=.7在34.450 s出现零接触，末帧高度5.895 cm；无触觉175 g、μ=.3在34.4667 s出现零接触，末帧高度6.850 cm。两者均发生在原3秒保持区间之后，不改变原task/strict标签，也不能声称已验证长期稳定。

新增模型来源528项核验通过，涵盖训练配置、源码及18阶段续训链。两次新增 fixed2/fixed3 的全部记录通道和接近阶段物理记录均与种子3逐点一致；轨迹、环境和初始历史一致，模型哈希不变。独立复核见 `provenance_review.json` 与 `combined_integrity_review.json`。三个训练种子共享九种材料与同一条新手臂轨迹，27次模型条件评估不是27种独立材料；成功率收益并非每个种子都有，新增两颗种子的收益主要是载荷下降。

## 当前材料自适应实验（v3）

`adaptive_env.py`、`adaptive_policy.py`、`adaptive_train.py`、`adaptive_benchmark.py` 是当前实验入口。下文旧 `train.py` 入口和两个独立增益的约定仅对应 v2，两个版本的模型与观测不能互换。

- 时钟保持物理 300 Hz、触觉 60 Hz、策略 15 Hz。触觉与机器人状态均保存同步的 12 帧历史；无触觉对照保留完全相同的机器人历史。
- 动作为一个拇指/食指共同增益，指数映射到 `[0.6,3]`，每步最多变化 `0.2`。零动作约为 `1.342` 倍；它不等于原始力度。原关节目标和力矩限制保留。
- 每条 lane 使用显式的质量/摩擦组合，并用对应材料物理回放前缀。当前 32 个训练条件、16 个验证条件和 9 个预留测试条件分别保存为 JSON。9 个测试条件现已完成一次冻结评估，后续调参不能再把它们当作未见数据。重复 episode 不增加材料条件数。
- 同时报告 `task_success`（原三秒保持区间全程达到高度且无碰撞）和 `strict_success`（另要求每个样本都有至少两个带触觉指节物理接触）。旧严格标准没有删除；v3 的终止奖励使用前者，支持连续性另行计分与报告。
- 稠密奖励积分接触支持下的高度、总接触载荷、物理滑速及失去支持的代价，再加动作变化代价及终止奖惩。无物体接触时没有高度收益；当前高度项允许单指接触，区别于 v2 的双指门控。物理真值仅用于奖励和评估，不传入 actor。
- 评估比较初始网络、多个固定增益、独立训练的无触觉模型，以及冻结策略后的触觉置零、跨 lane 置换、试抬后冻结触觉和试抬后保持增益。后两项用于区分持续反馈与早期接触识别。输入干预造成的退化本身不能证明触觉比其它方法更有效。
- 每个评估在最后重复第一个固定控制器，核对全部物理、观测和动作轨迹。源码、材料、奖励和 checkpoint 哈希与结果一同保存。
- 载荷分别报告时间均值、各lane采样峰值的均值、所有lane最大采样峰值。峰值来自 `episode_metrics.peak_contact_force_N`，按60 Hz接触样本统计；不是已标定的指腹法向力，也不能覆盖采样间隔内的真实瞬时峰值。平均节力不代表峰值降低。

实验文件在 `outputs/residual_rl_20261025_v3/`。`predeclared_protocol.json` 保存验证门槛，`revision_*.json` 保存每次训练前的改动理由。最新 fresh seed1/2/3 均通过原定 9 组留出材料、3 秒保持及载荷/接触门槛，结论为**预声明范围通过，触觉收益有边界**。完整 50–200 g、μ=.2–.8 范围和长期夹持仍存在失败；尤其 seed3 一项在任务末端开始失持。详见下文“三种子冻结确认”和 `diagnostics/confirmation_acceptance_review.json`，不能以验收通过代替这些限制。

```bash
.venv/bin/python -m unittest \
  ur7e_xhand_sim.residual_rl.test_core \
  ur7e_xhand_sim.residual_rl.test_adaptive -v

.venv/bin/python -m ur7e_xhand_sim.residual_rl.adaptive_train \
  --conditions-json outputs/residual_rl_20261025_v3/train_conditions.json \
  --config-json outputs/residual_rl_20261025_v3/config.json \
  --output outputs/my_adaptive_training --total-timesteps 131072 \
  --seed 0 --initial-gain 2.75

.venv/bin/python -m ur7e_xhand_sim.residual_rl.adaptive_benchmark \
  --conditions-json outputs/residual_rl_20261025_v3/validation_conditions.json \
  --config-json outputs/residual_rl_20261025_v3/config.json \
  --output outputs/my_adaptive_validation --split validation \
  --models initial=outputs/my_adaptive_training/initial_model.zip \
           tactile=outputs/my_adaptive_training/final_model.zip \
  --intervene-model tactile \
  --interventions zero permuted frozen_after_probe gain_hold_after_probe

.venv/bin/python -m ur7e_xhand_sim.residual_rl.adaptive_report \
  --root outputs/residual_rl_20261025_v3
```

训练命令默认独立高斯探索。`--use-sde --sde-sample-freq 30` 切换至 SB3 原生 gSDE；此时 `--action-std-init` 是**潜在噪声权重的标准差**，不能直接解释成动作标准差。动作分布还取决于当前 actor 特征。校准记录在 `sde_prefix_noise_calibration.json`，更换初始化种子或网络后需重新核对。每 30 个 rollout 局部步及每轮 rollout 开始时重采样，各 lane 独立；episode 结束不会自动重采样。实际执行还受到动作裁剪与增益变化率限制。

后续分支可增加 `--normalize-sde-latent --action-std-init .12 --gamma 1`：使用显式的新策略类，仅把探索噪声特征归一化，动作均值和价值网络保留原特征，采样、log probability 与 rollout buffer 仍走 SB3。等权初始噪声下，非零特征的实际动作标准差约为 `sqrt(.12²+1e-6)=.120004`；学习后仍应记录实际分布，不能只看噪声权重。此改动针对原生 gSDE 中观测到的特征幅度增长，而不是传感器标定。策略创建和加载使用局部 CPU 设备上下文，避免 Genesis 的默认 GPU 设备改变初始化随机数路径；不会修改 Genesis 的全局设备设置。

`--gamma 1` 让有限 181 步回合的优化目标与报告的累计回报一致。原 `.995` 下终止奖励折扣至约 40.57%；材料扫描中，跨材料平均的固定增益排序因此发生反转。每种材料内“成功优于失败”的比较仍成立，不能将该发现说成同一材料下奖励鼓励失败。审计脚本和逐步奖励在实验目录 `diagnostics/reward_discount_audit.*` 与 `reward_discount_steps.npz`。

`fullmc_tactile_seed0_266k` / `fullmc_no_tactile_seed0_266k` 候选采用 `--n-steps 181 --batch-size 362 --gamma 1 --gae-lambda 1 --total-timesteps 266432 --gradient-diagnostics`，两组均为 32 lanes、46 个完整 rollout、每次 PPO 更新 5 epochs。每个 rollout 覆盖一个完整局部回合，使用未折扣、λ=1 的回报目标；诊断只读取梯度，不执行额外优化。它同时改变优势估计、更新时机、minibatch 和预算，**不是仅改变 λ 的单变量消融**。266432 相比前版262144多1.64%，同一候选的触觉/无触觉两组则使用相同预算。配置冻结记录见 `revision_full_episode_mc.json`；运行是否完成以各目录状态及冻结评估为准。

MC冻结评估在 `validation_fullmc_266k` / `development_fullmc_266k`。验证16条件触觉策略task/strict均16/16，保持双指比例全1、最长中断全0，最低保持高度69.96 mm。平均载荷从固定g=2的16.8955 N降至12.9619 N（23.28%，16条lane均降低）；独立无触觉为15/16，错配触觉10/16，探测后冻结触觉11 task / 10 strict。`diagnostics/contact_coverage_fullmc` 对该MC checkpoint独立回放，确认五指节以外漏计载荷为0、原轨迹核对差值全0；此证据独立于旧262k模型的覆盖审计。

仍需保留两个边界。第一，探测后保持已选增益也16/16 strict、13.2087 N，动态策略仅再降低1.87%；开发集动态策略27/32 task、26/32 strict，无触觉28/32 strict，gain_hold29/32 task、27/32 strict，不能只凭冻结触觉退化宣称持续反馈必需。第二，开发集150 g/μ=.2、200 g/μ=.2/.4/.8、137.295 g/μ=.2018均先抬起再真实掉落；另有181.423 g/μ=.6819保持期断触41.67 ms。验证最大采样峰值从24.00 N升至26.03 N，开发最大60.25 N。详见 `diagnostics/fullmc_evaluation_independent_review.json`。这些是反复查看过的开发/验证集合、单seed结果，不是完整50–200 g及μ=.2–.8范围可靠或最终留出验收。

下一候选在相同MC配置上增加 `--separate-feature-extractors`，即 `share_features_extractor=False`，冻结记录为 `revision_separate_encoder_fullmc.json`。它复制同seed**未训练**共享模板的完整policy state至独立actor/critic编码器，保留新空优化器，并恢复模板构造后的RNG；两组的actor与critic都遵循相同触觉消融设置。新增的是一份critic编码器，actor推理结构与容量不变。初始化的张量、独立存储、actor特征、价值和噪声分布需要共同验证；只检查初始动作相同不够，因为零action-head权重会掩盖特征差异。全局梯度裁剪、critic相关优势及额外critic参数仍是解释边界，不能将候选效果直接归结为共享梯度冲突。

报告的matched-pair检查包含编码器共享拓扑和初始化方法。历史manifest缺少`share_features_extractor`时按SB3默认共享True解析，再用保存checkpoint的actor/critic张量存储别名核验；没有权重证据不标为verified。模板复制/RNG说明和参数总数随报告保存，候选声明本身不表示训练、有效性或goal已完成。

Adam的数值epsilon使用可选`--adam-eps`设置。省略时保持原生SB3构造方式；当前已启动的`separate_fullmc_*`仍使用原来的`1e-5`，未改变正在运行的优化器。`model_config.json`分别记录`optimizer_class`、`optimizer_param_group_eps`和`adam_eps_requested`。resume保留保存的实际eps，显式传入冲突值会报错，不能用CLI覆盖值冒充新实验。

报告配对检查新增**实际Adam各param-group eps相同**和**保存状态已核验**。历史metadata没有eps时，直接读取checkpoint中的`policy.optimizer.pth/param_groups`，不猜测默认值；初始和最终checkpoint的实际eps须一致。尚无模型的run只显示requested/unverified；历史state无法唯一识别优化器类，类名缺失时不据eps推断。所有读取均为CPU weights_only，不构建模型、不执行更新。

`revision_separate_adam_eps.json`起初登记条件候选，在前一轮分离编码器冻结评估未通过后启动同预算`eps=1e-8`配对实验。现有`validation_eps8_separate_266k`和`development_eps8_separate_266k`均已完成：触觉分别仅2/16与5/32 task/strict成功，平均载荷4.3392与4.1892 N；无触觉分别12/16与25/32 task（开发strict24/32）。这是失败结果，触觉低载荷包含大量未抓住或掉落回合，不能解释为成功抓握下的省力。实际参数仍以保存checkpoint核验为准，不从训练损失推断成功。

训练奖励缩放由保存模型属性`training_reward_scale`记录，actual model metadata同时记录实际scale。报告只读checkpoint ZIP的JSON，不构造或反序列化模型：历史checkpoint缺该属性时按原来的未缩放行为解析为`1`，再与显式metadata核验；新模型的初始、最终及同预算触觉/无触觉配对scale必须一致。没有保存模型时仅显示请求值或metadata，不能将默认值标为已验证。resume省略参数应保留已保存scale，显式冲突必须拒绝。

奖励缩放只影响训练中的PPO reward、return、value loss和相关梯度。`episodes.jsonl`的原始return、reward_components以及`rollout/ep_rew_mean`保持未缩放单位，冻结评估继续使用原奖励与物理判据。不同scale下的`train/value_loss`不在相同量纲中，**损失数值更小不表示成功率、节力或触觉反馈收益更好**；也不能只按scale平方反推训练质量，因为价值预测、优势和更新轨迹会一起改变。

`eps8_separate_*`冻结评估未达标后的候选登记在`revision_reward_scale_fullmc.json`：回到旧shared fullMC配置，仅将训练reward乘`0.01`，seed0、266432预算、Adam eps=`1e-5`、原初始化及物理奖励记录均保持旧shared配置。预定目录为`scaled_fullmc_tactile_seed0_266k`和`scaled_fullmc_no_tactile_seed0_266k`；这段说明不代表候选已经启动，实际执行以冻结revision、run状态和保存模型核验为准。不得把eps8分离编码器与该候选的差异解释成仅reward scale改变；单项比较对象是旧shared fullMC。

`revision_reliability_vs_step_size.json`登记了后续四个fresh run。A `penalty200_scaled_fullmc_*`相对原scaled参考只将训练`failure_penalty`从75提高到200，学习率仍为`3e-4`；B `lr1e4_scaled_fullmc_*`只将学习率降至`1e-4`，失败惩罚仍为75。两者均保持reward scale=`0.01`、shared、Adam eps=`1e-5`、seed0与266432预算；A与B直接相比较时有两个因素不同，应分别相对原scaled参考解释。

这四个模型的冻结评估统一使用原`config.json`（P75），使eval return的目标一致。改变终止失败惩罚不会改变物理轨迹或actor输入，成功、载荷和接触定义也不变。**A/B训练raw return不可直接比较**，因为P200和P75对同一次失败赋值不同；不重写历史训练回报，也不因优化损失更小判定成功。报告展示每run实际P与LR：P交叉核对完整resolved config、环境protocol和manifest请求；LR核对保存模型JSON、初末优化器group以及actual model metadata。P不保存在PPO ZIP中，缺失metadata不能根据目录名猜测；未完成的保存证据保持未核验。匹配的触觉/无触觉对必须同时满足完整奖励配置与实际LR一致。

### 训练日志的时间对应关系

SB3 2.9 的 `OnPolicyAlgorithm.learn()` 先收集 rollout，再 `dump_logs()`，最后 `train()`。因此同一条 `training_metrics.jsonl` 中，`time/total_timesteps` 和 rollout 指标通常描述刚收集完成的数据，而 `train/*` 是上一次 PPO 更新留下的指标；第一条可能没有 `train/*`，最终一次更新后也可能没有下一次 dump。不能把日志行的步数直接解释成该行训练损失对应的更新，或因最后一条缺失而断定最终更新没有执行。

`episodes.jsonl` 记录的是带探索动作的已完成回合；`training_metrics.jsonl` 的 `train/*` 记录优化指标，两者不能混作冻结策略评估。梯度诊断若启用，也需按自身记录的 rollout/update 时点解释。检查最终模型请使用保存的 checkpoint、完成状态和独立评估，不用最后一条 `train/*` 代替。

归一化262k版本的反馈审计也按材料范围区分：同32种训练材料中，探测后冻结触觉使 task 从31/32降至27/32、strict从31/32降至26/32；同一 checkpoint 的16种新验证材料中，冻结仍16/16且平均载荷更低。前者支持持续反馈在已见条件中有作用，后者意味着持续反馈的泛化收益仍未证明。这是后补结果审计，不改变后续候选的原配置冻结时间。

`outputs/residual_rl_20261025_v3/diagnostics/contact_coverage_audit/interpretation.json` 保存该归一化262k验证组的覆盖审计：固定g=2和触觉策略在60 Hz已记录样本中，漏计手部接触载荷均为0，重跑与原轨迹各核对通道逐点差均为0。触觉策略的平均载荷降低约5.64%，但所有lane最大采样峰值从24.00 N升至26.69 N，各lane峰值均值从21.95 N升至24.16 N；预紧和试抬阶段平均载荷也增加。该结果支持这组采样上的平均节力，不能解释成普遍降力或峰值改善，也不自动推广到其它姿态、材料或后续模型。

`--resume` 恢复模型和优化器，保留 checkpoint 的 PPO 和探索参数；它使用新命令指定的材料及环境配置。输出目录必须是新目录，`--total-timesteps` 是追加的步数。不得通过命令行看似相同的参数覆盖来假称已改变恢复后的学习率或探索方式。

`revision_low_lr_continuation.json` 预声明四个续训分支 `continued_lr1e4_p{75,200}_{tactile,no_tactile}_seed0_533k`。它们分别恢复已完成的 B 组触觉/无触觉父模型；阶段1均为 P75、266432 transitions，阶段2再追加266432，目标累计532864。阶段2的P75分支只增加训练预算，P200分支仅相对前者提高本阶段失败惩罚。LR=`1e-4`、reward scale=`0.01`、shared与Adam eps=`1e-5`保持父模型配置。两种P分支在各自触觉消融组内复用同一父checkpoint；触觉与无触觉父权重已经分别训练，不能要求两者权重相同。

报告的`continuation_audits`与续训表分别记录父run/checkpoint完整SHA、各阶段实际P及配置核验、起点、申请新增、申请累计、实际新增和已记录累计步数；申请预算不代表完成。父来源只从已经读取的metadata关联，缺失时保留未解析，不根据目录名猜测。续训保存的`initial_model.zip`是已经训练的父模型，不是原始未训练恒定增益基线；冻结评估应保留`original_constant_initial`及原固定g=2主对照。每个目录的episode只属于其新增阶段，续训不增加独立训练seed数。

通用触觉/无触觉配对的raw检查及原all-match值保留，不放宽fresh门槛。两者分别恢复已训练父模型时，学得的权重和噪声可以不同；fresh初始化及等权初始噪声假设标注为不适用，不能拿该false判断阶段2惩罚对照失败。真正的阶段2对照在同一消融组内比较P75/P200，报告的`continuation_parent_contrasts`单列父checkpoint身份；已知SHA冲突为mismatch、缺失为unverified，其他配置差异继续显示。恢复张量/优化器逐项相等由预声明引用的`diagnostics/low_lr_continuation_startup_audit.json`独立记录，报告的父身份关联不冒充重新执行了这项审计。

本次恢复保留模型及Adam状态，但环境重新reset、seed0随机数重启，旧rollout buffer不恢复，因此不等于逐位连续训练。只评估最终累计532864的模型，所有冻结评估仍统一P75；不同阶段P下的训练raw return不可直接比较。若进入seed1/2/3确认，每个seed须从头重做266432+266432的完整两阶段流程，不能从seed0父模型开始。续训预声明与启动不表示有效性验收通过，历史实验预算和判据保持原记录。

`revision_normalized_continuation.json` 对旧normalized候选只追加262144 transitions至累计524288，保留LR=`3e-4`、γ=1、GAE λ=.95、128步rollout、P75、scale=1、shared与Adam eps=`1e-5`；不是P75/P200对照。父模型处在回合内第47步，恢复时新环境丢弃未完成回合并重启随机流，不能称作不中断训练。启动审计核对58个模型张量、学得噪声及Adam状态与各自父模型逐项相同；新增epsilon合法性检查不改变本次有效epsilon下的策略运算。

该候选最终冻结评估未通过原成功率门槛：验证触觉15/16 task/strict、13.2560 N（父模型16/16、15.9428 N），开发28/32 task、27/32 strict、13.3860 N（父模型31/32 strict）。共同成功条件对固定g=2的载荷分别降低22.93%（15条）和22.07%（28条），但全条件成功分别少1条和2条，开发strict少3条，不能以条件节力替代可靠性验收。与独立无触觉比较，验证+4 task/+6 strict、共同10条节力4.78%，开发+12 task/+11 strict、共同16条节力9.34%；与探测后固定增益比较，触觉成功更多，但共同成功条件载荷分别高7.82%和6.66%。两组reset核对全零。保留这些正负结果，不能将单纯延长预算解释为已解决问题；在这一历史阶段留出数据尚未打开，验收仍pending。

随后第三阶段各追加262144至累计786432，P75分支继续原目标，P200分支仅提高该阶段失败惩罚。P75仍未通过：验证15/16、13.1236 N，开发28/32、13.3182 N；其无触觉仅3/16与4/32，不能把约4.52 N低载荷当作有效节力。P200触觉验证16/16 task/strict、14.1043 N，比原固定g=2降低16.52%；开发31/32 task/strict、14.3588 N，仅200 g/μ=.2失败。此结果允许进入预声明的fresh seed1/2/3确认，**不等于完成最终验收**，也不临时增加开发集32/32门槛。

P200的探测后固定增益验证同为16/16 task、但strict仅15/16，平均12.9082 N低于动态策略；开发则只有23/32。动态反馈不能概括为所有条件下更省力。另有150 g/μ=.2在原保持区间通过后，于31.900 s失去全部接触；末帧31.9333 s木块底面仍高于桌面约6.2177 cm，未观测到跌破5 cm。这是保持区间外的尾帧接触丢失，不能改判原保持失败，也不能宣称终止后仍能持续稳定夹持。完整细节保存在`diagnostics/repaired_normalized_p200_evaluation_review.json`。当时登记的fresh确认须逐seed从新权重完成P75/P75/P200三阶段，不能复用seed0权重或选择中间checkpoint；留出条件保持未打开直至下述完整确认与内部复核结束。

## 三种子冻结确认（2026-10-25）

seed1/2/3 的 tactile 与 no_tactile 分别从头训练，配对初始化相同，均依次完成 262144 步 P75、262144 步 P75、262144 步 P200。共六个最终 786432 步模型，未挑选中间 checkpoint 或最佳 seed。最终模型保存在 `outputs/residual_rl_20261025_v3/confirmation_normalized_p200_seeds123/confirm_normalized_p200_seed{1,2,3}_{tactile,no_tactile}_stage3_786432/final_model.zip`。训练配方及全部命令见 `diagnostics/confirmation_seeds123_protocol.json`；48 次历史训练、39 套评估均保留在自动汇总报告中。

全部验证/开发评估结束、模型及执行完整性复核完成后，才固定开启 75/125/175 g × μ=.3/.5/.7 共9组留出条件。每种子使用自己的初始模型、触觉/无触觉最终模型，以及 fixed2、fixed3、四种触觉干预、reset，共10个控制器。物理、模型、动作与评价标签在报告整理之前保持冻结。原门槛逐seed分别为9/9任务成功、相对fixed2均载荷降低至少10%、每材料接触覆盖至少99%、最长失持不超过50 ms，没有新增validation16/16入口门槛，也没有合并种子补救失败。

| 训练 seed | 触觉 task / strict | 无触觉 task | 探测后固定增益 task | 触觉均载荷 | 相对 fixed2 降低 | 共同成功时相对无触觉载荷变化 |
|---|---|---|---|---|---|---|
| 1 | 9/9 / 9/9 | 8/9 | 5/9 | 12.520 N | 26.11% | −21.71%（8组） |
| 2 | 9/9 / 9/9 | 9/9 | 7/9 | 13.645 N | 19.47% | −45.45%（9组） |
| 3 | 9/9 / 8/9 | 3/9 | 6/9 | 12.498 N | 26.24% | +13.32%（3组） |

三seed均通过原工程门槛，且持续反馈均比探测后保持增益成功更多。但不能说始终更省力：seed3主要提高成功率，在双方共同成功的3组上反而用力更多。载荷为五个触觉指节接触力模之和在整个残差控制区间的均值，不是指腹纯法向力；三个seed的validation全手接触回放均未发现原统计遗漏，且与原动作/物理结果逐点一致，这不替代heldout全手或300 Hz逐子步核验。

**保留末端失持边界。** seed3 的175 g、μ=.3在保持区间末样本只剩一个支撑指节，原区间权重计入8.33 ms；覆盖率99.722%，仍通过99%/50 ms门槛，但strict失败。31.900 s时已失去全部接触并下落，末帧31.933 s底面为5.455 cm，尚未观测到跌破5 cm。不能把3秒任务通过解释为终止后持续稳定抓持。

较宽范围也保留负结果：validation触觉各seed为14/16、16/16、15/16，无触觉为15/16、16/16、5/16；development触觉为27/32、30/32、28/32，无触觉为27/32、32/32、7/32。开发材料用于训练，验证材料曾反复用于方案选择。三个确认seed共用9组留出材料，不能当作27种独立材料，也不能由此声明总体95%可靠性、未知形状或真机泛化。

可核查资料（均位于 `outputs/residual_rl_20261025_v3/`）：

- `diagnostics/confirmation_acceptance_review.json`：受限验收结论及绑定的证据SHA。
- `diagnostics/confirmation_heldout_audit.json`、`confirmation_heldout_independent_review.json`：逐材料原指标重算；独立审计使用整数时间权重复核末端边界。
- `diagnostics/confirmation_heldout_review.json`、`confirmation_heldout_launch_record.json`、`confirmation_heldout_execution/`：测试前固定选择、命令和实际执行日志。
- `diagnostics/confirmation_fresh_seed_comparison.png`、`confirmation_heldout_comparison.png`：所有确认seed的对照图。
- `ADAPTIVE_REPORT.zh-CN.md`：全历史汇总。默认报告不读取留出结果；显式纳入及审核证据使用：

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.adaptive_report \
  --root outputs/residual_rl_20261025_v3 --include-heldout
```

## v2 运行约定

- 物理 300 Hz，触觉 60 Hz，策略 15 Hz。每次 `step` 推进 20 个物理步，得到 4 帧新触觉。
- 观测：过去 12 帧 `5×120×3` 触觉，关节位置和速度，当前/下一时刻名义目标，实际增益，阶段与进度。质量、摩擦、物体位姿和求解器接触不进入策略。
- 动作：2 维 `[-1,1]` 映射到 `[1,2]` 倍增益；`-1` 是原增益，`0` 是 1.5 倍。每次动作最多变化 0.25 倍；阻尼随增益平方根变化；力矩上限保持原值。这不是直接力控制。
- 原示范总长 31.875 秒。按绝对时间线性插值，在 15 Hz 采样并保持目标；这改变了原来的 40 Hz 零阶保持，必须使用新频率的对照。
- 首次物理回放前缀至 19.866667 秒（预紧前最近的动作边界），保存同一批环境的完整 `SceneCheckpoint` 和触觉历史。每回合恢复此前缀，之后只运行预紧、试抬、抬升与保持。最后一个完整动作步终止在 31.933333 秒。
- 显式启用 Genesis 确定性求解选择。普通 `get_state()/reset()` 的 `SimState` 不足以精确续接本任务；必须同时恢复材料、控制、求解器 warm-start 和派生状态，恢复后不重新计算质量。应用层触觉历史另行还原。独立诊断中，固定求解器加完整检查点恢复使重复轨迹的高度和接触载荷差从厘米级降为逐点零。
- 质量按种子在每条并行环境构建时采样，此后该环境反复使用同一质量；每条环境的前缀使用其实际质量生成。`set_mass` 同步缩放惯量并读回核验。运行时的等效密度是质量/体积，构建时的 `material.rho` 不是各环境密度读数。
- 接触摩擦两侧同时设置，并逐次核对真实接触对系数。无粘附、无中途物体位姿改写。
- 成功要求：原最终保持区间 `[28.875,31.875]` 的每个 60 Hz 检查点，旋转后木块底面距桌面至少 5 cm，至少两个触觉所在指节存在有效物理接触，且无机器人与桌面/自身的有效碰撞。
- 滑移评价用物理接触点的切向相对速度；载荷是接触力模长之和，不是法向夹持力，也不是标定后的真实传感器读数。

## 命令

从仓库根目录运行；输出目录必须不存在。首次依赖编译可能较慢。

```bash
.venv/bin/python -m unittest ur7e_xhand_sim.residual_rl.test_core -v

.venv/bin/python -m ur7e_xhand_sim.residual_rl.train train \
  --output outputs/my_tactile_pilot --total-timesteps 32768 \
  --n-envs 8 --n-steps 256 --batch-size 128 --seed 0 \
  --action-std-init 0.1 \
  --mass-range-kg 0.1 0.2 --friction-range 0.8 0.8

.venv/bin/python -m ur7e_xhand_sim.residual_rl.train eval \
  --output outputs/my_tactile_eval --controller model \
  --model outputs/my_tactile_pilot/final_model.zip \
  --n-envs 8 --episodes 8 --seed 10000 \
  --mass-range-kg 0.2 0.2 --friction-range 0.8 0.8
```

固定对照使用 `--controller nominal|mid|strong|reactive`，不传 `--model`。训练无触觉对照时加 `--ablate-tactile`：保持网络结构与预算相同，在特征提取器内将触觉输入置零。其关节状态仍可用于反馈，不能称作纯开环。

每次训练保存 `initial_model.zip`、`final_model.zip`、周期检查点、训练指标和逐回合指标。评估保存 `evaluation.json`、`episodes.jsonl`、`step_trace.jsonl` 和 `trace.npz`。`environment/environment_protocol.json` 描述物理协议；`manifest.json` 与 `source_snapshot/` 保存源码、示范及依赖版本。独立进程评估，不在同一进程同时创建两个 Genesis 环境后关闭其中一个。

两组模型训练完成后，用同一份物理前缀配对比较全部控制器，并在末尾重复原增益回放检查重置一致性：

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.benchmark \
  --output outputs/my_comparison \
  --tactile-run outputs/my_tactile_pilot \
  --no-tactile-run outputs/my_no_tactile_pilot \
  --mass-range-kg 0.1 0.2 --seed 10000 --n-envs 8
```

2026-10-25 首轮完整训练和对照保存在 `outputs/residual_rl_20261025_v2/`，报告为该目录下的 `REPORT.zh-CN.md`，模型分别在 `ppo_tactile_seed0/final_model.zip` 和 `ppo_no_tactile_seed0/final_model.zip`。重新汇总报告：

```bash
.venv/bin/python -m ur7e_xhand_sim.residual_rl.report \
  --root outputs/residual_rl_20261025_v2
```

旧目录 `outputs/residual_rl_20261025/` 保留重置问题的诊断记录。v2 同时修改了求解器选择、完整状态恢复和探索噪声初始化，不能将两版差异归因于其中单独一项。

## 如何解释第一轮结果

未训练网络的动作接近 0，因此初始增益接近 1.5 倍，可能已经成功。必须比较初始网络、固定 1/1.5/2 倍、规则反馈、训练策略和独立重训的无触觉策略。不能把初始化的夹持效果算成强化学习收益。

相同条件的并行环境和重复重置不代表独立的多样实验；当前只是材料变化下的局部学习试验。正式泛化测试还需要更多初始状态、运动时序、触觉噪声/延迟、扰动和独立训练种子。当前剪切触觉由相对运动构造，仍需仅法向触觉的消融与真实数据验证。没有经过这些实验，不应声称已获得通用或可直接上真机的触觉控制器。
