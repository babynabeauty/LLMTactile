"""Method flow chart of the risk MVP (matplotlib only). python docs_figs/method_flow.py"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

STATUS = {'done': ('#e7f5ea', '#2b8a3e', '已完成'), 'doing': ('#fff4e0', '#e8590c', '进行中'),
          'todo': ('#f1f3f5', '#868e96', '未做'), 'synthetic': ('#e7f0fb', '#1c7ed6', '仅合成数据验证')}

fig = plt.figure(figsize=(26, 17))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 260)
ax.set_ylim(0, 170)
ax.axis('off')
boxes = {}


def box(key, x, y, w, h, title, lines, status='done', fs=10.5, title_fs=12.5, dashed=False):
    face, edge, _ = STATUS[status]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=face, ec=edge,
                                lw=2.2, ls='--' if dashed or status == 'todo' else '-'))
    ax.text(x+1.5, y+h-1.6, title, fontsize=title_fs, weight='bold', va='top', color='#212529')
    ax.text(x+1.5, y+h-5.6, '\n'.join(lines), fontsize=fs, va='top', color='#343a40', linespacing=1.35)
    boxes[key] = (x, y, w, h)


def anchor(key, side):
    x, y, w, h = boxes[key]
    return {'l': (x-.4, y+h/2), 'r': (x+w+.4, y+h/2), 't': (x+w/2, y+h+.4), 'b': (x+w/2, y-.4)}[side]


def arrow(a, b, text='', color='#495057', ls='-', rad=0., fs=9.5, lw=1.8, toff=(0, 1.2)):
    pa = anchor(*a) if isinstance(a, tuple) and isinstance(a[0], str) else a
    pb = anchor(*b) if isinstance(b, tuple) and isinstance(b[0], str) else b
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle='-|>', mutation_scale=18, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((pa[0]+pb[0])/2+toff[0], (pa[1]+pb[1])/2+toff[1], text, fontsize=fs, color=color, ha='center',
                va='bottom', bbox=dict(fc='white', ec='none', alpha=.85, pad=.6))


def path(points, text='', color='#495057', ls='-', lw=1.8, fs=9.5, tpos=None):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle='round')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=18, color=color, lw=lw, ls=ls))
    if text:
        tx, ty = tpos or points[len(points)//2]
        ax.text(tx, ty, text, fontsize=fs, color=color, ha='center', va='center',
                bbox=dict(fc='white', ec='none', alpha=.9, pad=.6))


def band(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=2', fc=color, ec='none', alpha=.35))
    ax.text(x+1.2, y+h-1.2, label, fontsize=15, weight='bold', va='top', color='#495057')


# ---------------------------------------------------------------- title + legend
ax.text(130, 167.5, '灵巧手安全分力：基于删失损伤信念的机会约束分力 —— 方法流程（risk_mvp，小规模仿真版）',
        fontsize=21, weight='bold', ha='center', va='top')
ax.text(130, 163, '核心：力不够 → 打滑，可检测、可挽回（F_min 在线可测）；力太大 → 捏坏，不可逆，且"没坏"只说明 F_safe > F（右删失）'
        '  ⇒  对每块区域的 F_safe 维护信念，按信念分力', fontsize=12.5, ha='center', va='top', color='#495057')
lx = 198
for i, k in enumerate(['done', 'synthetic', 'doing', 'todo']):
    face, edge, name = STATUS[k]
    ax.add_patch(FancyBboxPatch((lx+i*15.5, 154.5), 3.2, 2.6, boxstyle='round,pad=0.2', fc=face, ec=edge, lw=2,
                                ls='--' if k == 'todo' else '-'))
    ax.text(lx+i*15.5+4.2, 155.8, name, fontsize=11, va='center')

# ---------------------------------------------------------------- bands
band(2, 112, 92, 40, '① 离线准备（每个物体一次）', '#d0ebff')
band(97, 112, 80, 40, '② 慢回路：大模型看图（每个物体 / 实例一次）', '#e5dbff')
band(180, 112, 78, 40, '③ 风险模型（小 Transformer，需训练）', '#ffe3e3')
band(2, 44, 256, 65, '④ 快回路：一轮抓取', '#fff3bf')
band(2, 2, 256, 39, '⑤ 数据与评估', '#d3f9d8')

# ---------------------------------------------------------------- ① offline
box('obj', 4, 126, 28, 22, '物体 objects.py', [
    '8 个 YCB 物体：草莓 李子 桃 柠檬',
    '065-a/d 杯子 泡沫砖 网球',
    '类别 → 整体强度范围（唯一人工输入）',
    '  软水果 0.5–2 N  硬水果 2–5 N',
    '  薄杯 1–4 N  结实 5–10 N',
    '可画可见记号（淤伤/压痕）'])
box('reg', 34, 126, 28, 22, '分区 + 阈值 regions.py', [
    '凸包表面 FPS → K=5 块',
    '强制 食指/中指 落在不同区域',
    '实例阈值 = 整体强度 ×',
    '  相关的区域系数（GP）',
    '20% 概率一个隐藏薄弱区(×0.3–0.6)',
    '→ 只用于判坏和评估，模型不可见'])
box('grasp', 64, 126, 28, 22, '抓取姿态 + 快照', [
    'pregrasp_search：三指拿模板',
    '  + 45–75 个偏移并行试抓',
    '  合拢至接触 → 轻触 → 一起加力',
    'snapshot：按 swap_window 选姿态',
    '  (分力器评估两侧可调空间)',
    '保存"三指已接触"的完整状态'])
box('render', 4, 114, 64, 9.5, '多视角带编号图 render.py / objects_check.py', [
    '4 方位 × 35° 俯角，编号标在可见部分；"有线索"时薄弱区带记号'],
    fs=10, title_fs=11.5)
arrow(('obj', 'r'), ('reg', 'l'))
arrow(('reg', 'r'), ('grasp', 'l'))

# ---------------------------------------------------------------- ② VLM
box('vlm', 99, 126, 37, 22, 'Astra（gpt-6-astra via codex exec）', [
    '输入：多视角编号图（+ 物体名）',
    '问：每块区域 脆 / 一般 / 结实',
    '    + 理由 + 把握程度',
    '    + 两两比较 哪块更结实',
    '    + 物体整体强度',
    '只做判断，不出动作'], status='todo')
box('vlmout', 139, 126, 36, 22, '区域判断 JSON', [
    'region k: {标签, 把握 0–1, 理由文本}',
    '理由 → 冻结文本编码器 → 向量',
    '物体整体强度档位',
    '',
    '（慢回路的换手型 / 放回去：',
    '  本次不做）'], status='todo')
box('slow', 99, 114, 76, 9.5, '为什么需要外观线索（已验证）', [
    '阈值若与外观无关，翻转大模型标签只改变预测约 2% → 大模型无信息；所以用真实物体 + 可见记号'],
    status='done', fs=10, title_fs=11.5)
path([(68.4, 118.7), (95, 118.7), (95, 131), (98.6, 131)], '编号图', color='#7048e8', tpos=(82, 118.7))
arrow(('vlm', 'r'), ('vlmout', 'l'), color='#7048e8')

# ---------------------------------------------------------------- ③ risk model
box('cards', 182, 126, 36, 22, '输入卡片', [
    '区域卡（每块一张）：大模型标签 +',
    '  把握 + 理由向量 + 几何 8 维 +',
    '  触觉 3 维 + 受力历史(GRU)',
    '档案卡（0–10 张，无位置编码）：',
    '  {区域几何, 当时标签, 最大力,',
    '   没坏@L / 在(L,U]坏}  信任卡 ×1'], status='synthetic')
box('tf', 220, 126, 36, 22, 'Transformer → 风险曲线', [
    '2 层，128 维，4 头；注意力掩码：',
    '  档案↔档案，信任→档案，',
    '  区域→区域/档案/信任',
    '输出每块 H_r(f)，0–10 N 101 点',
    '  单调且凸（非负递增斜率）',
    'P(坏 | f) = 1 − exp(−H_r(f))'], status='synthetic')
box('loss', 182, 113.4, 74, 10.6, '训练：删失负对数似然（train_risk.py）', [
    '没坏@L: H(L)；在(L,U]坏: H(L)−log(1−e^{−(H(U)−H(L))})；开卷版对照用真实阈值',
    '合成数据：档案 0→10 条，误差 0.71→0.17；真实数据训练未做'],
    status='synthetic', fs=9.4, title_fs=11)
arrow(('vlmout', 'r'), ('cards', 'l'), color='#7048e8')
arrow(('cards', 'r'), ('tf', 'l'), color='#c92a2a')

# ---------------------------------------------------------------- ④ fast loop
box('restore', 4, 82, 30, 22, '开始一轮 episode.py', [
    '恢复快照（三指已轻触）',
    '每个 env 设定：',
    '  分力方式（6 种之一）',
    '  隐藏区域阈值',
    '  物体质量、接触摩擦',
    '时间线：预加力 0.5 s → 抬 5 cm 1 s',
    '        → 保持 2 s'], fs=10)
box('contact', 38, 82, 33, 22, '读接触（求解器）scene.py', [
    '每根手指：法向力 fn、接触点、',
    '  内法向、切向力',
    '接触点 → 物体坐标系 → 所在区域',
    '物体位姿 → 重力在物体系中的方向',
    '（触觉阵列在仿真中只有 1 点/指，',
    '  暂只用切/法向力比）'], fs=10)
box('alloc', 75, 74, 52, 30, '分力器 allocator.py（凸优化 SOCP，每 4 tick 解一次，~3 ms）', [
    '变量：3 个接触点上的力向量 f_i',
    '① 托得住：Σf_i = −m(1+10%)g，Σ p_i×f_i = 0',
    '② 硬摩擦锥：‖f_t‖ ≤ μ̂·f_n（μ̂ 保守估计）',
    '③ 软"防打滑余量"：‖f_t‖ ≤ (μ̂/2)·f_n + m_i，罚 10·m_i',
    '   打滑可挽回、捏坏不可 → 危险区域先放弃余量',
    '④ 不捏坏（随方法不同）：',
    '   我们：Σ_r H_r(F_r + 0.1 N) ≤ −log(1−ε)，ε=5%',
    '   oracle：F_r ≤ 真实阈值 − 0.1 N；上限法：F_r ≤ cap',
    '   平均分力：f_n,食指 = f_n,中指；纯触觉：无 ④',
    '⑤ 关节力矩上限 → f_n ≤ 8 N',
    '目标：min Σ f_n + λ‖f − f_prev‖²；无解 → 松弛 + 报告'], fs=9.8)
box('fc', 131, 82, 38, 22, '力控 force_ctrl.py', [
    '目标法向力 → 关节力矩 τ = J_cᵀ(f·n)',
    '  J_c：接触点雅可比（由 link 原点',
    '  雅可比 + 力臂换算）',
    '关节目标 q* = q + τ / kp',
    '  （PD 力矩恰为 τ → 等效力矩源）',
    '+ 每指 PI 用实测法向力修正',
    '5090 实测稳态误差 < 1%'], fs=10)
box('phys', 173, 82, 36, 22, 'XHand + Genesis 物理', [
    '手指关节位置 PD（kp 3×握力增益 3）',
    '转子惯量 0.003（消除合拢尖峰）',
    '关节摩擦 = 0（真机待测）',
    '手臂 IK：抬起从下发目标起步',
    '  （避免 ~4 mm 下垂导致脱手）',
    '1/300 s 物理，5 子步 = 60 Hz'], fs=10)
box('damage', 213, 82, 43, 22, '虚拟损坏判定 damage.py', [
    '区域受力 F_r = 该区域手指法向力之和',
    '  （按快照中手指所在区域计）',
    '持续力 s_t = min(F_t, F_{t−1}, F_{t−2})',
    's_t > 阈值 → 判坏，本轮结束',
    '结果：握住 / 掉落（抬起<3 cm 或脱指）',
    '      / 捏坏（区域、时刻、力值）'], fs=10)
box('archive', 213, 48, 43, 28, '写档案（删失记录）', [
    '每个被碰过的区域一行：',
    '  没坏：F_safe ≥ max s_t   （右删失）',
    '  坏了：max_{t<t*} s_t ≤ F_safe < s_{t*}',
    '        （区间删失，严格成立）',
    '下一次抓同一实例时作为档案卡输入',
    '→ 风险曲线更新（上下文学习，',
    '   不改模型参数）'], fs=10)
box('hazard', 131, 48, 78, 24, '风险曲线 → 分力器（我们的方法）', [
    '每块区域 H_r(f) 的分段线性凸包作为约束 ④；',
    '例（实测接触，0.2 kg，μ̂ 0.8）：薄弱区在食指 → 食指 1.01 N / 中指 1.63 N；',
    '                            薄弱区在中指 → 食指 1.64 N / 中指 1.01 N（不考虑风险时 1.27/1.34 N）',
    '静力学限制：三点接触时每指承担的重量份额固定，f_n ≥ 份额/μ̂，只有"余量"可以挪',
    '抓法决定窗口：拇指须与两指对顶（swap_window 选姿态）'], status='done', fs=10)
arrow(('restore', 'r'), ('contact', 'l'))
arrow(('contact', 'r'), (74.6, 93), '接触几何、区域、位姿')
arrow((127.4, 93), ('fc', 'l'), '每指目标 f_n')
arrow(('fc', 'r'), ('phys', 'l'), '关节目标 q*')
arrow(('phys', 'r'), ('damage', 'l'), '实测接触力')
arrow(('damage', 'b'), ('archive', 't'), '本轮结果', color='#c92a2a')
path([(191, 104.4), (191, 106.6), (54.5, 106.6), (54.5, 104.4)], '回到读接触：下一 tick（60 Hz）', color='#868e96', ls='--',
     tpos=(150, 106.6))
arrow(('archive', 'l'), ('hazard', 'r'), '', color='#c92a2a', ls='--')
path([(140, 72.4), (140, 75.5), (127.4, 75.5)], '约束 ④', color='#c92a2a', tpos=(133, 75.5))
path([(211, 113), (211, 74), (209.4, 70)], 'H_r(f)', color='#c92a2a', tpos=(211, 108))
path([(256.4, 62), (258.6, 62), (258.6, 137), (256.4, 137)], '档案回流（下一次抓取）', color='#c92a2a', ls='--',
     tpos=(251.5, 79), fs=9)
path([(78, 125.6), (78, 110.5), (19, 110.5), (19, 104.4)], '快照', color='#1971c2', tpos=(50, 110.5))

box('swap', 4, 48, 123, 24, '最小验证 swap_check.py（当前步骤）', [
    '同一快照，薄弱区放在食指侧或中指侧；比较 平均分力 / 纯触觉 / oracle（知道真实阈值），每格 16 个 env',
    '窗口人为调窄：质量 0.15 kg，真实摩擦 0.6，μ̂ 0.5，余量系数 2，薄弱阈值 1.2–1.6 N，其余 4–8 N',
    '第 1 轮（065-d）：oracle 力已换边 —— 薄弱在食指：食指 1.30 / 中指 0.93 N；在中指：1.91 / 0.34 N',
    '  问题：抬起瞬间脱手、跨区域叠加 → 已修',
    '第 2 轮：快照误选"两指同向推"的抓法，所需力 ×3 → 已改为 swap_window 选姿态',
    '第 3 轮：待 5090 运行（预期 065-a：平均分力两侧 ~2 N 都捏坏，oracle 两侧都能压到 ~1 N 以下）'],
    status='doing', fs=10)

# ---------------------------------------------------------------- ⑤ data + evaluation
box('collect', 4, 4, 60, 30, '采数据 collect.py（2000–3000 轮）', [
    '每个物体 ~50 组阈值 × 5–10 轮',
    '随机：最大总力（有轻有重，保证部分捏坏）',
    '      食指:中指 分配 2:8 ~ 8:2',
    '保存：物体/实例 ID、各区域阈值（仅评估）、',
    '  大模型判断、各区域受力曲线、',
    '  触觉特征、结果、档案行',
    '→ 用真实数据训练风险模型 ③'], status='todo', fs=10)
box('methods', 67, 4, 66, 30, '6 种方法（硬件、抓法、流程完全相同，只换约束 ④）', [
    '1 平均分力                    2 纯触觉分力（TacDexGrasp 式）',
    '3 静态力上限                  4 手写概率模型（每区对数正态 + 删失更新）',
    '5 我们：训练的风险模型 + 分力器',
    '6 已知真实阈值（上限参照）',
    '',
    '交换薄弱区实验：每实例 薄弱区在 食指侧 / 中指侧，',
    '  连续抓 10 次，档案逐次累积'], status='todo', fs=10)
box('metrics', 136, 4, 60, 30, '指标与关键图', [
    '成功率、捏坏率、掉落率',
    '每根手指力随时间曲线',
    '第 1 次 vs 第 10 次捏坏率（档案有效性）',
    '关键图：横轴 薄弱区在食指侧 / 中指侧',
    '  纵轴 食指、中指用力 → 我们的方法',
    '  两线交叉，1、2 不交叉',
    '辅助：有线索 vs 外观相同；重心/摩擦扰动'], status='todo', fs=10)
box('claims', 199, 4, 57, 30, '最终要证明（成功标准）', [
    '① 力跟着薄弱区换边',
    '② 成功率相近时，捏坏率低于 1/2/3',
    '③ 训练的风险模型优于手写模型 4',
    '④ 同一实例抓得越多，捏坏越少',
    '⑤ 拿掉大模型判断，第 1 次捏坏明显增多',
    '→ 之后：放大规模、真机（含关节摩擦）'], status='todo', fs=10.5)
arrow(('collect', 'r'), ('methods', 'l'))
arrow(('methods', 'r'), ('metrics', 'l'))
arrow(('metrics', 'r'), ('claims', 'l'))
arrow((40, 47.6), (34, 34.4), '同一 episode 流程', color='#2f9e44')

fig.savefig('docs_figs/method_flow.png', dpi=110)
print('ok')
