"""Build the evidence report for the isolated October 28 tactile validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tactile-validation-report-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    report = root / "report"
    report.mkdir(exist_ok=True)
    before = read(root / "force_before/result.json")
    after = read(root / "force_final/result.json")
    a, b = before["records"], after["records"]
    assert len(a) == len(b)
    commands_same = all(
        x[k] == y[k] for x, y in zip(a, b)
        for k in ("imposed_overlap_m", "imposed_tangent_velocity_m_s")
    )
    force_error = max(float(np.max(np.abs(np.array(x["physical_force_on_sensor_body_N"])
                                        - y["physical_force_on_sensor_body_N"])))
                      for x, y in zip(a, b))
    comparison = dict(same_imposed_commands=commands_same,
                      physical_force_max_difference_N=force_error,
                      before_checks=before["checks"], after_checks=after["checks"],
                      before_sensor_sha=before["source_sha256"],
                      after_sensor_sha=after["source_sha256"],
                      scope="Matched prescribed actions; force magnitudes are not calibrated taxel forces.")
    assert commands_same and force_error == 0 and all(after["checks"].values())
    (root / "paired_force_final_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    colors = ["#456b86", "#b3423b", "#198071"]
    for ax, velocity in zip(axes, (-0.5, 0.5)):
        old = next(x for x in a if x["imposed_tangent_velocity_m_s"] == velocity)
        new = next(x for x in b if x["imposed_tangent_velocity_m_s"] == velocity)
        values = [new["physical_force_on_sensor_body_N"][0],
                  old["isolated_shear_term_nominal_N"][0], new["isolated_shear_term_nominal_N"][0]]
        ax.bar([0, 1, 2], values, color=colors, width=.6)
        ax.set_xticks([0, 1, 2], ["Solver force\n(reference)", "Taxel before", "Taxel final"])
        ax.axhline(0, color="#333", linewidth=.8)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"Object moves {'left' if velocity < 0 else 'right'}")
        ax.set_ylabel("X component: physical N / nominal taxel N")
        ax.grid(axis="y", alpha=.15)
        for i, v in enumerate(values):
            ax.text(i, v + (.12 if v > 0 else -.2), f"{v:+.3f}", ha="center", fontsize=11)
    fig.suptitle("Force on the sensor: shear direction corrected\nCompare signs only; taxel magnitudes are uncalibrated", fontsize=12)
    fig.savefig(report / "shear_direction_final.png", dpi=170)
    plt.close(fig)

    scan = read(root / "controlled_scan_after_box_fix/analysis.json")
    scan_before = read(root / "controlled_scan/analysis.json")
    missing = []
    positive_samples = 0
    for path in sorted((root / "controlled_scan_after_box_fix").glob("*/records.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            positive_samples += row["total_contact_load_N"] > 1e-8
            if row["contact_count"] > 0 and not row["active_taxel_ids_1based"]:
                missing.append({k: row[k] for k in ("finger", "shape", "target", "nominal_gap_mm",
                                                   "steps_after_initial_placement", "total_contact_load_N")})
    loaded_missing = [r for r in missing if r["total_contact_load_N"] > 1e-8]
    summary = dict(
        sensor_version="kinematic_force_on_sensor_v3", sensor_sha256=after["source_sha256"],
        scope="Thumb/index; plate/sphere; prescribed initial gaps; 1 and 5 physics steps. Not free grasp validation.",
        samples=sum(v["samples"] for v in scan.values()),
        analytic_mask_matches=sum(v["analytic_mask_matches"] for v in scan.values()),
        all_finite=all(v["finite"] for v in scan.values()),
        no_contact_zero_response=all(v["no_contact_zero_response"] for v in scan.values()),
        positive_load_threshold_N=1e-8, positive_load_samples=positive_samples,
        positive_load_but_zero_response_samples=len(loaded_missing),
        zero_load_contact_record_and_zero_response_samples=len(missing) - len(loaded_missing),
        no_response_contact_records=missing,
        matched_scan_dynamics_unchanged=all(
            v["paired_comparison"]["max_pose_component_error"] == 0
            and v["paired_comparison"]["max_total_contact_force_error_N"] == 0
            and v["paired_comparison"]["all_contact_records_exactly_equal"] for v in scan.values()),
        final_force_checks=after["checks"], formal_training_started=False,
        limitations=["Remaining tip coverage gap", "Uncalibrated force magnitudes",
                     "Shear is relative-velocity proxy, not static friction load",
                     "Non-BOX mesh SDF remains approximate", "Other fingers and dynamic grasp coverage not established"],
    )
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = "".join(f"<tr><td>{title}</td><td>{scan_before[name]['analytic_mask_matches']} / {scan[name]['samples']}</td>"
                   f"<td>{scan[name]['analytic_mask_matches']} / {scan[name]['samples']}</td></tr>"
                   for name, title in [("thumb_plate", "拇指 × 小平板"), ("index_plate", "食指 × 小平板"),
                                       ("thumb_sphere", "拇指 × 球体"), ("index_sphere", "食指 × 球体")])
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XHand 触觉受控验证 · 2026-10-28</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f6;color:#24313d;font:16px/1.75 system-ui,sans-serif}
main{max-width:1080px;margin:40px auto;padding:0 24px 50px}h1{font-size:32px;line-height:1.3}h2{font-size:23px;margin:0 0 12px}
.eyebrow{letter-spacing:.08em;color:#53727a;font-size:13px}.lead{font-size:19px}.panel{background:white;border:1px solid #dde4e7;border-radius:12px;padding:25px;margin:22px 0}
.notice{border-left:5px solid #c4852e;background:#fff8eb;padding:16px 20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric{padding:16px;background:#eaf2ef;border-radius:8px}.metric strong{font-size:29px;display:block;color:#197365}
table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border-bottom:1px solid #dfe5e8;padding:10px;text-align:left}th{background:#eef3f5}
figure{margin:22px 0}img{width:100%;height:auto;display:block}figcaption,.muted{color:#586c78;font-size:14px}code{overflow-wrap:anywhere;background:#edf1f4;padding:2px 5px}a{color:#126b83}li{margin:7px 0}.small{max-width:800px;margin:auto}
@media(max-width:650px){main{margin:20px auto;padding:0 14px}.cards{grid-template-columns:1fr}.panel{padding:18px}h1{font-size:26px}}
</style><main><p class="eyebrow">XHAND · CONTROLLED TACTILE VALIDATION · 2026-10-28</p>
<h1>两处触觉计算问题已修复；<br>拇指末端仍有覆盖缺口</h1>
<p class="lead">本轮可以确认指定测试下的测点激活与方向计算已通过复测，不能据此认定所有物理接触均可被感知，或触觉数值已经等于真实力。</p>
<div class="cards"><div class="metric"><strong>280 / 280</strong>修复后激活测点集合符合解析预测</div><div class="metric"><strong>0</strong>配对扫描的位姿／接触力变化</div><div class="metric"><strong>3 帧</strong>拇指末端带载但无触觉响应</div></div>
<section class="panel"><h2>测试怎么做</h2><p>保留真实加载后的碰撞网格与固定子组件，隔离固定拇指和食指。用直径 45 mm 球体及 8 × 8 × 6 mm 小平板，扫描指腹中央、边缘、指尖；拇指额外扫描原球体示范的接触位置。物体为 50 g。</p>
<p>每个位置设置 10 档初始间隙，运行真实碰撞和动力学，在第 1、5 步读取物理接触、测点响应与解析距离。共 280 个样本，300 Hz 物理步进。这是规定初始位姿的隔离诊断，并非完整抓取成功率实验，也不是力标定。</p></section>
<section class="panel"><h2>发现并修复的两处代码问题</h2>
<ol><li><b>切向触觉方向反了。</b>旧代码的切向、扭转项与“作用在传感器上的力／力矩”约定相反。实物接触仿真中，物体向右运动时物理摩擦对传感器向右，旧触觉却向左。已修正两项符号。当前应用扭转系数为零，旧抓取的扭转项未受影响。</li>
<li><b>小 BOX 在距离网格外漏测。</b>该小平板的距离网格只向外延伸约 0.80 mm，触觉探针半径却为 1.5 mm。旧网格外近似把实际 1.14–1.21 mm 的距离估成 5.71–8.30 mm。现改用传感器内部的 BOX 精确距离与法向；全局物理求解器保持原实现。</li></ol>
<figure><img src="shear_direction_final.png" alt="修复前后切向触觉与物理摩擦方向比较"><figcaption>仅比较方向；蓝色是物理求解器接触力，红／绿色是未标定的触觉估计，幅值不应直接等同。最终版本已再次运行真实接触测试。</figcaption></figure>
<table><thead><tr><th>扫描组</th><th>BOX 修复前符合解析预测</th><th>最终版本符合解析预测</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>全部数据有限，无物理接触记录的样本全部零响应。配对扫描的物体位姿、接触记录和物理接触力逐条完全一致。另有 9 项 CPU 编译／真实接触回归全部通过，覆盖旋转、单／双环境及距离／法向梯度。</p>
<figure><img src="../controlled_scan_after_box_fix/controlled_scan_active_points.png" alt="拇指与食指在球体和小平板上的响应点数"><figcaption>实线为记录，虚线为解析重叠加物理接触门控的预测，两者重合。横轴为规定的初始间隙，并非运行后的实际压入量；图中仅展示第 1 步。</figcaption></figure></section>
<section class="panel"><h2>为什么原球体示范的拇指触觉偏弱</h2>
<p>原示范的球体主要接触拇指末端的一个小固定碰撞组件 <code>thumb_rota_tip</code>，而测点集中在另一主表面。最近测点到球面的间隙约 1.43 mm，几乎达到 1.5 mm 探针半径，所以只有少量、浅重叠的测点响应。原始点位与厂商坐标数据一致，未发现整体坐标错位；球体曲率和接触位置让这个覆盖问题更明显。</p>
<figure class="small"><img src="../geometry/thumb_sphere_tip_detail.png" alt="原球体接触位点与拇指触觉探针位置关系"><figcaption>原示范某帧的局部投影示意：红点为物理接触点，蓝点为测点中心，淡蓝圆表示 1.5 mm 虚拟探针范围。这是几何解释，不是软皮肤受力仿真。</figcaption></figure>
<div class="notice"><b>仍有覆盖缺口：</b>最终扫描中，同一末端目标有 8 帧“存在接触记录但零触觉”。其中只有 3 帧带正载荷（约 0.023–0.047 N），另 5 帧载荷已为零。这里是 3 个带载漏感知样本，并非 8 个独立空间盲区。解析几何也预测其测点未覆盖，不能靠读数程序修复消除。</div>
<p>当前没有删除厂商碰撞组件、移动测点或任意放大探针。是否应由真实外层触觉皮肤传递这部分载荷，仍需要硬件接触实验或皮肤结构证据。</p></section>
<section class="panel"><h2>对后续训练意味着什么</h2>
<ul><li>本轮不启动 PPO。已有动作示范可以回放，但旧的原始触觉数据保持原版本，不能直接当成修复后的数据使用。重新采样应另存，记录新版本。</li>
<li>当前模型为 <code>kinematic_force_on_sensor_v3</code>。法向信号来自探针重叠；切向信号主要是相对速度代理，静止承重时不能代表真实静摩擦载荷。评估平均接触力时应使用物理求解器的接触力，不能把测点和当作已标定的牛顿值。</li>
<li>本轮仅验证拇指／食指、球／BOX 与这些测试位置。圆柱、六棱柱等 mesh 距离查询仍为网格近似，其他指头和动态抓取的覆盖也未由本轮证明。</li>
<li>下一步应先确认末端外层传力范围，再重放示范采集新触觉，避免模型把“零触觉”学成“没有接触”。</li></ul></section>
<section class="panel"><h2>原始证据与复现记录</h2><ul>
<li><a href="../validation_summary.json">最终验证汇总与带载漏感知样本</a> · <a href="../paired_force_final_comparison.json">最终方向配对测试</a></li>
<li><a href="../controlled_scan_after_box_fix/analysis.json">280 样本扫描分析</a> · <a href="../controlled_scan_after_box_fix/manifest.json">扫描版本与参数</a></li>
<li><a href="../geometry/actual_engine_mesh_alignment.json">实际引擎碰撞网格审计</a> · <a href="../vendor_point_check.json">测点坐标比对</a></li>
<li><a href="../data_compatibility.json">旧数据兼容性与冻结哈希</a></li>
<li><a href="../../tactile_validation_20261028/compiled_regression_after_box_fix.log">9 项回归日志</a> · <a href="../../tactile_validation_20261028/regression_after_box_manifest.json">命令与代码哈希</a></li>
</ul><p class="muted">最终传感器源码 SHA256：<code>__SHA__</code>。原始扫描、示范和修复前证据均保留。</p></section></main></html>'''
    (report / "index.html").write_text(html.replace("__ROWS__", rows).replace("__SHA__", after["source_sha256"]))
    evidence = [root / "validation_summary.json", root / "paired_force_final_comparison.json",
                root / "controlled_scan_after_box_fix/manifest.json", root / "force_final/result.json",
                root / "data_compatibility.json", report / "index.html", report / "shear_direction_final.png"]
    (report / "manifest.json").write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in evidence}, indent=2) + "\n")
    print(report / "index.html")
    print(json.dumps({k: v for k, v in summary.items() if k != "no_response_contact_records"}, indent=2))


if __name__ == "__main__":
    main()
