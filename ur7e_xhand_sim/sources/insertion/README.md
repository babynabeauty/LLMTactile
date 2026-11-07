# 插入任务资产

这里打包了 UniVTAC 的 `TestTube / TestTubeBase / TestTubeSlot` 和 ContactWorld 发布的 `USB_plug / USB_socket`。运行时直接使用包内 OBJ 和 URDF，不依赖原仓库、Isaac Sim、USD 或 `/home/sai/zx` 下的文件。所有坐标使用米，缩放为 1。

`ur7e_xhand_sim.insertion_assets.load_insertion_assets(task)` 返回移动物体、固定支架/目标座、几何参数及推荐的 Genesis 导入参数。场景布局由环境代码指定。固定孔座必须使用返回的 `morph_options` 和 `material_options`；默认凸包会封住插孔。

| 任务 | 移动物体外包围尺寸 | 目标几何 | 默认移动质量 |
|---|---|---|---|
| `tube_insert` | 20 × 20 × 130 mm | 原始直径 21 mm 的倾斜孔，轴线相对 +Z 向 −X 倾斜 30° | 30 g |
| `usb_insert` | 12.17 × 30 × 120.958 mm | 9.18 × 23.4 mm 矩形孔及内部舌片 | 50 g |

质量是本地仿真假设，不是来源资产的标定值。试管使用原始闭合网格体积与惯量；USB 使用下面所述闭合碰撞代理的体积与惯量。URDF 写入显式惯量，`align=False` 保留原始局部坐标原点。两种物体的插入端都位于局部 `[0, 0, 0]`，物体沿局部 +Z 延伸。

## 视觉与碰撞

- 试管、初始支架及倾斜目标座：保留原始 USD 经完整坐标变换后的网格、原始尺寸和基础颜色。独立的 `*_visual.obj` 保留原始逐面角点法线和未焊接的角点，避免硬边被错误平滑；碰撞使用单独焊接的 OBJ。试管本身是闭合凸网格。支架和目标座使用原始非凸闭合表面的 SDF，不包覆、不简化、不凸化，目标 SDF 单元为 0.5 mm，最大分辨率为 256。
- USB 外观：保留发布 OBJ 的顶点、面和法线，只移除外部 MTL 引用，并使用原始 URDF 的蓝/红色。插座碰撞仅合并重复接缝顶点，原始矩形孔和内部舌片均保留；目标 SDF 单元为 0.25 mm，最大分辨率为 320。
- USB 插头碰撞：原 OBJ 包含开面和退化重复面，因此使用明确闭合的刚性代理。握持主体为原主体凸壳，底部裁到局部 z=22 mm；金属壳由四个 0.6 mm 厚壁组成，外截面 6 × 22 mm、内截面 4.8 × 20.8 mm，长度 22 mm；另保留内部塑料体的长方体代理。原有锁扣开口、柔性金属小片及握持主体内部空腔被简化。视觉模型保持原样。该代理适用于刚性抓取/对准插入，未建模真实 USB 的金属弹性锁扣。

USB 目标方向是相对插座绕局部 Z 旋转 180°（wxyz=`[0,0,0,1]`），使插头内部塑料到 +X 一侧，避开插座 −X 一侧的舌片。只对齐长轴、忽略此方向，会让塑料体和舌片互相穿透。

几何验证在试管插入 40 mm、USB 插入 17 mm 的参考姿态下，对每个碰撞件的所有顶点及 4,000 个固定种子表面样本检查是否进入目标座实体。两个正确姿态均为 0 个穿透采样点；USB 错误朝向存在穿透。此采样检查不是动力学执行成功率，也不能替代真实接触求解验证。

## 来源与转换

`manifest.json` 记录原文件 SHA-256、包内文件 SHA-256、来源、局部坐标、尺寸、惯量、碰撞近似说明及目标方向。

- UniVTAC：<https://github.com/univtac/UniVTAC>，提交 `d541e5568227ca3b66104d294f63c80acad7c52c`。根目录 LICENSE 为 Apache-2.0，已保留在 `LICENSE-UniVTAC.txt`；上游 README 同时写有 MIT，与 LICENSE 不一致。
- ContactWorld：<https://github.com/PokuangZhou/ContactWorld>，资产来自 <https://huggingface.co/datasets/Pokuang/ContactWorld> 的 `released_assets/industreal`。所检查仓库根目录没有 LICENSE，资产发布包的许可未确认，这里不推定额外许可授权。

`rebuild_assets.py` 是离线转换脚本，需要显式提供来源路径。默认使用 `pxr` 读取 UniVTAC USD；也可通过 `--tube-npz-dir` 传入含完整变换后顶点/面/颜色的中间导出。运行仿真不调用此脚本。
