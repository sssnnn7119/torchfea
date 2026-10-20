# torchfea CAE 图形界面

CAE UI 面向模型定义和静力隐式计算；它使用 `FEAController`、`Assembly` 和 `Part` 的原生对象，不会另存一套与内核分离的模型数据。材料理论和 Python 直接调用方式见 [materials.md](materials.md)。

启动前请确保项目依赖已经安装，并在当前 Python 环境中运行：

```bash
python -m torchfea.ui
```

界面分为两个页面。前处理页面的模型树依次定义零件、OCC 几何特征、材料、节点集、单元集、表面集、装配实例、参考点、边界条件、载荷和静力隐式求解器；模型树上方的过滤框可按名称或类型快速定位对象。右侧有四个互相独立的视图：`Part Geometry`、`Part Mesh`、`Assembly Geometry`、`Assembly Mesh`。几何视图只显示 BREP 的点、OCC 曲线和实体面，所有未选中的面使用统一 CAD 面色并通过 OCC 边界线区分（面上的临时三角剖分不会作为网格线显示）；网格视图只显示有限元外表面的三角面片；Part 网格要在该 Part 的“剖分此 Part”完成后才能进入，Assembly 网格则要求所有 Part 都已剖分。装配体几何和网格视图会显示参考点标记。几何尺寸、OCC 特征或实例平移/转角在修改后会自动刷新 BREP（350 ms 防抖），同时使旧网格视图失效。拖拽鼠标直接旋转视图，滚轮缩放，中键拖拽平移；“充满视图”会将模型居中并缩放到当前窗口，`前/后/左/右/顶/底/等轴`按钮切换标准视角，“线框”和“边线”用于检查拓扑。视图中还显示带刻度的 XYZ 坐标轴。通过“新增零件”创建的零件会自动带一个 instance。前处理右下角“进入后处理”只提交当前已应用的定义和已生成的网格；未完成剖分不能进入计算，后处理页直接使用已提交网格求解，完成后可重新计算，或用“返回前处理”继续编辑。

“统一网格尺寸”是项目级唯一的体网格密度。点击“生成 FE 网格”时，所有 Part 以同一尺寸调用 Gmsh 三维剖分；此后才会显示有限元网格。未生成完整实例网格时不能进入后处理。后处理页的“1 · 开始计算”直接使用前处理生成的原生网格控制器，不会再次调用 Gmsh。修改几何后，预览自动返回 BREP。工具栏的撤回/重做记录所有已应用的 CAD 特征和模型定义操作。数值框会立即更新 BREP；一个数值编辑结束时自动形成一个撤回步骤，不必为保留历史专门点击“应用”。每个 Part 的“几何 / 网格”节点下直接列出 CAD 特征；选中某一项可直接编辑该特征，应用后仍保持当前树位置和展开状态。

当前的真实几何入口是内核中的 `torchfea.cad` Gmsh OCC 特征历史。CAD 历史属于 `FEAController.cad`，按 Part 名称保存离散构造过程；离散后的 `Part` 只包含节点、单元和集合。由 INP 创建的原生 `.npz` 可能没有 `FEAController.cad`，这类 Part 会在 UI 中标记为“离散几何 / 网格”，直接读取已有节点、单元、材料和集合，不虚构 CAD 历史，也不调用 Gmsh 重剖分；它的网格可直接进入 Part/Assembly Mesh 视图，集合需直接编辑文本而不能进行 BREP 拾取。每个 CAD 特征都是明确的 Python 类，例如 `Box`、`Cut` 和 `Rotate`，参数是对象的显式属性，不使用 `kind + values` 字典。每个操作通过 `apply_occ()` 实现自己的 Gmsh OCC 行为，统一的 `OCCBuilder` 只负责历史依赖校验、已消费特征和 OCC 实体生命周期。UI 直接编辑并提交这些内核对象，不再拥有另一份 CAD 实现。点击 CAD Part 下的“几何 / 网格”节点才显示“添加基本体 / 布尔运算 / 变换 / 剖分此 Part”；点击具体的 `Box-1` 等特征时，右侧只保留该特征的属性编辑，避免混淆操作上下文：

也可以绕过 UI 直接构造同一份模型：

```python
from torchfea import FEAController
from torchfea.cad import Box, Cylinder, Cut

fe = FEAController()
cad = fe.cad.add_part("Part-1")
cad.add(Box("Box-1", dx=20, dy=10, dz=5))
cad.add(Cylinder("Cylinder-1", x=10, y=5, radius=2, dz=5))
cad.add(Cut("Cut-1", object_id="Box-1", tool_id="Cylinder-1"))
part = fe.cad.mesh_part("Part-1", mesh_size=1.5)
# fe.cad.parts["Part-1"].operations 会随 FEAController.save_model 保存。
# fe.cad.parts["Part-1"].import_step("bracket.step") 会把 STEP 内容嵌入历史。
```

也可以使用签名同样明确的便捷方法 `cad.add_box(...)`、`cad.add_cylinder(...)` 和 `cad.cut(...)`；它们返回的仍分别是 `Box`、`Cylinder` 和 `Cut` 对象。

`torchfea.load_model` 会恢复 `FEAController.cad`。

CAD 集合也属于这份历史。例如 `history.add_surface_set("top", 7)`、
`history.add_node_set("corner", 12)` 和 `history.add_element_set("body", 3)`
分别记录 OCC 面、顶点和体的选择；这些整数是 OCC 拓扑标签，不是网格节点号或单元号。调用 `mesh()` 或构造求解器时，内核才把它们投影到当前网格。

- 基本体：`Box`、`Wedge`、`Cylinder`、`Sphere`、`Cone`、`Torus`。
- 外部几何：STEP 导入会作为 Part 的 `step_import` 特征保存，内容随原生模型序列化。
- 布尔特征：`Fuse`、`Cut`、`Intersect`、`Fragment`；两个或更多基本体可在同一 Part 的特征历史中组合。
- Part 内变换：`Translate`、`Rotate`、`Dilate`（非均匀缩放）、`Mirror`。它们直接改变 OCC 特征历史，和 assembly instance 的显示/装配变换是两层不同概念。
- 跨 Part 并集：工具栏“Part 并集合并”固定调用 OCC `fuse`，把两个未施加 instance 平移/转角的 Part 合并为一个带并集特征历史的新 Part；差集、交集和分割在 `FEAController.cad` 的 Part 历史中完成。

CAD 集合属于 Part 的 CAD 历史，而不是网格历史。布尔或变换后，历史中的拓扑选择会在新的 OCC 结果上重新解析；只有生成有限元对象时，才把这些选择映射为节点、单元和面集合。

CAD Part 的节点集、单元集和表面集保存为 `cad:vertex:<tag>`、`cad:volume:<tag>`、`cad:surface:<tag>` 等 OCC 拓扑选择，不保存网格节点号或单元面号。离散 Part 才使用零基编号，例如 `0, 3:8, 12`，以及 `单元号:面号` 表面语法。每个 Part 都有一个自动汇总全部外部面的外表面，默认名为 `extern`；它位于 Part 上，只能重命名，不能删除。剖分后系统生成 `all`（全部节点）、`all`（全部单元）和 `surface_all`（全部外表面节点）；每个自定义 surface 另外生成 `surface_<表面名>_nodes`。这些 FE 集合是 CAD 选择的网格投影，不是 CAD 定义本身。

创建一个集合后，选中该集合并使用“从预览选节点 / 体 / 面”：界面会自动切换到当前 Part 的 BREP 预览并进入对应选择模式。CAD 顶点以白色小球明确标识，仅在“选节点”模式下可拾取；选中后变为橙黄色。最终 FE 网格生成时，CAD 选择才解析为节点、单元和面。当前选中的 BREP 曲面显示为红色，并在集合写入后保持高亮；模型树中的 Part 外表面会将该 Part 的全部 BREP 子面同时标红。

材料位于 Part 子节点下。选中“材料”根节点可单独设置该 Part 材料集合的密度；密度会写入所有元素块并用于质量矩阵。UI 当前支持线弹性、Neo-Hookean 和 Neo-Hookean lnJ；内核还支持 Mooney–Rivlin、Yeoh、Gent、Arruda–Boyce 和 Ogden，但这些模型需要在 Python 中构造后通过 `element.set_materials(...)` 分配。每个 UI 材料通过一个单元集分配，多个材料不能重叠且必须覆盖该 Part 的全部单元。新增对象统一从模型树右键菜单创建；在“零件”节点右键可导入 `.stp`/`.step`，导入文件内容会嵌入 CAD 历史并自动创建 Part 和 Instance，不依赖原始路径。选中 Part 后可在属性页或右键菜单导出其 OCC BREP 为 `.stp`/`.step`，该操作不生成有限元网格。顶栏提供新建、打开、保存、INP 导出和撤回。边界、耦合、材料和载荷中的 Part/Instance/集合/表面/参考点引用均使用下拉框，只能选择当前已经定义的对象；修改实例后，节点集和表面目标列表会自动刷新。载荷类型包括压力、体力、集中力、力矩、参考点弹簧、自由度罚函数、自接触和接触；目标和参数会按类型显示。边界条件支持实例节点集上的 U1/U2/U3 约束以及参考点的 U1/U2/U3、R1/R2/R3 约束；约束节点提供参考点耦合。单元集已经可用于材料分配、选择和 `.inp` 导出。

文件菜单直接读写原生模型文件：

- `*.npz`：调用 `FEAController.save_model` 写出的原生 torchfea 模型。若模型由 CAD 创建，包含 CAD 历史；若模型由 INP 创建，则包含已有离散 Part。两种模型都包含 Assembly、Instance、集合、材料、载荷、约束和边界，并写入模型哈希码；打开时由这些内核对象重建编辑器状态，不另存 UI 文件。
- `*.inp`：调用 `Part.export_inp` 写出的 Abaqus Part deck，包含节点、单元、Nset、Elset 与 Surface。多零件会各写一个文件。

后处理页按 Abaqus 的结果查看逻辑组织。顶部可以选择“全部实例”或单个 Instance，并选择位移、变形梯度 `F`、体积比 `J`、Green-Lagrange 应变、Cauchy 应力（含 Mises）和应变能密度；右侧显示当前变量的最小值、最大值和平均值，并列出每个 Instance 的节点/单元数、最大位移和最大 Mises 应力，视图中显示对应色标。应力、应变和 `F` 由每个单元的高斯点值计算，再按单元连接平均到节点用于云图显示，不写回求解器结果。`显示变形` 和 `变形倍率` 控制变形后的网格位置，关闭后只显示未变形网格上的结果云图。前处理和后处理视图都提供充满视图、标准视角和 X/Y/Z 剖视开关与剖切面位置滑块，用于查看实体内部；剖视只改变显示，不修改 CAD 或有限元数据。

后处理页的“导出结果”支持 `.npz` 和 `.csv`：导出时重新计算当前模型哈希并校验结果；模型结构或载荷目标改变时要求重新计算，只有载荷数值改变时仍可导出和查看原结果。`.npz` 由内核 `StaticResult.save` 保存原始 `GC`、工作条件、求解时间、收敛信息、严格模型哈希和忽略载荷 `_parameters` 数值的兼容哈希；重新导入时由内核 `StaticResult.load` 读取，云图字段按当前模型重新计算。`.csv` 是 UI 的展示导出，按 Instance/节点展开，包含模型哈希、原始与变形坐标、位移、`F`、应力、应变、`J` 和应变能密度，便于外部后处理。

模型 `.npz` 和结果 `.npz` 均带有 8 位模型校验码（CRC32）。校验码在求解、模型保存和结果校验等边界按需重新计算，直接通过 Python 修改模型后无需清理缓存；计算前会规范化集合顺序、表面面片顺序、标量参数形状和自动派生集合，避免实际相同的模型因容器表示不同而得到不同校验码。结果文件同时保存严格模型哈希和结果兼容哈希；兼容哈希只忽略载荷对象的 `_parameters` 数值，仍校验载荷类型、目标、模型结构及自由度长度。旧版结果文件没有兼容哈希，只能在严格哈希相同时导入。校验码用于发现常见错配，不是密码学级唯一标识。

## 诊断策略

UI 到 Gmsh/torchfea 的边界不再用旧结果或默认对象掩盖失败：缺少 OCC 特征、重复/失效历史引用、空 BREP 面、缺少四面体、集合越界、实例/载荷/边界引用不存在时会立即报错并清空对应视图。Part 外表面始终自动重建为全部边界面，重命名不会改变其覆盖范围。

目前保留的兜底只有两类：Gmsh 曲线闭环失败时通过 STEP 导入重建拓扑（重建前后会比较 OCC 体积，改变布尔结果就直接失败），以及 PyVista 不可用时使用 VTK 原生渲染。网格或 BREP 失败不会继续显示上一次缓存。
