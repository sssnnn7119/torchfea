# 线性表面绑定

`LinearTie` 负责固定插值约束的自由度消元；`NodeToSurfaceTie` 在初始装配构形上生成点面插值关系。两者都继承现有的 `BaseConstraint`，通过 `Assembly.add_constraint()` 使用。

```python
from torchfea.model.constraints import NodeToSurfaceTie

tie = NodeToSurfaceTie(
    main_instance_name="Part-A-1",
    main_surface_name="interface",
    secondary_instance_name="Part-B-1",
    secondary_surface_name="interface",
    position_tolerance=1e-6,  # 绝对距离，与模型坐标使用相同单位
)
assembly.add_constraint(tie, name="interface-tie")
assembly.initialize()

print(tie.tied_nodes)
print(tie.unmatched_nodes)
print(tie.projection_distances)
```

主从面必须是已定义的有限元表面。搜索使用 `Instance.nodes`，因此包含实例的初始平移和转动。主面支持 T3、T6、Q4、Q8，并保留原有限元形函数；二次面的中间节点参与插值，不会用显示用的三角剖分替代原面。

## 初始投影与容差

对每个唯一的从面节点，寻找其到主面片的最近点，包括面片边界，并在投影点计算形函数权重。初始化完成后，投影位置、关联主节点和权重固定；后续迭代只传递位移，不重新搜索或滑移。

主面的候选面片通过 AABB BVH（包围盒层次树）筛选。曲面采用 Bernstein 控制点包围盒，避免二次面在节点包围盒之外的凸起被漏掉。BVH 按包围盒中心最长轴做中位划分，内部节点保存子面片的联合包围盒；单个大面片不会扩大所有查询的搜索半径。查询按点到包围盒的距离下界优先遍历，找到实际投影后，用当前最佳距离进一步剪枝。T3 使用解析最近点计算；T6、Q4、Q8 使用有解析梯度的有界优化，并对内部投影使用解析 Hessian 做 Newton 修正。

包围盒距离相同的候选按面片 ID 排序；实际投影距离相同时也选择较小的面片 ID，使共享边界上的对应关系不依赖树的构建或遍历顺序。包围盒大量重叠时 BVH 仍可能退化，不能保证所有几何上的对数查询开销。

`position_tolerance` 必须是有限非负数。默认 `allow_unmatched=False`：任何从面节点超出距离容差都会报错并列出节点 ID。显式设置 `allow_unmatched=True` 后，只消去成功匹配节点的自由度，其余节点保持独立；它们不会在后续变形中自动加入 Tie。浮点投影计算允许与坐标尺度相应的舍入误差。

本实现不会调整节点初始位置。对于非零初始间隙，线性位移插值保持的是固定的全局偏移，不能描述该偏移随刚体转动的运动；有限转动分析应使用初始重合的接口。它也不处理壳厚度、转动自由度或 Abaqus/Standard 的非线性偏移约束。

## 紧凑映射与解析导数

插值关系为

\[
\mathbf u_i^s=\sum_a w_{ia}\mathbf u_a^m.
\]

节点级映射保存为三个等长的一维向量：从节点行编号、主节点编号和权重。点面 Tie 每行最多有 3、4、6 或 8 个条目；混合面类型可以有不同的行长度。计算采用 gather 和 index-add，不构造稠密 C，也不存储位移的稠密 Jacobian 或 Hessian。相同接口可供后续面面算法生成不规则的关联权重。

三个平移分量共用标量权重。其一、二阶导数分别为

\[
\frac{\partial u^s_{i\alpha}}{\partial u^m_{a\beta}}
=w_{ia}\delta_{\alpha\beta},\qquad
\frac{\partial^2 u^s_{i\alpha}}
{\partial u^m_{a\beta}\partial u^m_{b\gamma}}=0.
\]

设 H 为保留其他坐标并将从节点位移替换成插值结果的映射，则

\[
R'=H^T R,\qquad K'=H^T K H,\qquad M'=H^T M H.
\]

`Assembly` 的约束接口累加返回值，所以 `LinearTie.modify_R_K()` 返回 `R'-R` 和 `K'-K`，质量接口返回 `M'-M`。只有接触从自由度的 COO 条目需要展开；不对整个矩阵做稠密变换。非对称切线、重复 COO 条目和链式约束使用相同处理。

面形函数在任意自然坐标处的求值通过 `BaseSurface.evaluate_shape_functions()` 返回 N、dN、d²N。投影距离函数

\[
f=\tfrac12\|x(\xi,\eta)-X_s\|^2
\]

的解析梯度和 Hessian 为

\[
f_{,i}=x_{,i}\cdot(x-X_s),\qquad
f_{,ij}=x_{,i}\cdot x_{,j}+x_{,ij}\cdot(x-X_s).
\]

这些导数均由形函数多项式直接计算，不使用自动微分；固定投影不对参考网格或投影搜索求敏感度。

## 约束组合与保存

约束统一通过 `get_input_dofs()` 和 `get_output_dofs()` 提供一维 NumPy int64 全局 RGC 输入和输出自由度数组，允许重复索引。建图使用排序后的紧凑输出索引和批量二分查找，不保存逐自由度 Python 字典，也不分配覆盖整个模型的稠密所有权数组；Python 循环只处理约束及其依赖边。装配按实际依赖执行正向位移重构、反向残差与矩阵传递。支持无环的 `A -> B -> C` Tie 链。主节点可以施加边界条件；同一个从自由度被多个约束控制时会报错。边界条件沿用原有处理流程，不参与依赖图和冲突检查；使用时应避免在 Tie 从自由度上再次施加边界条件。

模型保存只记录主从引用、容差、是否允许未匹配和 enabled 配置。投影、插值和索引缓存在加载后初始化时重新生成。

本轮提供 `LinearTie` 和 `NodeToSurfaceTie`；surface-to-surface 系数构造尚未实现。Abaqus 两种离散方式的说明见 [Mesh Tie Constraints](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-tiedconstraint.htm)。本实现不宣称复现 Abaqus 的全部 Tie 选项。

## 测试

```bash
conda run -n mine python -m pytest -q
```

`tests/model/constraints/test_linear_tie.py` 用中心差分验证面形函数、投影目标函数的一二阶导数，以及位移映射的一阶导数和零 Hessian；同时检查非线性能量的梯度/切线、真实有限元装配链、质量映射、实例变换、曲面投影、约束冲突和序列化重建。
