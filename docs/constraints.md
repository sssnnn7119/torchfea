# 约束算法说明

本文说明 `src/torchfea/model/constraints/` 中运动学约束的坐标组织、约束消元以及残差和切线矩阵映射方法。

当前约束模块包含：

- `BaseConstraint`：约束接口，不直接改变坐标。
- `Couple`：将参考点的刚体运动耦合到实例中的一组节点。
- `Joint`：在两个参考点之间建立具有指定相对平移/转动自由度的运动学关节。

位移边界条件位于 `model/boundarys/`，它们通过屏蔽 RGC 自由度实现，不属于本文的运动学消元约束。

## 1. RGC、GC 与约束消元

装配使用两层坐标：

- **RGC（redundant generalized coordinates）**：每个实例、参考点、载荷和约束拥有自己的坐标块。
- **GC（generalized coordinates）**：从 RGC 中提取未被屏蔽的自由度，作为求解器的实际未知量。

约束不直接改变各对象的 RGC 块大小，而是通过两类操作实现：

1. `modify_RGC(RGC)`：根据独立坐标重构被消元的坐标块。
2. `modify_R_K(...)`：将被消元坐标上的残差和刚度贡献映射回独立坐标。
3. `modify_mass_matrix(...)`：将被消元坐标上的质量贡献映射回独立坐标。

一次装配的主要顺序为：

```text
GC → RGC
   ↓
正向应用约束：重构从属坐标
   ↓
计算单元和载荷的 R、K
   ↓
反向应用约束：把从属坐标贡献映射回独立坐标
   ↓
屏蔽边界条件，得到约化 R、K
```

多个运动学约束之间通过 `get_input_blocks()` 和 `get_output_blocks()` 建立依赖图。坐标重构采用正向拓扑顺序；残差、切线和质量矩阵消元采用反向顺序。

## 2. Couple：参考点到节点集的刚体耦合

### 2.1 几何定义

设参考点初始位置为 $x_{rp}^0$，被耦合节点的初始位置为 $x_i^0$，定义节点相对于参考点的初始位置：

$$
r_i = x_i^0 - x_{rp}^0.
$$

参考点的 RGC 为：

$$
q_{rp} = [v,\ \theta],
$$

其中 $v$ 是平移向量，$\theta$ 是旋转向量。节点位移由刚体运动给出：

$$
u_i = v + R(\theta)r_i - r_i.
$$

因此节点当前位置为：

$$
x_i = x_i^0 + u_i = x_{rp}^0 + v + R(\theta)r_i.
$$

代码中的 `rotate_vectors()` 使用 Rodrigues 公式计算 $R(\theta)r_i$。

### 2.2 自由度处理

`set_required_DoFs()` 的处理为：

- 保留参考点的 6 个自由度；
- 删除耦合节点的 3 个平移自由度；
- 耦合节点位移由 `modify_RGC()` 重构。

因此，Couple 不增加新的独立未知量，而是把节点集的运动转换为参考点的刚体运动。

### 2.3 残差与切线映射

设节点集上的结构残差为 $f_i$，则平移残差映射为：

$$
R_v = \sum_i f_i.
$$

令

$$
Y_i(\theta)=R(\theta)r_i,
$$

则旋转残差为：

$$
R_\theta = \sum_i
\left(\frac{\partial Y_i}{\partial\theta}\right)^T f_i.
$$

代码中的 `_calculate_Ydotz()` 计算 $\partial Y_i/\partial\theta$ 及其二阶导数。切线矩阵包含两部分：

1. 节点原有切线通过运动学雅可比映射到参考点；
2. 残差与运动学二阶导数形成几何刚度项。

### 2.4 使用方式

```python
from torchfea.model.constraints import Couple
from torchfea.model.reference_points import ReferencePoint

rp_name = assembly.add_reference_point(
    ReferencePoint([0.0, 0.0, 10.0]),
    name="head-rp",
)

assembly.add_constraint(
    Couple(
        instance_name="part-1",
        set_nodes_name="head-nodes",
        rp_name=rp_name,
    ),
    name="head-couple",
)
```

该约束要求节点集和参考点都已经注册到同一个 `Assembly` 中。

### 2.5 刚体建模原则

一个连续刚体应只建立一次 `Couple`，并让节点集覆盖该刚体的全部节点，例如 CAD 网格自动生成的 `all` 节点集：

```python
Couple(
    instance_name="link-0",
    set_nodes_name="all",
    rp_name="link-0-rp",
)
```

这个 `rp_name` 就是该刚体的运动描述。若刚体的一端和另一刚体连接，应通过两个刚体的参考点建立 `Joint(connection_point=...)`；不要为了连接另一端再对同一个实例增加第二个 `Couple`。多个 Couple 会让同一实例的节点被不同参考点重复驱动，运动学依赖也会变得不明确。

## 3. Joint：两个参考点之间的运动学关节

### 3.1 独立坐标和方向基底

Joint 包含三个坐标块：

- master 参考点的 6 个坐标：平移 3 个、旋转 3 个；
- Joint 自己的相对自由坐标；
- slave 参考点的 6 个坐标。

其中 slave 坐标不是独立未知量，而是由前两个坐标块重构。Joint 的活动坐标排列为：

$$
q_a = [q_m, q_j]
  = [u_m,\ \phi_m,\ a_t,\ a_r].
$$

`translation_free_directions` 和 `rotation_free_directions` 分别定义允许的相对平移和相对转动方向。它们在 `initialize()` 中进行：

1. 形状、有限性和单位长度检查；
2. 线性独立性检查；
3. 正交归一化，形成方向基底 $B_t$、$B_r$。

如果方向数量分别为 $n_t$、$n_r$，Joint 申请 $n_t+n_r$ 个额外 RGC。方向列表为空表示该类相对运动完全被约束。

例如：

- 平移和转动方向都为空：刚性连接；
- 只有 3 个转动自由方向：球铰；
- 只有一个转动方向：铰链；
- 一个平移方向加一个转动方向：圆柱类关节的简化表示。

注意这里的 `free_directions` 表示**允许运动的方向**，不是被约束的方向。

### 3.2 相对运动

设允许的相对平移方向共有 $n_t$ 个，允许的相对转动方向共有 $n_r$ 个。经过初始化阶段的单位化和正交化后：

- $B_t\in\mathbb{R}^{n_t\times 3}$ 是相对平移方向基底；每一行是一个 master 初始体坐标系中的单位方向；
- $B_r\in\mathbb{R}^{n_r\times 3}$ 是相对转动方向基底；每一行是一个 master 初始体坐标系中的单位转动方向；
- $a_t\in\mathbb{R}^{n_t}$ 是 Joint RGC block 中对应相对平移的自由坐标；
- $a_r\in\mathbb{R}^{n_r}$ 是 Joint RGC block 中对应相对转动的自由坐标；
- $t_{rel}\in\mathbb{R}^{3}$ 是三维相对平移向量；
- $\rho_{rel}\in\mathbb{R}^{3}$ 是三维相对转动向量。

因此，Joint 自己的自由坐标通过方向基底转换为 master 初始体坐标系中的相对运动：

$$
t_{rel} = B_t^T a_t,
\qquad
\rho_{rel} = B_r^T a_r.
$$

例如，若只有一个允许的平移方向 $b_t\in\mathbb{R}^3$，则 $B_t=b_t^T$、$a_t=[\alpha]$，从而 $t_{rel}=\alpha b_t$。也就是说，$a_t$ 和 $a_r$ 不是三维向量，而是每个允许方向上的标量坐标；转置乘法将这些标量重新组合成三维运动向量。

由旋转向量对应的指数映射得到：

$$
R_m = \operatorname{Exp}([\phi_m]_\times),
\qquad
R_{rel} = \operatorname{Exp}([\rho_{rel}]_\times).
$$

slave 的姿态为：

$$
R_s = R_m R_{rel},
\qquad
\phi_s = \operatorname{Log}(R_s).
$$

其中：

- `skew_symmetric()` 构造叉乘矩阵 $[\cdot]_\times$；
- `rotation_matrix_from_vector()` 使用矩阵指数计算旋转矩阵；
- `rotation_vector_from_matrix()` 将旋转矩阵转换回旋转向量。
- `rotate_vectors()` 使用 Rodrigues 公式直接旋转向量，不构造旋转矩阵；

这些通用运算位于 `torchfea/utils/maths.py`，并保持 PyTorch 自动微分计算图。

### 3.3 连接点几何关系

设连接点的初始全局位置为 $x_c$，master 和 slave 初始参考点位置分别为 $x_m^0$、$x_s^0$。代码保存：

$$
d_m = x_c-x_m^0,
\qquad
d_s = x_c-x_s^0,
\qquad
d_0=x_m^0-x_s^0.
$$

master 和 slave 连接点的当前相对位置由下式保持：

$$
u_s = u_m+d_0+R_m d_m+R_m t_{rel}
      -R_mR_{rel}d_s.
$$

因此 slave 参考点的 6 个 RGC 为：

$$
q_s = [u_s,\ \phi_s].
$$

当 `connection_point=None` 时，连接点取 master 参考点初始位置，此时 $d_m=0$。

### 3.4 RGC 消元

`set_required_DoFs()` 保留：

- master 参考点的 6 个坐标；
- Joint 自己的相对自由坐标。

slave 参考点的 6 个坐标全部屏蔽，由 `modify_RGC()` 根据：

$$
q_s=h(q_a)
$$

重构。这样求解器只需要求解 $q_a$，不会把 slave 的坐标作为独立未知量重复求解。

### 3.5 残差、刚度和质量映射

令运动学雅可比为：

$$
J=\frac{\partial q_s}{\partial q_a}.
$$

若 slave 坐标上的残差为 $R_s$，则其对活动坐标的贡献为：

$$
R_a^{(s)}=J^T R_s.
$$

代码中的 `_kinematic_derivatives()` 通过 PyTorch 自动微分计算 $J$。

刚度映射包含普通链式法则项和几何项：

$$
K_a^{(s)} = J^T K_s J
            + \sum_{i=1}^{6} R_{s,i} H_i,
\qquad
H_i=\frac{\partial^2 h_i}{\partial q_a^2}.
$$

`_map_sparse_entries()` 处理稀疏矩阵中涉及 slave 行或列的条目，包括：

- slave-slave；
- slave-active；
- active-slave。

如果只装配残差，则不计算 Hessian；如果装配切线，则额外加入几何项。质量矩阵采用同一运动学变换：

$$
M_a^{(s)}=J^T M_s J.
$$

### 3.6 使用方式

```python
from torchfea.model.constraints import Joint

assembly.add_constraint(
    Joint(
        master_rp_name="master-rp",
        slave_rp_name="slave-rp",
        translation_free_directions=[[1.0, 0.0, 0.0]],
        rotation_free_directions=[[0.0, 0.0, 1.0]],
        connection_point=[0.0, 0.0, 0.0],
    ),
    name="hinge-joint",
)
```

构造函数只记录参数；方向归一化、校验、方向基底构造以及参考点索引解析在 `Assembly.initialize()` 调用 `Joint.initialize()` 时完成。由于 Assembly 需要提前分配 RGC block，方向条目数量会在构造阶段用于确定 block 大小；具体数值合法性仍在初始化阶段检查。

如果在 RGC 分配后修改方向列表数量，应重新初始化 `Assembly`。否则初始化阶段会检测到已分配 block 大小与新定义不一致并报错。

## 4. 约束接口与扩展

新增运动学约束通常需要实现：

```python
class MyConstraint(BaseConstraint):
    def get_input_blocks(self) -> set[int]:
        """Return blocks read during coordinate reconstruction."""

    def get_output_blocks(self) -> set[int]:
        """Return blocks reconstructed by this constraint."""

    def modify_RGC(self, RGC):
        """Reconstruct dependent RGC blocks."""

    def modify_R_K(self, RGC, R0, K_indices=None, K_values=None,
                   if_onlyforce=False, *args, **kwargs):
        """Map residual and tangent contributions to active coordinates."""
```

实现时应保持以下约定：

1. `modify_RGC()` 按正向依赖顺序执行后，所有从属坐标都可被后续单元和载荷读取；
2. `modify_R_K()` 按反向依赖顺序执行，将从属坐标贡献传回独立坐标；
3. `get_input_blocks()` 和 `get_output_blocks()` 必须准确描述坐标块依赖，否则可能得到错误的约束执行顺序；
4. 需要二阶一致切线时，必须包含运动学映射的 Hessian 或等价几何刚度项；
5. 所有索引映射应使用 Assembly 分配的全局 RGC 起始位置，不能假设对象注册顺序固定。
