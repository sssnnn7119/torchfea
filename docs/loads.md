# Loads 载荷文档

载荷、边界和约束均属于 `Assembly` 工况的一部分。常用注册形式为 `assembly.add_load(load, name="load-name")`、`assembly.add_boundary(boundary, name="boundary-name")` 和 `assembly.add_constraint(constraint, name="constraint-name")`；名称用于后续工况切换和灵敏度分析。所有长度、力、时间和密度参数必须遵守同一单位制。

## 1. BaseLoad 基类接口

所有载荷类都继承自 `BaseLoad`（间接继承自 `BaseObj`）。载荷对象先构造，再通过 `assembly.add_load(load, name=...)` 注册；不要使用旧的 `assembly.add_load(name, load)` 参数顺序。一个自定义载荷需要实现以下方法：

### 1.1 必须实现的方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_stiffness` | `(RGC, if_onlyforce) -> tuple` | 返回力向量和刚度矩阵的贡献 |
| `get_potential_energy` | `(RGC) -> torch.Tensor` | 返回载荷的势能（Assembly 中会取负号加入总势能） |
| `set_required_DoFs` | `(RGC_remain_index) -> list` | 标记该载荷激活了哪些自由度 |

### 1.2 可选覆写的方法

| 方法 | 说明 |
|------|------|
| `initialize(assembly)` | 装配时调用，用于缓存索引、初始化参数等 |
| `get_F0()` | 静态方法，返回初始载荷向量（用于初始猜测） |

### 1.3 关键属性

- **`_parameters`**（`torch.Tensor`）：载荷的参数张量，所有可变参数应存入此张量。求解器的灵敏度分析通过对此张量求导来实现。
- **`_assembly`**（`Assembly`）：所属装配体的引用，`initialize()` 后可用。

### 1.4 `get_stiffness` 返回值

```python
# 当 if_onlyforce=True（只需力向量，快路径）:
return F_indices, F_values

# 当 if_onlyforce=False（需要力+刚度）:
return F_indices, F_values, K_indices, K_values
```

其中：
- `F_indices` / `F_values`：力向量的 COO 稀疏表示
- `K_indices` / `K_values`：刚度矩阵的 COO 稀疏表示（$2 \times N$ 索引 + $N$ 个值）

### 1.5 势能的符号约定

`get_potential_energy` 返回的势能以 $W(\mathbf{u})$ 记（外力功），在 `Assembly._total_Potential_Energy` 中做减法：

$$\Pi(\mathbf{u}) = U(\mathbf{u}) - \sum_{\text{loads}} W_{\text{load}}(\mathbf{u})$$

因此，对于弹簧等"内部"类型的载荷，`get_potential_energy` 应返回**负的**弹性势能（即 $-\frac{1}{2}k\Delta L^2$），使得在总势能中表现为加项。

---

## 2. Concentrate_Force — 集中力

### 2.1 用法

```python
from torchfea.model.loads import Concentrate_Force

# 在参考点 rp_name 上施加集中力 [Fx, Fy, Fz]
force = Concentrate_Force(rp_name="RP-1", force=[100.0, 0.0, -500.0])
assembly.add_load(force, name="my_force")
```

### 2.2 代码实现

```python
class Concentrate_Force(BaseLoad):
    def __init__(self, rp_name: str, force: list[float]):
        self.rp_name = rp_name          # 参考点名称
        self._parameters = torch.tensor(force, dtype=torch.float64)

    def initialize(self, assembly):
        # 缓存力作用的 DOF 索引（参考点的前 3 个自由度：x, y, z 位移）
        rp_index = assembly.get_reference_point(self.rp_name)._RGC_index
        self._indices_force = torch.arange(
            assembly._RGC_list_indexStart[rp_index],
            assembly._RGC_list_indexStart[rp_index] + 3
        )

    def get_stiffness(self, RGC, if_onlyforce=False):
        F_indices = self._indices_force
        F_values = self.force
        if if_onlyforce:
            return F_indices, F_values
        # 集中力是"死"载荷，不依赖位移，刚度为 0
        return F_indices, F_values, torch.zeros([2,0]), torch.zeros([0])

    def get_potential_energy(self, RGC):
        return (self.force * RGC[self.rp_index][:3]).sum()
```

### 2.3 理论

集中力是最简单的外载荷。外力功为力与位移的点积：

$$W(\mathbf{u}) = \mathbf{F} \cdot \mathbf{u}_{\text{RP}} = F_x u_x + F_y u_y + F_z u_z$$

其中 $\mathbf{u}_{\text{RP}}$ 是参考点的位移向量。对应的势能贡献为 $-W$。

集中力是**保守力**且**不依赖位移**（dead load），因此其刚度贡献为零：

$$\frac{\partial \mathbf{F}}{\partial \mathbf{u}} = \mathbf{0}$$

---

## 3. Moment — 集中力矩

### 3.1 用法

```python
from torchfea.model.loads import Moment

# 在参考点 rp_name 上施加力矩 [Mx, My, Mz]
moment = Moment(rp_name="RP-1", moment=[0.0, 0.0, 1000.0])
assembly.add_load(moment, name="my_moment")
```

### 3.2 代码实现

```python
class Moment(BaseLoad):
    def __init__(self, rp_name: str, moment: list[float]):
        self.rp_name = rp_name
        self._parameters = torch.tensor(moment, dtype=torch.float64)

    def initialize(self, assembly):
        # 缓存力矩作用的 DOF 索引（参考点的后 3 个自由度：Rx, Ry, Rz 旋转）
        rp_index = assembly.get_reference_point(self.rp_name)._RGC_index
        self._indices_force = torch.arange(
            assembly._RGC_list_indexStart[rp_index] + 3,
            assembly._RGC_list_indexStart[rp_index] + 6
        )

    def get_stiffness(self, RGC, if_onlyforce=False):
        # 类似集中力，力矩也是"死"载荷
        return self._indices_force, self.moment, ...

    def get_potential_energy(self, RGC):
        return (self.moment * RGC[self.rp_index][3:]).sum()
```

### 3.3 理论

参考点有 6 个自由度：前 3 个为平动位移 $(u_x, u_y, u_z)$，后 3 个为旋转位移 $(\theta_x, \theta_y, \theta_z)$。力矩作用在旋转自由度上：

$$W(\mathbf{u}) = \mathbf{M} \cdot \boldsymbol{\theta} = M_x \theta_x + M_y \theta_y + M_z \theta_z$$

与集中力类似，集中力矩也是不依赖位移的保守载荷，刚度贡献为零。

---

## 4. Pressure — 压力载荷

### 4.1 用法

```python
from torchfea.model.loads import Pressure

# 在 instance.surface 上施加压强 p
pressure = Pressure(instance_name="Part-1", surface_set="Surf-1", pressure=1.0)
assembly.add_load(pressure, name="my_pressure")
```

### 4.2 代码实现

Pressure 是最复杂的非接触载荷，因为它涉及曲面上的积分和位移依赖的刚度。

```python
class Pressure(BaseLoad):
    def __init__(self, instance_name: str, surface_set: str, pressure: float):
        self._parameters = torch.tensor([pressure], dtype=torch.float64)

    def initialize(self, assembly):
        # 获取表面单元，预计算形状函数在高斯点的值，缓存索引
        self.surface_element = assembly.get_instance(self.instance_name) \
                                  .surfaces.get_elements(self.surface_set)
        # 预计算力向量和刚度矩阵的 COO 索引结构
        # ...

    def get_potential_energy(self, RGC):
        # V = 封闭曲面所围体积的 1/3
        # 遍历高斯点: r = [r, r_ξ, r_η]，计算 det(r) 并积分
        # Π_pressure = -p * V
        ...

    def get_stiffness(self, RGC, if_onlyforce=False):
        # 力向量: f_a = -p * ∂V/∂u_a
        # 刚度矩阵: K_ab = -p * ∂²V/∂u_a∂u_b
        ...
```

### 4.3 理论

#### 压力势能

根据散度定理，作用在封闭曲面上的均匀压力 $p$ 可以写成体积势能。当前 `Pressure` 实现按曲面围成的体积计算，因此目标表面应是方向一致的封闭表面集合：

$$W_p(\mathbf{u}) = -p \cdot V(\mathbf{u}),
\qquad \Pi=U-W_p=U+pV$$

其中 $V(\mathbf{u})$ 是变形后曲面所围区域的体积。对于参数曲面 $\mathbf{r}(\xi, \eta)$：

$$V = \frac{1}{3} \int_{\Omega} \det\left[\mathbf{r}, \frac{\partial\mathbf{r}}{\partial\xi}, \frac{\partial\mathbf{r}}{\partial\eta}\right] d\xi d\eta$$

#### 力向量（一阶导数）

`Pressure.get_stiffness` 返回的是载荷势能的一阶导数；Assembly 将其从结构残差中减去：

$$q_a=\frac{\partial W_p}{\partial \mathbf{u}_a}
   =-p\frac{\partial V}{\partial \mathbf{u}_a},
\qquad R_p=-q$$

其中 $\frac{\partial V}{\partial \mathbf{u}_a}$ 通过对行列式的导数计算，最终通过高斯积分得到各节点的等效节点力。

#### 刚度矩阵（二阶导数）

压力是**跟随力**（follower load）——力的方向随曲面法向变化，因此载荷势能的二阶导数一般非零：

$$K^W_{ab}=\frac{\partial q_a}{\partial \mathbf{u}_b}
   =-p\frac{\partial^2 V}{\partial \mathbf{u}_a\partial\mathbf{u}_b},
\qquad K^R=-K^W$$

这对于几何非线性问题至关重要：忽略压力刚度会导致收敛缓慢甚至发散。

---

## 5. BodyForce — 体力（体积力）

### 5.1 用法

```python
from torchfea.model.loads import BodyForce

# 对 instance 的单元块 element_name 施加体力密度 [fx, fy, fz]（单位：力/体积）
# 默认值为重力 [0, 0, -9.81e-6]（单位制为 N/mm³）
body_force = BodyForce(
    instance_name="Part-1",
    element_name="EAll",
    force_density=[0.0, 0.0, -9.81e-6]
)
assembly.add_load(body_force, name="gravity")
```

### 5.2 代码实现

```python
class BodyForce(BaseLoad):
    def __init__(self, instance_name, element_name,
                 force_density=[0.0, 0.0, -9.81e-6]):
        self._parameters = torch.tensor(force_density, dtype=torch.float64)

    def initialize(self, assembly):
        # 预计算高斯积分权重与形函数的乘积
        # pdU_values = Σ_g w_g · N_a(ξ_g) · f （节点等效力）
        self._pdU_values = torch.einsum(
            'i, ge, gea->eai',
            self.force_density,
            element.gaussian_weight,
            element.shape_function_d0_gaussian
        )

    def get_potential_energy(self, RGC):
        # U = ∫ f · r(ξ) dV
        # 在高斯点上插值位移，与体力密度点乘并积分
        displacement_gaussian = Σ_a N_a(ξ_g) · u_a
        return Σ_g w_g · (f · r_g) · det(J_g)

    def get_stiffness(self, RGC, if_onlyforce=False):
        # 体力在小变形假设下是"死"载荷（刚度为零）
        # 在几何非线性下可能有刚度贡献，当前实现为零
        return F_indices, F_values, zeros(2,0), zeros(0)
```

### 5.3 理论

体力（如重力）的势能是体力密度 $\mathbf{f}_b$ 在变形体上的积分：

$$W_b(\mathbf{u}) = \int_{\Omega} \mathbf{f}_b \cdot \mathbf{r}(\mathbf{u}) \ dV$$

其中 $\mathbf{r}(\mathbf{u}) = \mathbf{X} + \mathbf{u}$ 是变形后的位置。在高斯积分点上：

$$W_b = \sum_{g} w_g \cdot \mathbf{f}_b \cdot \mathbf{r}_g(\mathbf{u}) \cdot \det\mathbf{J}_g$$

**等效力**通过对形状函数加权得到：

$$\mathbf{F}_a = \sum_g w_g \cdot N_a(\xi_g) \cdot \mathbf{f}_b \cdot \det\mathbf{J}_g$$

当前实现将体力视为不依赖位移的载荷（刚度为零），这对于大多数情况（位移远小于几何尺寸）是合理的近似。

---

## 6. Spring — 弹簧

弹簧系列包含两种类型：
- `Spring_RP_RP`：连接两个参考点
- `Spring_RP_Point`：连接参考点到空间中固定点

### 6.1 Spring_RP_RP 用法

```python
from torchfea.model.loads import Spring_RP_RP

# 连接 RP-1 和 RP-2，刚度 k=100，原长为初始距离（或手动指定）
spring = Spring_RP_RP(
    rp_name1="RP-1", rp_name2="RP-2",
    k=100.0, rest_length=None  # None = 使用初始距离
)
assembly.add_load(spring, name="spring_12")
```

### 6.2 Spring_RP_Point 用法

```python
from torchfea.model.loads import Spring_RP_Point

spring = Spring_RP_Point(
    rp_name="RP-1",
    point=[100.0, 0.0, 0.0],  # 空间固定点
    k=50.0, rest_length=30.0
)
assembly.add_load(spring, name="spring_ground")
```

### 6.3 代码实现

```python
class Spring_RP_RP(BaseLoad):
    def __init__(self, rp_name1, rp_name2, k, rest_length=None):
        rl = rest_length if rest_length is not None else -1.0
        self._parameters = torch.tensor([k, rl], dtype=torch.float64)

    def initialize(self, assembly):
        # 缓存两个参考点的位移 DOF 索引
        # 如果未指定原长，用初始几何距离
        if self.rest_length < 0:
            self.rest_length = ||p2 - p1||

    def get_stiffness(self, RGC, if_onlyforce=False):
        x1 = p1_init + RGC[rp1][:3]
        x2 = p2_init + RGC[rp2][:3]
        d = x2 - x1                    # 方向向量
        l = ||d||                       # 当前长度
        # 力: f = k * (l - L0) * d/l
        f = k * (l - L0) * (d / l)
        # 刚度: K = k * [n⊗n + (l-L0)/l * (I - n⊗n)]  (3×3 块)
        # 组装成 6×6: [K, -K; -K, K]

    def get_potential_energy(self, RGC):
        l = ||x2 - x1||
        return -0.5 * k * (l - L0)^2   # 负号：作为内力类型
```

### 6.4 理论

#### 势能

两个参考点间的非线性弹簧（几何精确轴向弹簧）的弹性势能为：

$$U_{\text{spring}} = \frac{1}{2} k \ (l - l_0)^2$$

其中 $l = \|\mathbf{x}_2 - \mathbf{x}_1\|$ 是当前长度，$l_0$ 是原长。如前所述，`get_potential_energy` 返回 $-U_{\text{spring}}$。

#### 节点力

对势能求导得节点力：

$$\mathbf{f}_1 = -\frac{\partial U}{\partial \mathbf{x}_1} = k(l - l_0) \cdot \frac{\mathbf{d}}{l}, \quad \mathbf{f}_2 = -\mathbf{f}_1$$

其中 $\mathbf{d} = \mathbf{x}_2 - \mathbf{x}_1$。

#### 切线刚度

$$\mathbf{K}_{11} = \frac{\partial \mathbf{f}_1}{\partial \mathbf{x}_1} = -k\left[\mathbf{n}\mathbf{n}^T + \frac{l-l_0}{l}(\mathbf{I} - \mathbf{n}\mathbf{n}^T)\right]$$

其中 $\mathbf{n} = \mathbf{d}/l$。整体 $6\times 6$ 刚度矩阵为：

$$\mathbf{K} = \begin{bmatrix} \mathbf{K}_{11} & -\mathbf{K}_{11} \\ -\mathbf{K}_{11} & \mathbf{K}_{11} \end{bmatrix}$$

---

## 7. Penalty_DoF — 自由度惩罚

### 7.1 用法

```python
from torchfea.model.loads import Penalty_DoF

# 惩罚 Part-1 的第 5 个自由度跟踪目标值 0.0，惩罚系数 k=1e6
penalty = Penalty_DoF(
    obj_name="Part-1",
    s=5,             # 该对象 RGC 段内的局部扁平移位索引
    target=0.0,      # 目标值
    k=1e6,           # 惩罚刚度
    obj_type="instance"  # 或 "auto" 自动检测
)
assembly.add_load(penalty, name="fix_dof5")
```

### 7.2 代码实现

```python
class Penalty_DoF(BaseLoad):
    def __init__(self, obj_name, s, target, k, obj_type="auto"):
        self._parameters = torch.tensor([k, target], dtype=torch.float64)

    def initialize(self, assembly):
        # 自动或手动定位对象，将局部索引 s 映射为全局索引
        obj = self._resolve_obj(assembly)
        # global_s = _RGC_list_indexStart[rgc_index] + s
        self._global_s = global_s

    def get_stiffness(self, RGC, if_onlyforce=False):
        s_now = RGC[rgc_idx].reshape(-1)[local_s]
        f = k * (target - s_now)       # 力 = k·(目标 - 当前)
        if not if_onlyforce:
            K = -k                       # 刚度 = -k
        return ...

    def get_potential_energy(self, RGC):
        return -0.5 * k * (s_now - target)^2
```

### 7.3 理论

自由度惩罚用二次势能强制单个自由度 $u_s$ 逼近目标值 $u_{\text{target}}$：

$$U_{\text{penalty}} = \frac{1}{2}k \ (u_s - u_{\text{target}})^2$$

对应的节点力（指向目标值）：

$$f_s = -\frac{\partial U}{\partial u_s} = k \ (u_{\text{target}} - u_s)$$

对应的刚度（常数）：

$$K_{ss} = \frac{\partial f_s}{\partial u_s} = -k$$

这是一种**软约束**方式——$k$ 越大，约束越"硬"，但过大的 $k$ 会导致刚度矩阵病态。选择 $k$ 时需要权衡约束精度和数值稳定性。

---

## 8. Contact — 接触

接触是有限元中最复杂的非线性载荷。本框架提供两种接触类型：
- `ContactSelf`：同一表面内的自接触
- `Contact`：两个不同表面之间的接触

### 8.1 用法

```python
from torchfea.model.loads import ContactSelf, Contact

# 自接触
self_contact = ContactSelf(
    instance_name="Part-1",
    surface_name="Surf-All",
    penalty_distance_f=1e-5,     # 穿透距离缩放
    penalty_factor_f=40.0,       # 惩罚因子
    penalty_start_g=-0.8,        # 法向对齐阈值
    penalty_end_g=-0.85,         # 法向完全分离阈值
    penalty_threshold_h=1.5,     # 检测距离阈值
    penalty_ratio_h=0.9,         # 距离衰减比率
    mesh_size=1.0                # 网格尺寸
)
assembly.add_load(self_contact, name="self_contact")

# 两表面接触
contact = Contact(
    instance_name1="Part-1", instance_name2="Part-2",
    surface_name1="Surf-Master", surface_name2="Surf-Slave",
    **same_penalty_params
)
assembly.add_load(contact, name="contact_12")
```

### 8.2 核心参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `penalty_distance_f` | 1e-5 | 距离尺度因子，控制穿透罚函数的"陡峭度" |
| `penalty_factor_f` | 40.0 | 穿透罚函数的指数系数 |
| `penalty_start_g` | -0.8 | 法向对齐罚函数的起始点（$\mathbf{n}_1\cdot\mathbf{n}_2$） |
| `penalty_end_g` | -0.85 | 法向对齐罚函数的结束点 |
| `penalty_threshold_h` | 1.5 | 距离罚函数的激活距离（×网格尺寸） |
| `penalty_ratio_h` | 0.9 | 距离罚函数的衰减范围比率 |
| `mesh_size` | 1.0 | 用于 KDTree 搜索和尺度归一化的特征网格尺寸 |

### 8.3 代码结构

接触的核心流程：

```
1. _filter_point_pairs():
   - 用 KDTree 搜索潜在接触对
   - 对自接触：排除法向背离的对（ratio_d 过滤）

2. get_potential_energy() / get_stiffness():
   - 计算变形后的表面位置 y 和法向 n
   - 构建接触量:
     * dy: 两点间距离向量
     * dn: 两点法向差值
     * D = (dn·dy)/2: 穿透指标
     * M = n1·n2: 法向对齐度
     * L = ||dy||: 两点距离
   - 组合三个罚函数: penalty = g(D) · f(M) · h(L) · weight
   - 链式法则求导得到力和刚度
```

### 8.4 理论

#### 罚函数法

接触采用**罚函数法**（Penalty Method），将不可穿透约束转化为势能惩罚项。总接触势能为高斯点对上的积分：

$$\Pi_{\text{contact}} = \sum_{(i,j) \in \text{pairs}} w_i w_j \cdot g(D_{ij}) \cdot f(M_{ij}) \cdot h(L_{ij})$$

其中三个光滑罚函数各司其职：

#### (1) 穿透罚函数 $g(D)$

衡量两点的法向穿透程度：

$$D_{ij} = \frac{1}{2}(\mathbf{n}_i - \mathbf{n}_j) \cdot (\mathbf{y}_i - \mathbf{y}_j)$$

$$g(D) = \exp(\beta_f \cdot D) \cdot \varepsilon_f$$

当 $D>0$（穿透），惩罚指数增长；当 $D<0$（分离），惩罚迅速衰减。

#### (2) 法向对齐罚函数 $f(M)$

防止背对背的表面产生虚假接触力：

$$M_{ij} = \mathbf{n}_i \cdot \mathbf{n}_j$$

$$f(M) = \text{smoothstep}\left(\frac{M - M_{\text{start}}}{M_{\text{end}} - M_{\text{start}}}\right)$$

其中 $\text{smoothstep}(t) = t^3(6t^2 - 15t + 10)$ 是 $C^2$ 光滑的 Hermite 插值函数。当两法向基本反向（$M \approx -1$，对向）时 $f \approx 1$，当法向正交或同向时 $f \to 0$。

#### (3) 距离截断罚函数 $h(L)$

限制接触力的作用范围，避免远距离点对产生不必要的计算：

$$L_{ij} = \|\mathbf{y}_i - \mathbf{y}_j\|$$

$$h(L) = \text{smoothstep}\left(\frac{L_{\text{threshold}} - L}{r \cdot L_{\text{threshold}}}\right)$$

当 $L > L_{\text{threshold}}$ 时 $h=0$，接触力完全消失。

#### 为什么要三个罚函数

| 罚函数 | 解决的问题 |
|--------|-----------|
| $g(D)$ | 核心的穿透惩罚——越穿透，力越大 |
| $f(M)$ | 过滤背对背的虚假接触（自接触中尤其重要） |
| $h(L)$ | 截断远距离计算，提高效率，避免"幽灵力" |

#### 光滑性

所有罚函数使用 $\text{smoothstep}$ 实现 $C^2$ 连续，确保在截断边界处力和刚度都连续，这对牛顿法的二次收敛至关重要。如果罚函数不光滑，牛顿法会在激活边界处振荡甚至发散。

---

## 9. 载荷在 Assembly 中的集成

在 `Assembly._total_Potential_Energy` 中：

```python
def _total_Potential_Energy(self, RGC):
    energy = 0
    # 结构应变能
    for ins in self._instances_enabled:
        energy += ins.potential_energy(RGC)
    # 所有载荷的势能（取负号）
    for f in self._loads_enabled:
        energy -= f.get_potential_energy(RGC)
    return energy
```

在 `Assembly.assemble_Stiffness_Matrix` 中，载荷的刚度和力贡献与结构贡献按 COO 格式拼接：

```python
for f in self._loads_enabled:
    F_ind, F_val, K_ind, K_val = f.get_stiffness(RGC)
    K_indices.append(K_ind)
    K_values.append(-K_val)  # 残差切线 = 结构切线 - 载荷切线
    R_indices.append(F_ind)
    R_values.append(-F_val)  # 残差力 = 结构力 - 载荷力
```

这种设计使得添加新载荷类型只需实现 `BaseLoad` 的三个核心方法，无需修改 Assembly 或 Solver。
