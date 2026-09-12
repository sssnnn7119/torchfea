# 求解器与非线性迭代

求解器只接收已经建立好的 `FEAController`。通常先把材料、载荷、边界和约束加入 `controller.assembly`，再设置 `controller.solver`，最后调用 `controller.solve()`。收敛容差等求解器参数在构造函数中设置。

## 1. 核心思想：势能驻值原理

本框架的求解器基于**势能驻值原理**（在稳定问题上对应最小势能）构建。核心思路是：将可保守的内力和外力统一表达为系统总势能 $\Pi(\mathbf{u})$，然后寻找满足平衡条件的位移场 $\mathbf{u}^*$：

$$\nabla_{\mathbf{u}}\Pi(\mathbf{u}^*)=\mathbf{0}$$

静力求解的常用入口是：

```python
fe.solver = torchfea.solver.StaticImplicitSolver(
    maximum_iteration=10000,
    tol_error=1e-5,
)
result = fe.solve(GC0=None, if_initialize=True)
```

`FEAController.solve` 默认自动初始化装配。只改变材料或载荷参数、且不改变拓扑和约束结构时，可以传入上一个 `result.GC` 并设置 `if_initialize=False`。

### 1.1 总势能的构成

总势能由两部分组成：

$$\Pi(\mathbf{u}) = U(\mathbf{u}) - W(\mathbf{u})$$

- **$U(\mathbf{u})$**：结构应变能（内力势能）。由各单元在高斯积分点上积分得到，包含当前内核中的线弹性、超弹性和几何非线性。

  ```python
  # assembly.py: _total_Potential_Energy
  energy = 0
  for ins in self._instances_enabled:
      energy = energy + ins.potential_energy(RGC=RGC)  # 结构应变能
  ```

- **$W(\mathbf{u})$**：外力功。包括节点力、压力、接触力等所有外载荷做的功。

  ```python
  for f in self._loads_enabled:
      energy = energy - f.get_potential_energy(RGC=RGC)  # 外力势能（注意减号）
  ```

这种统一的好处是：结构内力和可保守载荷（包括压力、弹簧和接触）可以用同一套势能/残差接口处理。位移边界通过约化自由度施加，耦合约束则在残差和切线矩阵上做映射；它们不是简单地追加到 `Assembly._total_Potential_Energy` 中。求解器最终求解处理后自由度上的平衡驻点。

### 1.2 平衡条件

系统的平衡状态对应总势能的驻点：

$$\nabla \Pi(\mathbf{u}) = \mathbf{R}(\mathbf{u}) = \mathbf{0}$$

其中 $\mathbf{R}(\mathbf{u})$ 是残差力向量（residual force vector），它是总势能对位移的梯度。$\mathbf{R} = \mathbf{0}$ 即为有限元的平衡方程。

在实现中，`assemble_force()` 返回的正是 $\mathbf{R}$，而 `assemble_Stiffness_Matrix()` 同时在组装 $\mathbf{R}$ 和切线刚度矩阵 $\mathbf{K} = \nabla^2 \Pi(\mathbf{u})$。

---

## 2. 静力学隐式求解器（StaticImplicitSolver）

### 2.1 算法总览

静力学求解采用 **Newton-Raphson 方法 + 回溯线搜索（Backtracking Line Search）** 的全局化策略：

```
1. 初始位移 GC₀（默认取当前装配体的位移）
2. 循环迭代直到收敛：
   a. 组装残差力 R(GC) 和切线刚度矩阵 K(GC)
   b. 求解牛顿方向: K · dGC = -R
   c. 回溯线搜索: 寻找 α ∈ (0,1] 使 Π(GC + α·dGC) 充分下降
   d. 更新: GC ← GC + α·dGC
   e. 检查收敛: |dGC|_∞ < tol 且 |R|_∞ < tol
```

### 2.2 牛顿方向的计算

第 $k$ 步迭代的牛顿方程为：

$$\mathbf{K}(\mathbf{u}_k) \cdot \Delta \mathbf{u}_k = -\mathbf{R}(\mathbf{u}_k)$$

其中：

- $\mathbf{K} = \frac{\partial \mathbf{R}}{\partial \mathbf{u}}$ 是切线刚度矩阵（总势能的 Hessian）
- $\Delta \mathbf{u}_k$ 是牛顿方向（搜索方向）

刚度矩阵和残差力通过两次独立的组装获得：

```python
# 组装残差力 R（仅力向量）
R = self.assembly.assemble_force(GC=GC)

# 组装残差力 R 和切线刚度矩阵 K
R, K_indices, K_values = self.get_stiffness_matrix(GC_now=GC)
```

**注意**：这里调用两次组装是因为在 `assemble_force` 中只计算力向量，省去了刚度矩阵的计算，用于监控；而 `get_stiffness_matrix` 同时计算力和刚度矩阵，用于实际求解。两者在相同位移下结果一致。

### 2.3 回溯线搜索（Backtracking Line Search）

单纯的牛顿法可能不收敛，因此采用**回溯线搜索**保证每一步的势能充分下降。

#### Armijo 条件

搜索步长 $\alpha$ 需满足 **Armijo 条件**（充分下降条件）：

$$\Pi(\mathbf{u}_k + \alpha \Delta \mathbf{u}_k) \leq \Pi(\mathbf{u}_k) + c_1 \cdot \alpha \cdot \nabla\Pi(\mathbf{u}_k)^T \Delta \mathbf{u}_k$$

其中 $c_1 = 0.3$，$\nabla\Pi(\mathbf{u}_k)^T \Delta \mathbf{u}_k = \langle \mathbf{R}, \Delta\mathbf{u} \rangle$ 为方向导数。

#### 搜索过程

```
1. 初始 α = 1.0（尝试完整牛顿步）
2. 检查牛顿方向是否为下降方向:
   - 若 ⟨R, dGC⟩ > 0，取反方向 dGC = -dGC
3. 检查 dGC 合法性（是否含 NaN/Inf）
4. 回溯循环:
   - 试探 GC_new = GC + α·dGC，计算 Π(GC_new)
   - 若满足 Armijo 条件且步长不超过最大步长限制 → 接受
   - 否则 α ← 0.5·α，继续试探
   - α < 1e-12 → 放弃，α = 0
```

#### 步长约束

设置了最大步长 `_maximum_step_length = 1e10`，防止单步位移过大导致单元畸变。

### 2.4 收敛判据

收敛需同时满足两个条件（或残差足够小）：

$$|\Delta \mathbf{u}|_\infty < \text{tol} \quad \text{且} \quad |\mathbf{R}|_\infty < \text{tol}$$

实现中，当 $|\mathbf{R}|_\infty < 10^{-6}$ 时也会提前结束；否则需要同时满足步长和残差均小于构造求解器时设置的 `tol_error`（默认 `1e-5`）。

### 2.5 失效处理

- **最大迭代次数**：超过 `maximum_iteration`（默认 10000）→ 返回未收敛
- **低步长累积**：连续 10 次 $\alpha < 0.01$ → 认为无法收敛，返回当前结果
- **线搜索失败**：$\alpha = 0$ 且残差未达标 → 返回未收敛

---

## 3. 线性方程组求解：混合策略

线性方程组 $\mathbf{K}\Delta\mathbf{u} = -\mathbf{R}$ 的求解是每步迭代中最耗时的部分。本框架采用**共轭梯度法（CG）+ PyPardiso 直接求解器**的混合策略，在速度和鲁棒性之间取得平衡。

### 3.1 对角预处理（Diagonal Preconditioning）

首先对系统进行对角缩放预处理，改善条件数：

$$\tilde{K}_{ij} = \frac{K_{ij}}{\sqrt{|K_{ii}|} \cdot \sqrt{|K_{jj}|}}, \quad \tilde{R}_i = \frac{R_i}{\sqrt{|K_{ii}|}}$$

求解后还原：$\Delta u_i = \Delta\tilde{u}_i / \sqrt{|K_{ii}|}$

### 3.2 共轭梯度法（Conjugate Gradient）

- **初始解**：使用**上一步牛顿迭代的解** $\Delta\mathbf{u}_{k-1}$ 作为 CG 的初始猜测（经预处理后为 `x0 = dGC_prev * diag`）。由于牛顿迭代中相邻两步的方向通常接近，这能显著减少 CG 所需的迭代次数。
- **收敛容差**：`tol = 1e-5`，要求较低。因为 CG 只是为牛顿法提供一个搜索方向，不需要极高的精度——线搜索会修正步长。
- **最大迭代次数**：静力学为 1200（平时）/ 3000（困难时）；动力学为 1500（平时）/ 6000（困难时）。

### 3.3 PyPardiso 直接求解器

在以下情况切换为 PyPardiso（基于 Intel MKL Pardiso 的直接稀疏求解器）：

| 触发条件 | 说明 |
|----------|------|
| `__low_alpha_count > 5`（静力学）或 `> 3`（动力学） | 连续多步线搜索步长过小，说明牛顿方向质量差，需要精确求解 |
| `\|R_preconditioned\|_\infty < 1e-2` | 残差已经很小，接近最优解，精确求解可加速最终收敛 |
| `device == 'cpu'` | 在 CPU 上运行时直接使用 PyPardiso（CG 在 CPU 上较慢） |
| 动力学额外条件：`iter_now % 20 == 0` | 每 20 步做一次精确求解，定期"校准"方向 |

使用 PyPardiso 后会重置 `__low_alpha_count = 0`。

### 3.4 为什么这样设计

这一混合策略利用了两种方法的各自优势：

- **CG 的优势**：迭代法，不需要矩阵分解，内存占用小，对于大规模问题单次求解快。初始解好时几步就收敛。
- **PyPardiso 的优势**：直接法，结果精确，鲁棒性强。在接近最优解或 CG 困难时介入，避免 CG 收敛缓慢拖累整体性能。

核心思想是：**在牛顿迭代的早期阶段，方向不需要太精确，CG 快速给出一个大致方向即可；在接近最优解时，精确的牛顿方向能带来二次收敛速度**。动力学中还加入每 20 步定期精确求解的策略，防止 CG 方向逐渐偏离。

---

## 4. 动力学隐式求解器（DynamicImplicitSolver）

### 4.1 与静力学的区别

动力学求解器在静力学的基础上增加了惯性效应。动力学方程（运动方程）为：
$$\mathbf{M}\ddot{\mathbf{u}} + \mathbf{R}(\mathbf{u}) = \mathbf{F}_{\text{ext}}$$

### 4.2 Newmark-$\beta$ 时间积分

采用 **Newmark-$\beta$ 方法**（$\gamma=0.5$，$\beta=0.25$，即平均加速度法/梯形法则）进行时间离散。从时刻 $t_n$ 到 $t_{n+1}$：

**预测步**（根据上一时刻的状态预估当前位移）：

$$\tilde{\mathbf{u}}_{n+1} = \mathbf{u}_n + \Delta t \ \dot{\mathbf{u}}_n + \frac{\Delta t^2}{2}(1-2\beta)\ddot{\mathbf{u}}_n$$

**校正步**（隐式求解 $\mathbf{u}_{n+1}$，使得运动方程在 $t_{n+1}$ 满足）：

$$\mathbf{R}_{\text{int}}(\mathbf{u}_{n+1}) + \mathbf{M} \frac{\mathbf{u}_{n+1} - \tilde{\mathbf{u}}_{n+1}}{\beta\Delta t^2} = \mathbf{F}_{\text{ext}}$$

**更新速度和加速度**：

$$\dot{\mathbf{u}}_{n+1} = \frac{\gamma}{\beta\Delta t}(\mathbf{u}_{n+1} - \mathbf{u}_n) + \left(1-\frac{\gamma}{\beta}\right)\dot{\mathbf{u}}_n + \Delta t\left(1-\frac{\gamma}{2\beta}\right)\ddot{\mathbf{u}}_n$$

$$\ddot{\mathbf{u}}_{n+1} = \frac{1}{\beta\Delta t^2}(\mathbf{u}_{n+1} - \mathbf{u}_n) - \frac{1}{\beta\Delta t}\dot{\mathbf{u}}_n - \left(\frac{1}{2\beta}-1\right)\ddot{\mathbf{u}}_n$$

### 4.3 增量势能（Incremental Energy）

动力学同样采用最小势能原理，但势能的定义扩展为包含惯性项。在时间步 $[t_n, t_{n+1}]$ 内，定义**增量势能**：

$$\Pi_{\text{inc}}(\mathbf{u}_{n+1}) = U(\mathbf{u}_{n+1}) - W(\mathbf{u}_{n+1}) + \frac{1}{2\beta\Delta t^2}(\mathbf{u}_{n+1} - \tilde{\mathbf{u}}_{n+1})^T \mathbf{M} (\mathbf{u}_{n+1} - \tilde{\mathbf{u}}_{n+1})$$

其中 $\tilde{\mathbf{u}}_{n+1} = \mathbf{u}_n + \Delta t \dot{\mathbf{u}}_n + \frac{\Delta t^2}{2}(1-2\beta)\ddot{\mathbf{u}}_n$ 是上一节定义的**预测位移**（predictor），它仅依赖于上一时间步的已知状态 $(\mathbf{u}_n, \dot{\mathbf{u}}_n, \ddot{\mathbf{u}}_n)$，在当前步的 Newton-Raphson 迭代中保持固定。前两项为静力势能（应变能 − 外力功），第三项为惯性效应对应的"动能惩罚项"，它惩罚 $\mathbf{u}_{n+1}$ 偏离预测位移的程度——偏离越大，动能惩罚越大。

### 4.4 动力学求解流程

```
1. 初始化: 给定 u₀, v₀
   计算初始加速度: a₀ = M⁻¹ · (F_ext(u₀) - R_int(u₀))
2. 对每个时间步 n = 0, 1, 2, ...:
   a. 计算预测位移: ũ = uₙ + Δt·vₙ + Δt²·aₙ·(1-2β)/2
   b. Newton-Raphson 求解 u_{n+1}:
      - 以 uₙ 为初始值
      - 组装增量刚度矩阵和增量残差
      - 同静力学的牛顿法+线搜索+混合线性求解
   c. 更新速度和加速度 (Newmark 公式)
   d. 检查是否到达终止时间
```

### 4.5 动力学线性求解策略

与静力学类似，但有以下差异：

| 方面 | 静力学 | 动力学 |
|------|--------|--------|
| CG 最大迭代（平时） | 1200 | 1500 |
| CG 最大迭代（困难） | 3000 | 6000 |
| 触发 PyPardiso 的 `low_alpha` 阈值 | >5 | >3 |
| 定期 CG 校准 | 无 | 每 20 步优先使用更高迭代上限的 CG |
| 残差触发阈值 | `< 1e-2` | `< 1e-3` |

动力学加入每 20 步定期执行 CG 校准的策略；在 CPU 上仍直接使用 PyPardiso。增量刚度矩阵中加入了质量矩阵的贡献 $\frac{1}{\beta\Delta t^2}\mathbf{M}$，当 $\Delta t$ 很小时该项很大，系统条件数可能变差。

### 4.6 初始加速度

初始加速度通过求解 $\mathbf{M}\ddot{\mathbf{u}}_0 = \mathbf{F}_{\text{ext}} - \mathbf{R}_{\text{int}}(\mathbf{u}_0)$ 得到，使用 PyPardiso 直接求解。

---

## 5. 求解流程总结

### 静力学

```
                 ┌─────────────┐
                 │  初始 GC₀   │
                 └──────┬──────┘
                        │
              ┌─────────▼─────────┐
              │ 组装 R(GC), K(GC) │◄────────────────────┐
              └─────────┬─────────┘                     │
                        │                               │
              ┌─────────▼─────────┐                     │
              │  求解 K·dGC = -R  │                     │
              │  (CG 或 PyPardiso)│                     │
              └─────────┬─────────┘                     │
                        │                               │
              ┌─────────▼─────────┐                     │
              │   回溯线搜索 α    │                     │
              └─────────┬─────────┘                     │
                        │                               │
              ┌─────────▼─────────┐                     │
              │ GC ← GC + α·dGC  │─────────────────────┘
              └─────────┬─────────┘       未收敛
                        │
                   收敛？─── 否 ──→ 继续迭代
                        │是
                 ┌──────▼──────┐
                 │  返回结果    │
                 └─────────────┘
```

### 动力学

```
           ┌──────────────────────┐
           │ 初始化 u₀, v₀, a₀    │
           └──────────┬───────────┘
                      │
        ┌─────────────▼─────────────┐
        │ 对每个时间步 tₙ → tₙ₊₁:   │
        │                          │
        │ 计算预测位移 ũ            │
        │                          │
        │ Newton-Raphson (同上)     │◄──┐
        │ 最小化增量势能 Π_inc      │  │
        │                          │  │
        │ Newmark 更新 vₙ₊₁, aₙ₊₁  │──┘未收敛
        │                          │
        │ 记录结果                  │
        └─────────────┬────────────┘
                      │
                 t ≥ t_end？
                  否 ──→ 下一时间步
                   │是
            ┌──────▼──────┐
            │  返回结果    │
            └─────────────┘
```

---

## 6. 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `maximum_iteration` | 10000 | 每个加载步/时间步的最大牛顿迭代次数 |
| `tol_error` | 1e-5 | 位移增量和残差的收敛容差 |
| `_maximum_step_length` | 1e10 | 线搜索允许的最大位移增量 |
| 线搜索 $c_1$ (Armijo) | 0.3 | 充分下降条件系数 |
| 线搜索收缩因子 | 0.5 | 每次回溯步长减半 |
| CG 容差 | 1e-5 | 共轭梯度法收敛容差 |
| Newmark $\gamma$ | 0.5 | Newmark 积分参数 |
| Newmark $\beta$ | 0.25 | Newmark 积分参数（平均加速度法） |

---

## 7. 设计哲学

1. **能量统一视角**：当前可保守的弹性、超弹性、接触和载荷通过势能贡献进入总势能，求解器负责求解其驻点。这使得添加新的能量型物理效应时通常无需修改求解器核心。

2. **全局化牛顿法**：纯牛顿法在远离最优解时可能发散，回溯线搜索保证了每步迭代的势能单调下降，大幅提高鲁棒性。

3. **混合线性求解**：CG 快但不保证收敛，PyPardiso 精确但开销大。通过自适应切换，在大多数迭代中享受 CG 的速度，在关键时刻获得直接法的可靠性。

4. **PyTorch 原生**：所有计算基于 PyTorch 张量，天然支持 GPU 加速和自动微分。这使得灵敏度分析（伴随法）可以直接利用 PyTorch 的计算图。
