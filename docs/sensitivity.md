# 灵敏度分析（伴随法）

伴随灵敏度分析用于计算目标函数对设计变量的总导数，适合拓扑优化、几何优化和基于物理的学习。`torchfea` 的 `StaticImplicitSolver` 在静力平衡解的基础上，通过一次或少量额外线性求解构造梯度。

## 基本用法

`torchfea` 的 `StaticImplicitSolver` 提供 `get_sensitivity`，并负责伴随方程、残差自动微分和结果清理。

### 1. 导入库与模型求解

首先，设置模型并求解一次静力学结果。在运行灵敏度分析之前，你需要一个收敛的静力学解（`fe_result`）。以下示例代码演示了该过程：

```python
import torch
import torchfea

# 初始化模型解析参数
fem = torchfea.FEA_INP()
fem.read_inp('example.inp')
fe = torchfea.from_inp(fem)

# 分配求解器
fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-6)

# 添加载荷和边界条件
fe.assembly.add_load(...)
fe.assembly.add_boundary(...)

# 求解器会在第一次 solve() 时自动初始化
feresult = fe.solve()
```

### 2. 定义设计变量的应用函数 (`apply_func`)

灵敏度分析需要知道设计变量如何影响有限元系统。你需要定义一个回调函数 `apply_func`，该函数会在灵敏度计算中修改 `Assembly` 的属性。这些修改**必须能通过 PyTorch 追踪其计算图 (Autograd)**。
例如，以节点坐标作为设计变量：

```python
def apply_design_vars(assembly: torchfea.Assembly, design_vars: torch.Tensor) -> None:
    # 获取需要修改的部件
    part = assembly.get_part('final_model')
    # 使用传入的含有梯度信息的 design_vars 更新模型属性
    # 注意 reshape 需要匹配原来的形状，避免破坏计算图
    part.nodes = design_vars.reshape(part.nodes.shape)
```

### 3. 定义目标函数 (`compute_objective_func`)

接下来，定义一个回调函数用来计算优化目标。当前接口中，目标函数接收 `StaticResult` 与当前装配体 `assembly`，并返回一个标量 `Tensor`。
例如，目标函数是针对某个特定自由度（如末端参考点位移）的最小化：

```python
def compute_objective(fe_result: torchfea.solver.StaticResult,
                      assembly: torchfea.Assembly) -> torch.Tensor:
    # 假设需要最小化索引为 -2 的自由度的值
    obj = fe_result.GC[-2]
    return obj
```

### 4. 计算敏感度梯度

通过调用求解器的 `get_sensitivity` 函数，将之前得到的结果和回调函数输入，即可完成梯度的自动计算：

```python
# 获取用于求导的设计变量的初始值 (必须要求是一维/多维的 torch.Tensor)
part = fe.assembly.get_part('final_model')
initial_design_vars = part.nodes.clone().detach().reshape(-1)

# 调用伴随灵敏度推导
grad_sensi = fe.solver.get_sensitivity(
    fe_result=feresult,
    design_vars=initial_design_vars,
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
)

print('敏感度梯度:', grad_sensi)
```

## 计算机制解释

`get_sensitivity` 采用计算图追踪与伴随状态方程（Adjoint Equation）相结合的离散伴随法：

1. **设置状态量**：利用 `apply_func` 将设计变量附着到了模型中，然后重建有限元系统的计算图对象。
2. **伴随载荷构建**：对 `compute_objective_func` 进行一次前向反向计算来得到目标对位移的偏导数 $\frac{\partial Obj}{\partial \boldsymbol{U}}$。
3. **伴随变量求解（Adjoint Equation Solve）**：使用求解静力学平衡方程时产生的刚度矩阵（已分解完成，使得求解非常快），求解伴随系统方程 $\boldsymbol{K} \boldsymbol{\lambda} = - \frac{\partial Obj}{\partial \boldsymbol{U}}$。
4. **梯度重构（Total Sensitivity）**：通过求内积重构残差功方程并调用 PyTorch 自动微分（`work.backward()`），最终求出目标对象对全部设计变量的总导数 $\frac{\partial Obj}{\partial X}$。

使用 `torchfea` 中的这种设计方式，可以处理接触、材料非线性和大变形场景下的隐式梯度；基础伴随过程通常只需求解一次额外的线性方程组。相比摄动法逐个扰动设计变量，这在设计变量较多时更高效。

## 雅可比灵敏度分析 (Jacobian Sensitivity)

在进行高级分析和设计优化时（如针对载荷参数不确定性的鲁棒性优化、追踪特定自由度的荷载位移曲线斜率等），有时并不只是关注单纯的位移，还会高度关注系统的响应对特定载荷参数的变化率（即雅可比矩阵，Jacobian）。

如果目标函数不仅依赖于系统状态变量（自由度 $u_i$），还依赖于系统位移对载荷参数的雅可比矩阵（$\frac{\partial u_i}{\partial p_n}$），可以使用 `StaticImplicitSolver` 提供的 `get_jacobian_sensitivity` 方法。该方法通过多重伴随系统，同时处理刚度矩阵和残余力的自动微分解析，实现了高阶（包含雅可比项）敏感度的高效求解。

### 用法举例

与基础的伴随灵敏度分析类似，使用雅可比灵敏度分析同样需要准备好收敛的静力学解 `fe_result` 和设计变量。主要区别在于：

1. 你需要指定关注哪些载荷参数（`load_names`）。
2. 当前接口中，目标函数接收 `StaticResult`（单工况）或 `list[StaticResult]`（多工况）。

#### 1. 定义受雅可比影响的目标函数

单工况（`get_jacobian_sensitivity`）目标函数签名如下：

```python
def compute_objective_jacobian(fe_result: torchfea.solver.StaticResult,
                               assembly: torchfea.Assembly) -> torch.Tensor:
    # fe_result.jacobian 是 dict[str, torch.Tensor]
    # 例如关注 pressure-1 对某个自由度的雅可比
    return fe_result.jacobian['pressure-1'][-2, 0] ** 2
```

多工况（`get_jacobian_sensitivity_multistep`）目标函数签名如下：

```python
def compute_objective_jacobian(fe_results: list[torchfea.solver.StaticResult],
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # 示例: 两个工况目标加权求和
    term_0 = fe_results[0].jacobian['pressure-1'][-2, 0] * fe_results[0].GC[-2]
    term_1 = fe_results[1].jacobian['pressure-1'][-2, 0] * fe_results[1].GC[-2]
    return term_0 + term_1
```

#### 2. 调用求解器推导雅可比灵敏度

提供需要计算偏导的载荷名称列表（`load_names`），并传入目标函数，会自动推导目标函数对设计变量的偏导数：

```python
solver = fe.solver
```

单工况调用：

```python
focus_loads = ['pressure-1', 'pressure-2']

grad_sensi_jacobian = solver.get_jacobian_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes.reshape(-1),
    load_names=focus_loads,
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective_jacobian,
)

print('单工况雅可比灵敏度:', grad_sensi_jacobian)
```

多工况调用：

```python
# 指定需要求雅可比相关的载荷名称，例如面载荷 'pressure-1' 和 'pressure-2'
focus_loads = ['pressure-1', 'pressure-2']

# 调用雅可比灵敏度分析
grad_sensi_jacobian_batch = solver.get_jacobian_sensitivity_multistep(
    fe_results=[feresult1, feresult2],
    design_vars=part.nodes.reshape(-1),
    load_names=focus_loads,
    apply_func=apply_design_vars,
    compute_objective_funcs=compute_objective_jacobian,
    )

print('计算获得最终的灵敏度:', grad_sensi_jacobian_batch)
```

注意：当前 `get_jacobian_sensitivity` 与 `get_jacobian_sensitivity_multistep` 都返回一个 `torch.Tensor`（即最终设计灵敏度）。

### 雅可比矩阵的计算方法

在 `torchfea` 中，雅可比矩阵 $J_{in} = \frac{\partial u_i}{\partial p_n}$ 通过 `StaticImplicitSolver.get_jacobian()` 方法计算。其核心思路是：

1. **组装载荷参数**：将所有指定载荷 $p_n$ 的参数拼合为一个一维张量 `total_params`。
2. **定义残差闭包**：定义一个函数 `closure_R(total_params)`，将参数设置回载荷对象后调用 `assemble_force` 计算残余力 $R_i$。
3. **逐列计算 $\partial R_i / \partial p_n$**：对每个载荷参数 $p_n$，利用 `torch.autograd.functional.jvp`（Jacobian-Vector Product）计算残差对参数的偏导数列向量：

   $$
   \frac{\partial R_i}{\partial p_n} = \text{JVP}(\text{closure\_R}, \text{total\_params}, \boldsymbol{e}_n)
   $$

   其中 $\boldsymbol{e}_n$ 是第 $n$ 个基向量。这一步等价于对每个参数 $p_n$ 求一次向量-雅可比积，得到 $(\partial R/\partial p_n)$。

4. **求解线性系统**：利用静力分析中已分解（$LDL^T$）的切线刚度矩阵 $K_{ij}$，求解：

   $$
   K_{ij} \frac{\partial u_j}{\partial p_n} = -\frac{\partial R_i}{\partial p_n}
   $$

   得到雅可比矩阵的各列 $J_{jn} = \partial u_j / \partial p_n$。

5. **按载荷分组输出**：将计算得到的完整雅可比矩阵按载荷名称拆分为 `dict[str, torch.Tensor]` 返回。

这种方法**逐参数计算** `dR/dp`，每次仅需一次 JVP 和一次回代求解。由于 $K$ 矩阵的分解已在前序静力分析中完成并缓存，每次回代的成本极低。当载荷参数数量 $N_p$ 较少时，该方法非常高效。

---

## 理论与计算机制 (Theory & Implementation)

### 常用符号表

|                符号                | 物理及数学意义                                                     |
| :--------------------------------: | :----------------------------------------------------------------- |
|               $L$               | 优化目标函数 (Objective function)                                  |
|              $u_i$              | 节点的广义自由度或系统位移 (Generalized DoFs)                      |
|              $b_m$              | 用于优化的设计变量 (Design variables)                              |
|              $p_n$              | 外部特定的系统参量，如载荷大小 (External parameters)               |
|              $R_i$              | 系统残余力 (Residual force, Internal force - External force)       |
|             $K_{ik}$             | 对称系统的切线刚度矩阵 (Tangent stiffness matrix)                  |
|             $J_{in}$             | 系统位移对待定参数的雅可比矩阵响应 (Jacobian matrix)               |
|           $\lambda_i$           | 计算伴随方程引入的一阶伴随变量 (Primary adjoint variables)         |
| $\lambda_{in}^*, \lambda_i^{**}$ | 计算雅可比伴随方程引入的二次伴随变量 (Secondary adjoint variables) |

### 1. 位移场灵敏度推导

我们首先定义位移场相关目标函数

$$
L = L(u_i, b_m)
$$

在基础分析中，静力平衡方程由残余力为零表示：

$$
R_i(u_k, b_m) = 0
$$

利用链式法则求其对设计变量 $b_m$ 的全导数为：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m}
$$

上式中隐式的 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$（位移对设计变量的偏导）直接计算代价极高。我们对平衡方程 $R_i = 0$ 再次对 $b_m$ 求全导数以建立联系：

$$
K_{ik} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} + \frac{\partial R_i}{\partial b_m} = 0 \quad \Rightarrow \quad \frac{\mathrm{d} u_k}{\mathrm{d} b_m} = - K^{-1}_{kr} \frac{\partial R_r}{\partial b_m}
$$

将 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$ 代入总灵敏度方程中，即得：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} - \frac{\partial L}{\partial u_k} K^{-1}_{kr} \frac{\partial R_r}{\partial b_m}
$$

对于复杂的系统，我们不想分别求每一个设计变量的导数。此时，我们定义伴随变量 $\lambda_r$ 满足伴随方程：

$$
\lambda_r = - K^{-1}_{rk} \frac{\partial L}{\partial u_k}
$$

可将复杂项化简。最终的总灵敏度退化为计算如下内积：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \lambda_r \frac{\partial R_r}{\partial b_m}
$$

最后对上述表达式进行自动微分即可得到目标函数对设计变量的总导数。

### 2. 包含雅可比的高阶灵敏度推导

在更复杂的优化场景中（例如针对参数不确定性的鲁棒性优化，或多载荷工况下的灵敏度分析），目标函数可能不仅依赖于系统当前的位移 $u_i$，还会显式依赖于系统位移对外设参数 $p_n$（如特定载荷边界大小、材料参数等）的雅可比矩阵 $J_{in} = \frac{\partial u_i}{\partial p_n}$。此时，目标函数可以一般化地表示为：

$$
L = L(u_i, J_{in}, b_m)
$$

对于给定的任意系统参数 $p_n$，系统必须依然满足静力等效平衡方程：

$$
R_j(u_i, b_m, p_n) = 0
$$

利用隐函数定理直接对上式关于外部参数 $p_n$ 求解全导数，即可得到系统雅可比矩阵的控制方程：

$$
K_{ji} \frac{\partial u_i}{\partial p_n} + \frac{\partial R_j}{\partial p_n} = 0 \quad \Rightarrow \quad J_{in} = - K_{ij}^{-1} \frac{\partial R_j}{\partial p_n}
$$

在此基础上，由于目标函数引入了雅可比依赖，根据多元微积分的链式法则，目标函数对设计变量 $b_m$ 的全导数扩展为：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} + \frac{\partial L}{\partial J_{in}} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m}
$$

推导的难点集中于上式的第三项 $\frac{\mathrm{d} J_{in}}{\mathrm{d} b_m}$。为了求解这一项，我们将此前得到的雅可比平衡等式 $\left(K_{ji} J_{in} + \frac{\partial R_j}{\partial p_n} = 0\right)$ 对设计变量 $b_m$ 此度求取全导数。需注意的是，刚度矩阵 $K$ 和残余力 $R$ 均包含对位移 $u$ 和设计变量 $b$ 的隐式依赖，我们必须严格应用链式扩展：

$$
K_{ji} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m} + \left( \frac{\partial K_{ji}}{\partial b_m} + \frac{\partial K_{ji}}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right) J_{in} + \left( \frac{\partial^2 R_j}{\partial b_m \partial p_n} + \frac{\partial^2 R_j}{\partial u_k \partial p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right) = 0
$$

为揭示内部机制，我们将上式中含有 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$ 的项提取得出，并在等式两端移项整理：

$$
K_{ji} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m} = - \left( \frac{\partial K_{ji}}{\partial u_k} J_{in} + \frac{\partial^2 R_j}{\partial u_k \partial p_n} \right) \frac{\mathrm{d} u_k}{\mathrm{d} b_m} - \left( \frac{\partial K_{ji}}{\partial b_m} J_{in} + \frac{\partial^2 R_j}{\partial b_m \partial p_n} \right)
$$

观察上述等式右侧的系数，括号内的物理意义分别代表对应参数关于 $p_n$ 的全微分。结合材料线弹性或严格切线刚度特性 $\frac{\partial K_{ji}}{\partial u_k} = \frac{\partial K_{jk}}{\partial u_i}$，我们引入全微分算子 $\frac{\mathbf{d}}{\mathbf{d} p_n}$ 进行聚合：

$$
K_{ji} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m} = - \frac{\mathbf{d} K_{jk}}{\mathbf{d} p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} - \frac{\partial}{\partial b_m}\left( \frac{\mathbf{d} R_j}{\mathbf{d} p_n} \right)
$$

回忆第一节中的位移场灵敏度基础结论 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m} = - K_{kr}^{-1} \frac{\partial R_r}{\partial b_m}$。将其代入上式，并同时在方程两侧左乘逆刚度矩阵 $K^{-1}$ 进行指标置换转化，即得雅可比全导数的显式代数表达式：

$$
\frac{\mathrm{d} J_{jn}}{\mathrm{d} b_m} = K^{-1}_{jr} \left[\frac{\mathbf{d} K_{rs}}{\mathbf{d} p_n} K^{-1}_{si} \frac{\partial R_i}{\partial b_m} - \frac{\partial}{\partial b_m} \left( \frac{\mathbf{d}R_r}{\mathbf{d}p_n} \right)\right]
$$

带入总灵敏度表达式中的雅可比项:

$$
\frac{\partial L}{\partial J_{jn}} \frac{\mathrm{d} J_{jn}}{\mathrm{d} b_m} =\frac{\partial L}{\partial J_{jn}} K^{-1}_{jr} \left[\frac{\mathbf{d} K_{rs}}{\mathbf{d} p_n} K^{-1}_{si} \frac{\partial R_i}{\partial b_m}- \frac{\partial}{\partial b_m} \left( \frac{\mathbf{d}R_r}{\mathbf{d}p_n} \right)\right]
$$

为了避免显式计算和存储极度庞大的 $\mathbf{d} K_{rs}/\mathbf{d} p_n$ 项与雅可比偏微分耦合张量，在代码中 `torchfea` 巧妙运用了高阶伴随方法（Secondary Adjoint Method），只需要执行**两次**额外的线性方程组顺次求解，即可计算上式所描述的总灵敏度修正项：

1. **一阶伴随方程求解**：引入雅可比一阶伴随变量 $\lambda_{rn}^*$，通过求解等价线性系统得到：

   $$
   K_{rj} \lambda_{rn}^* = - \frac{\partial L}{\partial J_{jn}}
   $$
2. **二阶伴随方程求解**：将得到的一阶伴随变量 $\lambda_{rn}^*$ 继续应用于刚度矩阵偏导投影，引入二阶伴随变量 $\lambda_i^{**}$，其对应的线性系统为：

   $$
   K_{is} \lambda_i^{**} = \frac{\mathbf{d} K_{rs}}{\mathbf{d} p_n} \lambda_{rn}^*
   $$

由于以上两步均可以直接复用原位移场静力分析已经完成因式分解（$LDL^T$ 分解）的 $K$ 矩阵，这些额外求解步骤的计算成本微乎其微。

最终组合位移场相关的伴随灵敏度项，系统对设计变量 $b_m$ 的总灵敏度可以通过自动微分自动拼装，完整表达式收敛为：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = 
\frac{\partial L}{\partial b_m} + \lambda_r \frac{\partial R_r}{\partial b_m} + \lambda_i^{**} \frac{\partial R_i}{\partial b_m} + \lambda_{rn}^{*} \frac{\partial}{\partial b_m} \left( \frac{\mathbf{d}R_r}{\mathbf{d}p_n} \right)
$$

## 伴随法 vs 直接法 (Adjoint vs Direct Method)

在计算灵敏度时，有两种基本的策略，选择哪一种取决于目标函数数量和设计变量数量之间的权衡。

### 伴随法 (Adjoint Method)

**适用场景：目标函数（约束）少，设计变量多。** 例如拓扑优化中，通常只有少数几个目标（如柔度最小化、屈曲因子最大化），但设计变量可能是数十万乃至百万级的单元密度。

**原理：** 求解伴随方程 $\boldsymbol{K} \boldsymbol{\lambda} = -\partial L / \partial \boldsymbol{u}$，得到一个伴随向量 $\boldsymbol{\lambda}$，然后通过一次反向传播（`work.backward()`）计算对所有设计变量的梯度。

**计算成本：** 每个目标函数只需求解**一次**额外的线性系统（回代），之后获得所有设计变量的梯度。总计算量约为 $O(N_{\text{obj}} \times N_{\text{dof}}^2)$，与设计变量数量 $N_{\text{var}}$ 无关。

**优点：** 设计变量再多，计算量也不增加，极其适合大规模设计空间的问题。

### 直接法 (Direct Method)

**适用场景：目标函数（约束）多，设计变量少。** 例如对少数几个几何参数（长度、角度、厚度等）进行多目标优化，或需要系统响应对少量参数的完整灵敏度矩阵。

**原理：** 直接对平衡方程 $R_i = 0$ 两端对每个设计变量 $b_m$ 求导：

$$
K_{ik} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} = - \frac{\partial R_i}{\partial b_m}
$$

每求解一个设计变量 $b_m$，都需要一次回代。若 $K$ 已分解，则每次回代成本很低。

**计算成本：** 总计算量约为 $O(N_{\text{var}} \times N_{\text{dof}}^2)$，与设计变量数量成正比，与目标函数数量无关。

**优点：** 一次计算即可得到所有目标函数对所有设计变量的完整灵敏度矩阵，当目标函数很多时无需为每个目标分别求解伴随方程。

### 对比总结

| 方法 | 适合场景 | 每次求解成本 | 总成本 |
|:---:|:--------|:------------|:-------|
| **伴随法** | 目标少、变量多 | 1 次回代 / 目标 | $O(N_{\text{obj}})$ 次回代 |
| **直接法** | 目标多、变量少 | 1 次回代 / 变量 | $O(N_{\text{var}})$ 次回代 |

在 `torchfea` 中，`get_sensitivity` 采用**伴随法**实现，而 `get_jacobian` 中逐参数计算 `dR/dp` 的做法本质上是**直接法**的思路（每个载荷参数一次回代）。常见的使用模式是：

- 当设计变量（如单元密度、节点坐标）数量远大于目标函数数量时 → **用 `get_sensitivity`（伴随法）**
- 当需要系统对少数载荷参数（如几个关键载荷大小）的完整灵敏度矩阵时 → **用 `get_jacobian`（直接法）**
- 当目标函数同时依赖于位移和雅可比矩阵时 → **用 `get_jacobian_sensitivity`（混合方法，结合了伴随法与直接法的思想）**
