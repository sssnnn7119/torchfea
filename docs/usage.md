# FEA 使用说明

本文档给出一条从模型输入到结果后处理的完整路径。推荐先理解“模型定义 → 工况 → 初始化 → 求解 → 结果”的顺序，再根据需要阅读 [材料模型与本构公式](materials.md)、[载荷](loads.md)、[求解器](solver.md)、[有限元单元](elements/elements_types_CHN.md) 和 [等参变换与数值积分](elements/isoparametric_and_quadrature_CHN.md)。

仓库中的可运行参考主要位于 `examples/`：压力算例在 `examples/pressure_test/`，材料比较在 `examples/material_test/`，动力学在 `examples/kinetic_test/`，接触在 `examples/contact_test/`，灵敏度在 `examples/gradient_test/`。

## 1. 环境与数值约定

`torchfea` 在包入口将默认 PyTorch dtype 设为 `float64`。有限元非线性求解建议保持双精度；GPU 只改变计算设备，不改变材料参数的单位。

```python
import torch
import torchfea

torch.set_default_dtype(torch.float64)
# 可选：在脚本最开始指定设备
# torch.set_default_device(torch.device("cuda"))
```

所有几何、材料、载荷和密度必须使用同一单位制。例如若长度用 mm、力用 N，则压力是 N/mm²，密度应是质量/mm³，时间单位也必须与此一致。

## 2. 构建模型

### 2.1 从 INP 读取

```python
from pathlib import Path
import torchfea

fem = torchfea.FEA_INP()
fem.read_inp(Path("C3D4.inp"))
fe = torchfea.from_inp(fem)
```

`from_inp` 会创建 `Assembly`、`Part`、`Instance`、单元、集合以及 INP 中能够识别的材料。当前材料关键字主要映射为 `*Hyperelastic -> NeoHookean` 和 `*Elastic -> LinearElastic`；其他超弹性模型请在 Python 中构造后分配，具体见 [材料文档](materials.md)。

### 2.2 直接使用 CAD

若需要从基本体、布尔特征或 STEP 历史生成网格，可使用 `FEAController.cad`。CAE UI 使用的也是同一套内核对象；示例和集合映射规则见 [CAE UI 文档](cae_ui.md)。

## 3. 设置材料

INP 模型中的单元块可以直接替换材料：

```python
part = fe.assembly.get_part("final_model")
element = part.elems["C3D4"]

element.set_materials({
    "Gent": torchfea.materials.Gent(mu=0.6, Jm=20.0, kappa=4.5),
})
```

也可以按单元块内的局部编号分配多个材料。区域不能重叠；参与计算的单元都应有材料：

```python
element.set_materials(
    torchfea.materials.MooneyRivlin(c10=0.3, c01=0.3, kappa=4.5),
    name="region-a",
    element_indices=[0, 1, 2],
)
```

当前内核提供 `LinearElastic`、`NeoHookean`、`NeoHookeanLnJ`、`MooneyRivlin`、`Yeoh`、`Gent`、`ArrudaBoyce` 和 `Ogden`。材料能量、应力、切线和小应变等价参数的公式集中写在 [materials.md](materials.md)；不要把“初始模量相同”理解为有限变形下响应完全相同。

## 4. 定义载荷、边界和约束

模型修改都通过 `fe.assembly` 完成。添加对象的统一形式是 `add_xxx(object, name="...")`，对象在前、名称在后。

### 4.1 压力与边界条件

```python
fe.assembly.add_load(
    torchfea.loads.Pressure(
        instance_name="final_model",
        surface_set="surface_1_All",
        pressure=0.06,
    ),
    name="pressure-1",
)

fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(
        instance_name="final_model",
        set_nodes_name="surface_0_Bottom",
        indexDoF=[0, 1, 2],       # U1, U2, U3
    ),
    name="fixed-bottom",
)
```

边界条件中的 `indexDoF` 使用零基索引 `[0, 1, 2]`，分别对应平移自由度 U1、U2、U3。参考点的 6 个自由度依次为 U1、U2、U3、R1、R2、R3：

```python
rp = fe.assembly.add_reference_point(
    torchfea.ReferencePoint([0.0, 0.0, 80.0]),
    name="RP-1",
)
fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition_RP(
        rp_name=rp,
        indexDoF=[0, 1, 2, 3, 4, 5],
    ),
    name="fixed-rp",
)
```

### 4.2 参考点耦合、接触和其他载荷

参考点耦合和参考点之间的运动学关节属于坐标消元约束。其 RGC/GC 组织、正向坐标重构以及残差和刚度映射算法见 [约束算法说明](constraints.md)。

```python
fe.assembly.add_constraint(
    torchfea.constraints.Couple(
        instance_name="final_model",
        set_nodes_name="surface_0_Head",
        rp_name=rp,
    ),
    name="head-couple",
)

fe.assembly.add_load(
    torchfea.loads.BodyForce(
        instance_name="final_model",
        element_name="element-0",
        force_density=[0.0, 0.0, -9.81e-6],
    ),
    name="gravity",
)

fe.assembly.add_load(
    torchfea.loads.Spring_RP_Point(
        rp_name=rp,
        point=[0.0, 0.0, 100.0],
        k=1e3,
        rest_length=None,        # 默认取初始距离
    ),
    name="spring-ground",
)
```

自接触使用 `ContactSelf(instance_name=..., surface_name=...)`，两实例接触使用 `Contact(instance_name1=..., instance_name2=..., surface_name1=..., surface_name2=...)`。参数和势能符号约定见 [loads.md](loads.md)。

## 5. 初始化与求解

### 5.1 静力隐式

将收敛容差传给求解器构造函数，而不是传给 `controller.solve`：

```python
fe.solver = torchfea.solver.StaticImplicitSolver(
    maximum_iteration=10000,
    tol_error=1e-8,
)
result = fe.solve()

print(result.converged)
U = result.GC                 # 约化后的全局自由度向量
```

首次 `fe.solve()` 会自动调用 `fe.initialize()`。如果希望分开初始化，也可以显式调用 `fe.initialize()` 后再调用 `fe.solve()`；正常使用时不需要重复初始化。

### 5.2 用第一次结果作为后续材料的初始解

比较材料模型时，建议保持网格、载荷和边界不变，只更换材料；第一个工况收敛后，把它的 `GC` 作为后续求解的初始猜测：

```python
first = fe.solve()
first_solution = first.GC.detach().clone()

element.set_materials({
    "Ogden": torchfea.materials.Ogden(
        mu=[0.3, 0.3], alpha=[2.0, 5.0], kappa=4.5,
    ),
})
second = fe.solve(GC0=first_solution, if_initialize=False)
```

`if_initialize=False` 适用于拓扑、集合、载荷和边界结构未改变，仅更新参数或材料的情况。若改变了网格、实例、集合、约束或载荷对象，应使用默认的 `if_initialize=True` 重新初始化。

### 5.3 动力学

动力隐式采用 Newmark-$\beta$：

```python
fe.solver = torchfea.solver.DynamicImplicitSolver(
    deltaT=1e-3,
    time_end=0.1,
    tol_error=1e-6,
)
result = fe.solve()
GC_hist = result.GC_list
GV_hist = result.GV_list
GA_hist = result.GA_list
T_hist = result.time_list
```

动力显式使用中心差分，并由内核估计临界时间步：

```python
fe.solver = torchfea.solver.DynamicExplicitSolver(
    time_end=0.1,
    time_per_storage=1e-4,
)
result = fe.solve()
```

显式计算对最小单元尺寸、材料波速和密度敏感；工程使用时应检查估计步长和能量稳定性。求解器的离散公式和收敛过程见 [solver.md](solver.md)。

## 6. 结果、持久化与可视化

静力结果包含 `GC`、收敛状态、工况和计时信息；若通过 `fe.solve(need_jacobian=True)` 请求雅可比，还会在结果中保存按载荷名称分组的 `jacobian`。

```python
result.save("static_result.npz")
loaded_result = torchfea.solver.StaticResult.load("static_result.npz")
```

`result.save/load` 保存求解结果；若要保存完整的 CAD、装配、材料、载荷、边界和求解器配置，应使用 controller 级别接口：

```python
fe.save_model("model_state.npz")
fe2 = torchfea.load_model("model_state.npz")
result2 = fe2.solve()
```

模型和结果带有模型哈希校验。加载结果用于后处理前，应确认结果对应当前模型；直接修改节点、材料或载荷后，应重新求解。

实例表面可以用 PyVista 显示：

```python
import pyvista as pv

instance = fe.assembly.get_instance("final_model")
rgc = fe.GC2RGC(result.GC)
mesh = instance.get_mesh(RGC=rgc, surf_name="surface_0_All")
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True)
plotter.show()
```

## 7. 诊断顺序

遇到不收敛、变形异常或材料之间结果差异很大时，可按以下顺序检查：

1. 单位制是否一致，尤其是 `E`/`mu`、`kappa`、压力和密度。
2. 约束是否去除了刚体运动，表面集合和实例名称是否存在。
3. `J` 是否保持为正，Gent 是否满足 `I1bar - 3 < Jm`。
4. 不同材料是否真的只在高阶项上不同；小应变对应关系见 [materials.md](materials.md)。
5. 非线性载荷是否需要更小的加载步、更好的 `GC0` 或更严格/合适的 `tol_error`。
