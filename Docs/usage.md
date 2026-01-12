# FEA 使用说明（基于 tests 实例）

本文档汇总了仓库中 tests 下常用范式，帮助你快速完成从 INP 构建模型、添加载荷与约束、选择求解器并运行，再到导出与可视化的完整流程。

> 参考来源：`tests/element_test/`、`tests/kinetic_test/`、`tests/contact_test/`、`tests/pressure_test/`、`tests/gradient_test/`、`tests/shape_optimization/`

## 环境建议

- 推荐使用 Python 3.10+，PyTorch 配置为默认 float64：
  ```python
  import torch
  torch.set_default_dtype(torch.float64)
  # 可选：使用 GPU
  # torch.set_default_device(torch.device('cuda'))
  ```
- Windows 下如需并行库不冲突，可设置：
  ```python
  import os
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
  ```

## 一、从 INP 构建模型

最常见流程（见 `tests/kinetic_test/test_implicit.py` 等）：

```python
import os
import FEA

current_path = os.path.dirname(os.path.abspath(__file__))

# 1) 读取 INP
fem = FEA.FEA_INP()
fem.read_inp(current_path + '/C3D4Less.inp')  # 某个 Abaqus INP

# 2) 根据 INP 构建控制器（包含 assembly 与 solver 占位）
controller = FEA.from_inp(fem)
```

## 二、添加载荷与约束

在 `controller.assembly` 上进行模型修改（tests 多数使用 `instance_name='final_model'`）：

- 表面压力（`tests/pressure_test/test_benchmark.py`、`tests/kinetic_test/`）：

  ```python
  controller.assembly.add_load(
      FEA.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06)
  )
  ```
- 固定边界（按节点集合，使用新的 boundary 接口；`index_nodes` 来自 INP 集合）：

  ```python
  import numpy as np
  bc_nodes = np.array(list(fem.part['final_model'].sets_nodes['surface_0_Bottom']))
  controller.assembly.add_boundary(
    FEA.boundarys.Boundary_Condition(instance_name='final_model', index_nodes=bc_nodes)
  )
  ```

  - 若需对参考点施加边界（例如固定全部 6 个自由度），可用：

  ```python
  rp = controller.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))
  controller.assembly.add_boundary(
    FEA.boundarys.Boundary_Condition_RP(rp_name=rp, indexDoF=[0, 1, 2, 3, 4, 5])
  )
  ```
- 参考点与耦合（`Couple`），见静力与接触示例：

  ```python
  rp_name = controller.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))
  head_nodes = fem.part['final_model'].sets_nodes['surface_0_Head']
  controller.assembly.add_constraint(
      FEA.constraints.Couple(instance_name='final_model', indexNodes=head_nodes, rp_name=rp_name)
  )
  ```
- 接触（体-体或自接触），见 `tests/contact_test/contact/test_benchmark.py` 与 `tests/gradient_test/test_grad.py`：

  ```python
  # 自接触
  controller.assembly.add_load(
      FEA.loads.ContactSelf(instance_name='final_model', surface_name='surface_0_All')
  )

  # 体-体接触（两个实例）
  controller.assembly.add_load(
      FEA.loads.Contact(instance_name1='final_model', instance_name2='Part-2',
                        surface_name1='surface_0_All', surface_name2='surfaceblock')
  )
  ```
- 体力（密度形式），见 `tests/shape_optimization/test_optimization.py`：

  ```python
  controller.assembly.add_load(
      FEA.loads.BodyForce(instance_name='final_model', element_name='element-0',
                          force_density=[-9.81e-6, 0, 0])
  )
  ```
- 弹簧载荷（新）：

  - 参考点-参考点（RP-RP）弹簧：

    ```python
    rp1 = controller.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))
    rp2 = controller.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 60]))
    controller.assembly.add_load(
        FEA.loads.Spring_RP_RP(rp_name1=rp1, rp_name2=rp2, k=1e3, rest_length=None)  # rest_length 默认初始距离
    )
    ```
  - 参考点-空间点（RP-Point）弹簧：

    ```python
    rp = controller.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))
    controller.assembly.add_load(
        FEA.loads.Spring_RP_Point(rp_name=rp, point=[0, 0, 100], k=1e3, rest_length=None)
    )
    ```

## 三、选择求解器并运行

框架支持静力隐式、动力隐式（Newmark-β）与动力显式（中心差分）。设置求解器后调用 `controller.solve()` 即可。

### 1) 静力隐式（牛顿-拉夫森）

参考：`tests/pressure_test/test_benchmark.py`

```python
controller.solver = FEA.solver.StaticImplicitSolver()
# 可传入 tol_error 调整收敛容限
controller.solve(tol_error=1e-3)

# 最终位移向量：
U_final = controller.assembly.GC  # torch.Tensor
```

### 2) 动力隐式（Newmark-β）

参考：`tests/kinetic_test/test_implicit.py`

```python
controller.solver = FEA.solver.DynamicImplicitSolver(deltaT=1e-3, time_end=0.1)
# 可选：调整 Newmark 参数
# controller.solver._gamma = 0.6
# controller.solver._beta  = 0.3025

controller.solve(tol_error=1e-6)

# 历史数据
GC_hist = controller.solver._GC_list  # [Tensor]
GV_hist = controller.solver._GV_list  # [Tensor]
GA_hist = controller.solver._GA_list  # [Tensor]
T_hist  = controller.solver._time_list  # [float]
```

### 3) 动力显式（中心差分）

参考：`tests/kinetic_test/test_explicit.py`

```python
controller.solver = FEA.solver.DynamicExplicitSolver(time_end=0.1, time_per_storage=1e-4)
controller.solve()

# 历史数据
GC_hist = controller.solver._GC_list
GV_hist = controller.solver._GV_list
GA_hist = controller.solver._GA_list
T_hist  = controller.solver._time_list
```

> 提示：显式法的时间步长稳定性取决于最小单元尺寸与材料波速，必要时缩小 `time_per_storage` 与内部临界步长设置（见 `FEA/solver/dynamic_explicit.py`）。

## 四、导出与可视化

- 导出 numpy 文件（见 kinetic 测试）：

  ```python
  import numpy as np
  out_dir = 'Z:/temp/'  # 自行修改为本地目录
  np.save(out_dir + '/explicitGC.npy', np.array([x.tolist() for x in controller.solver._GC_list], dtype=np.float32))
  np.save(out_dir + '/explicitGV.npy', np.array([x.tolist() for x in controller.solver._GV_list], dtype=np.float32))
  np.save(out_dir + '/explicitGA.npy', np.array([x.tolist() for x in controller.solver._GA_list], dtype=np.float32))
  ```
- 可视化（PyVista，见多处 tests）：

  ```python
  import pyvista as pv
  import numpy as np

  ins = controller.assembly.get_instance('final_model')
  U = controller.assembly.RGC[ins._RGC_index].cpu().numpy()
  undeformed = ins.nodes.cpu().numpy()
  deformed = undeformed + U

  # 三角面来自某表面集合
  tris = ins.surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()

  scalars = (U**2).sum(axis=1)**0.5
  faces = np.column_stack([np.full(len(tris), 3), tris])
  mesh = pv.PolyData(deformed, faces)
  mesh['displacement'] = scalars
  plotter = pv.Plotter()
  plotter.add_mesh(mesh, scalars='displacement')
  plotter.show()
  ```

## 五、常见技巧与排错

- 默认类型与设备：tests 多以 `float64`、有时 GPU 运行；如遇显存不足可退回 CPU 或降低模型规模。
- 显式不稳定/发散：
  - 减小临界时间步长（见 solver 内部估计）；
  - 采用集总质量（已内置）与轻微数值阻尼（可在自定义中加入）。
- 隐式难收敛：
  - 检查边界与接触设置是否矛盾；
  - 关注线搜索输出与最大步长限制（`_maximum_step_length`）。
- 约束与集合：`index_nodes`/`surface_set` 均来自 INP 内的集合；请确保 INP 定义与代码一致（常见实例名：`final_model`）。

## 六、更多示例指引

- 静力 + 自接触：`tests/gradient_test/test_grad.py`
- 双体接触：`tests/contact_test/contact/test_benchmark.py`
- 动力学（隐式/显式）：`tests/kinetic_test/`
- 单元与表面相关：`tests/element_test/`
- 压力加载：`tests/pressure_test/`
- 优化/灵敏度：`tests/shape_optimization/`、`tests/gradient_test/`
