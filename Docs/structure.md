# FEA 有限元分析框架架构文档

## 概述

FEA是一个基于PyTorch的有限元分析框架，支持非线性材料、接触分析等高级功能。该框架采用面向对象的设计，具有清晰的层次结构和模块化的组织方式。

## 整体架构（与当前仓库同步）

```
torchfea/
├── pyproject.toml              # 项目配置
├── readme.md                   # 项目简介
├── docs/                       # 文档
│   ├── structure.md            # 架构说明（本文档）
│   ├── usage.md                # 使用说明
│   └── elements/               # 单元文档
├── examples/                   # 示例与测试用例
│   ├── vis.py                  # 可视化示例
│   ├── contact_test/           # 接触分析示例
│   ├── element_test/           # 单元测试
│   ├── gradient_test/          # 梯度优化测试
│   ├── instance_test/          # 实例测试
│   ├── jacobian_test/          # 雅可比矩阵测试
│   ├── kinetic_test/           # 动力学测试
│   ├── pressure_test/          # 压力载荷测试
│   └── shape_optimization/     # 形状优化示例
└── src/
    └── torchfea/               # 核心包
        ├── __init__.py         # 模块入口
        ├── controller.py       # FEAController 主控制器
        ├── inp.py              # INP 文件解析器
        ├── model/           # 装配模块（核心组织层）
        │   ├── assembly.py     # Assembly 装配与全局矩阵装配
        │   ├── part.py         # Part/Instance 部件与实例
        │   ├── reference_points.py # 参考点
        │   ├── boundarys/      # 边界条件模块
        │   ├── constraints/    # 约束模块
        │   ├── elements/       # 单元模块
        │   │   ├── dimension3/ # 3D 单元实现
        │   │   └── materials/  # 材料模型
        │   └── loads/          # 载荷模块
        ├── optimizer/          # 优化器模块
        └── solver/             # 求解器模块
            ├── basesolver.py   # 求解器基类
            ├── _linear_solver.py # 线性方程组求解
            ├── static/         # 静力学求解器
            │   ├── solver.py   # 静力隐式求解器
            │   └── result.py   # 结果处理
            └── dynamic/        # 动力学求解器
                ├── implicit.py # 动力隐式
                └── explicit.py # 动力显式
```

## 核心组件架构

### 1. 控制层 (Controller Layer)

#### FEAController

- **文件位置**: `src/torchfea/controller.py`
- **功能**: 主控制器，协调装配(Assembly)和求解器(Solver)
- **主要方法**:
  - `initialize()`: 初始化模型
  - `solve()`: 执行求解过程
- **属性**:
  - `assembly`: 装配对象
  - `solver`: 求解器对象

### 2. 装配层 (Model Layer)

装配层是FEA框架的核心组织层，统一管理所有有限元模型组件，包括几何、材料、载荷、约束等。

#### Assembly (装配)

- **文件位置**: `src/torchfea/assemble/assembly.py`
- **功能**: 统一管理整个有限元模型的所有组件
- **核心属性**:
  - `_parts`: 部件字典
  - `_instances`: 实例字典
  - `_surfaces`: 表面集合
  - `_reference_points`: 参考点
  - `_loads`: 载荷集合
  - `_constraints`: 约束集合
  - `boundary_conditions`: 边界条件集合
  - `RGC`: 冗余广义坐标
  - `GC`: 广义坐标

#### 装配层子模块

##### 几何组件

- **Part (部件)**: `src/torchfea/assemble/part.py`，定义几何部件
- **Instance (实例)**: 部件的实例化
- **ReferencePoint (参考点)**: `src/torchfea/assemble/reference_points.py`

##### 单元模块 (`src/torchfea/assemble/elements/`)

- **BaseElement**: `src/torchfea/assemble/elements/base.py`
- **3D单元**: `src/torchfea/assemble/elements/dimension3/` (包含 brick.py, tetrahedral.py 等)
  - `surfaces/`: 表面单元定义
- **材料模块**: `src/torchfea/assemble/elements/materials/`

##### 载荷模块 (`src/torchfea/assemble/loads/`)

- **BaseLoad**: `src/torchfea/assemble/loads/base.py`
- **各类载荷**: `contact.py`, `pressure.py`, `concentrate_force.py` 等

##### 约束与边界 (`src/torchfea/assemble/boundarys/` & `constraints/`)

- **Boundarys**: `src/torchfea/assemble/boundarys/`
  - `boundary_condition.py`: 位移边界条件
  - `boundary_condition_rp.py`: 参考点边界条件
- **Constraints**: `src/torchfea/assemble/constraints/`
  - `couple.py`: 耦合约束

### 3. 求解器层 (Solver Layer)

```
src/torchfea/solver/
├── basesolver.py         # BaseSolver 基类
├── _linear_solver.py     # 线性求解工具
├── static/
│   ├── solver.py         # StaticImplicitSolver 静力隐式
│   └── result.py         # StaticResult 结果类
└── dynamic/
    ├── implicit.py       # DynamicImplicitSolver 动力隐式
    └── explicit.py       # DynamicExplicitSolver 动力显式
```

#### 求解器类型与要点

- **StaticImplicitSolver** (`static/solver.py`)
  - 牛顿-拉夫森迭代，支持线搜索与预条件线性求解（Pardiso/CG）。
  - 核心接口：`get_stiffness_matrix(GC_now) -> (R, K_idx, K_val)`，`solve()`。
- **DynamicImplicitSolver** (`dynamic/implicit.py`)
  - Newmark-β（默认 γ=0.5, β=0.25），以增量能量为目标进行非线性迭代。
  - 使用 `assemble_mass_matrix(GC_now)` 获取质量矩阵，`get_incremental_stiffness_matrix()` 组装切线矩阵。
  - 提供 `get_next_velocity()` 基于 Newmark 更新 `GV/GA`。
- **DynamicExplicitSolver** (`dynamic/explicit.py`)
  - 中心差分时间推进，建议质量集总（lumped mass），临界步长受网格与材料波速限制。
  - 关键步骤：半步速度、位移更新、残余力 R 计算与加速度更新（M⁻¹R）。
  - 支持按时间间隔存储：`time_per_storage`；需合理估计 `Δt` 以保持稳定性。

## 数据流架构

### 1. 模型构建流程

```
INP文件 → FEA_INP解析 → from_inp() → FEAController
                                       ↓
                                   Assembly (装配层)
                                       ↓
                          ┌─────────────┼─────────────┐
                          ↓             ↓             ↓
                      几何组件        载荷组件        约束组件
                    (Parts/Instances) (Loads)      (Constraints)
                          ↓
                    Elements + Materials
```

### 2. 求解流程

```
FEAController.initialize() → Assembly.initialize() → 组件初始化
         ↓                     ↓
       RGC 空间分配              GC/RGC 映射建立
         ↓
         Solver.initialize() → 动力学需 assembly.initialize_dynamic()
         ↓
FEAController.solve() → Solver.solve()
  ├─ 隐式：反复装配 R,K 与质量（动力隐式）→ 线性求解 → 线搜索/收敛
  └─ 显式：质量向量/对角线 M → 时间推进（中心差分）→ 状态存储
```

## 坐标系统

### 广义坐标系统 (GC/RGC)

#### RGC申请与分配机制

系统采用动态的自由度申请机制，各个组件根据自身需求申请RGC空间：

- **实例 (Instance)**: 申请节点数量 × 3个平移自由度的RGC空间

  - `_RGC_requirements = self.part.nodes.shape` (节点数 × 3)
- **参考点 (ReferencePoint)**: 申请6个自由度 (3个平移 + 3个旋转)

  - `_RGC_requirements = 6`
- **载荷 (Loads)**: 根据载荷类型申请相应的RGC空间

  - 基础载荷：继承自 `BaseLoad`，可申请额外的内部变量空间
  - 接触载荷：可能需要额外的拉格朗日乘子空间
- **约束 (Constraints)**: 根据约束类型申请RGC空间

  - 边界条件：不申请额外空间，而是修改现有RGC的自由度状态
  - 耦合约束：可能申请拉格朗日乘子空间

#### RGC分配流程

1. **分配阶段** (`Assembly.initialize()`):

   ```python
   # 为每个组件分配RGC空间
   for ins in self._instances.keys():
       RGC_index = self._allocate_RGC(size=self._instances[ins]._RGC_requirements)

   for rp in self._reference_points.keys():
       RGC_index = self._allocate_RGC(size=self._reference_points[rp]._RGC_requirements)

   for load in self._loads.keys():
       RGC_index = self._allocate_RGC(size=self._loads[load]._RGC_requirements)
   ```
2. **约束处理**: 通过 `set_required_DoFs()`方法标记哪些自由度被约束
3. **RGC到GC映射**: 只有未被约束的自由度参与求解

   - `GC = RGC[remain_index]` (提取活跃自由度)
   - `RGC = apply_constraints(GC)` (应用约束条件)

#### 坐标系统特点

- **动态分配**: 根据模型复杂度动态分配内存
- **分层管理**: 每个组件管理自己的RGC段
- **约束解耦**: 约束通过索引映射实现，不改变RGC结构
- **GPU友好**: 基于PyTorch张量，支持GPU加速计算

## 时间积分与稳定性（动力学）

- 质量矩阵：通过 `Assembly.assemble_mass_matrix(GC_now)` 获取稀疏格式（indices, values）。
  - 动力隐式：直接以稀疏形式加入切线矩阵（质量项系数依 β, Δt）。
  - 动力显式：推荐构建质量“向量”用于集总 M（仅使用对角项或行和集总）。
- 临界时间步长（显式）：与最小单元尺度 L 和材料波速 c 相关，Δt ≲ L/c；需按模型实际估计，并加入安全因子。
- 状态量：GC（位移）、GV（速度）、GA（加速度），显式以半步速度推进；隐式由 Newmark 关系更新。

## 扩展性设计

### 1. 插件化架构

- 所有组件都基于基类设计，支持继承扩展
- 单元、材料、载荷、约束都可以通过继承基类来添加新类型

### 2. 设备无关性

- 基于PyTorch，支持CPU/GPU计算
- 通过 `device`属性统一管理计算设备

### 3. 模块化设计

- 每个功能模块独立，便于维护和测试
- 清晰的接口定义，便于组合使用

## 使用示例

### 基本使用流程

```python
import torchfea

# 从INP文件加载模型
inp = torchfea.FEA_INP()
inp.read_inp('model.inp')
controller = torchfea.from_inp(inp)

# 设置求解器（示例 1：静力隐式）
controller.solver = torchfea.solver.StaticImplicitSolver()

# 初始化和求解
controller.initialize()
result = controller.solve()

# 示例 2：动力隐式（Newmark-β）
controller.solver = torchfea.solver.DynamicImplicitSolver(deltaT=1e-3, time_end=1.0)
controller.initialize()
controller.solve()

# 示例 3：动力显式（中心差分）
controller.solver = torchfea.solver.DynamicExplicitSolver(time_end=1.0, time_per_storage=1e-4)
controller.initialize()
controller.solve()
```

## 特色功能

### 1. 自动微分

- 基于PyTorch的自动微分功能
- 支持梯度计算和优化

### 2. 高阶单元

- 支持二次单元 (C3D10, C3D15, C3D20等)
- 自动生成二次单元的功能

### 3. 接触分析

- 支持自接触和多体接触
- 接触力的自动计算
- 测试样例见 `examples/contact_test/`、`examples/pressure_test/` 与 `examples/gradient_test/`。

### 4. 表面处理

- 丰富的表面单元类型
- 表面集合的管理和操作

## 开发指南

### 添加新单元类型

1. 继承 `BaseElement`或相应的基类
2. 实现必要的方法 (形函数、雅可比等)
3. 在 `__init__.py`中注册新单元

### 添加新材料模型

1. 继承 `Materials_Base`
2. 实现应力-应变关系
3. 在材料初始化函数中添加分支

### 添加新载荷类型

1. 继承 `BaseLoad`
2. 实现载荷计算方法
3. 在载荷模块中注册

## 性能优化

### 1. 内存管理

- 使用PyTorch张量进行高效内存管理
- 避免不必要的数据复制

### 2. 并行计算

- 支持GPU加速计算
- 向量化运算优化

### 3. 稀疏矩阵

- 使用稀疏矩阵存储刚度矩阵
- 高效的线性代数运算

## 测试与示例

- `examples/element_test/`：单元与表面几何、法向、编号一致性等测试
- `examples/kinetic_test/`：显式/隐式动力学时间积分与能量检查
- `examples/contact_test/`：接触算例与基准对比
- `examples/pressure_test/`：表面压力加载验证
- `examples/gradient_test/` 与 `examples/shape_optimization/`：梯度与优化流程

---

*此文档描述了FEA有限元分析框架的整体架构和设计理念，为开发者提供了全面的结构化认识。*
