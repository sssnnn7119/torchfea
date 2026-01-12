FEA —— 基于 PyTorch 的有限元分析框架
======================================

一个支持非线性材料、接触分析与动力学（隐式/显式）的研究型 FEA 框架。采用模块化设计，装配（Assembly）统一管理几何、材料、载荷与约束，求解器层提供静力隐式、动力隐式（Newmark-β）与动力显式（中心差分）。

## 特色

- 装配-求解器分层，接口清晰，易扩展
- 稀疏矩阵装配与 Pardiso/CG 线性求解
- 动力学两种积分：Newmark-β（隐式）与中心差分（显式，集总质量）
- 接触（自接触/体-体接触）、压力、体力等载荷组件
- 基于 PyTorch，可使用 GPU 与自动微分做灵敏度/优化

## 快速开始

1) 安装依赖（略）。推荐 Python 3.10+、PyTorch（float64）。
2) 运行最小示例脚本（静力 + 表面压力）：

在仓库根目录下执行（Windows PowerShell）：

```powershell
python .\Docs\examples\run_static_pressure.py
```

脚本会读取 `tests/pressure_test/C3D4Less.inp`，在 `final_model` 上施加表面压力并固定底部节点，调用静力隐式求解，并将位移向量保存至 `out/static_pressure_GC.npy`。

3) 更多用法示例：

- 使用说明（从 INP 到求解、导出与可视化）见：`Docs/usage.md`
- 框架架构与数据流见：`Docs/structure.md`

## 目录导航

- `src/`torchfea：核心代码（装配、元素、载荷、约束、求解器等）
- `examples/`：各类算例与验证（元素、压力、接触、动力学、梯度/优化）
- `docs/`：架构与使用文档，`Docs/examples/` 提供可直接运行的示例脚本

## 许可

研究用途优先。若用于生产或商业，请先评估并完善必要的数值与工程健壮性保障。
