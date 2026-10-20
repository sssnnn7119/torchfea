# 材料模型与本构公式

本文档说明 `torchfea` 当前材料内核的约定、参数含义和常见材料模型的理论形式。材料模型通过单位参考体积应变能密度 `W(F)` 定义，单元在高斯点上计算

$$
\mathbf P = \frac{\partial W}{\partial \mathbf F},
\qquad
\mathbf s = \mathbf P^T,
$$

其中 `Element_3D` 内部使用 `s` 的索引顺序。材料还提供一致切线

$$
\mathbb C_{A i k L}
 = \frac{\partial s_{A i}}{\partial F_{k L}},
$$

供静力 Newton 迭代和动力隐式求解使用。用户通常不需要直接调用这些低层方法，只需要构造材料对象并用 `element.set_materials(...)` 分配即可。

## 1. 统一记号与适用范围

本文采用总拉格朗日形式：

$$
\mathbf F = \frac{\partial \mathbf x}{\partial \mathbf X},
\qquad
\mathbf C = \mathbf F^T\mathbf F,
\qquad
J = \det\mathbf F.
$$

`J` 是体积比，必须保持为正。定义

$$
I_1 = \operatorname{tr}\mathbf C,
\qquad
I_2 = \frac12\left[(\operatorname{tr}\mathbf C)^2-\operatorname{tr}(\mathbf C^2)\right],
$$

以及等体积张量和不变量

$$
\bar{\mathbf C}=J^{-2/3}\mathbf C,
\qquad
\bar I_1=J^{-2/3}I_1,
\qquad
\bar I_2=J^{-4/3}I_2.
$$

对主伸长模型，令 $\lambda_i$ 为 $\mathbf C$ 的主伸长，

$$
\lambda_i=\sqrt{\operatorname{eig}_i(\mathbf C)},
\qquad
\bar\lambda_i=J^{-1/3}\lambda_i.
$$

除特别说明外，下面的可压缩模型都采用

$$
W(F)=W_{\mathrm{iso}}(\bar I_1,\bar I_2,\bar\lambda_i)
   +W_{\mathrm{vol}}(J),
\qquad
W_{\mathrm{vol}}=\frac{\kappa}{2}(J-1)^2.
$$

这里 $\kappa$ 是体积模量参数；$\mu$ 表示初始剪切模量时，材料在参考构形附近的线性响应满足

$$
\mu=\frac{E}{2(1+\nu)},
\qquad
\kappa=\frac{E}{3(1-2\nu)},
$$

反向换算为

$$
E=\frac{9\kappa\mu}{3\kappa+\mu},
\qquad
\nu=\frac{3\kappa-2\mu}{2(3\kappa+\mu)}.
$$

这些关系只保证小应变切线一致；有限变形时，不同 $W_{\mathrm{iso}}$ 的高阶项不同，变形结果可以明显不同。

### 1.1 从应变能到应力

对于下面的大多数不变量模型，设

$$
W=W(\bar I_1,\bar I_2,J),
\qquad
W_1=\frac{\partial W}{\partial \bar I_1},
\quad
W_2=\frac{\partial W}{\partial \bar I_2},
\quad
W_J=\frac{\partial W}{\partial J}.
$$

由

$$
\frac{\partial \bar I_1}{\partial \mathbf F}
=2J^{-2/3}\left(\mathbf F-\frac{I_1}{3}\mathbf F^{-T}\right),
$$

$$
\frac{\partial \bar I_2}{\partial \mathbf F}
=2J^{-4/3}\left(I_1\mathbf F-\mathbf F\mathbf C
  -\frac{2I_2}{3}\mathbf F^{-T}\right),
\qquad
\frac{\partial J}{\partial \mathbf F}=J\mathbf F^{-T},
$$

可得统一的一阶 Piola 应力

$$
\mathbf P
=W_1\frac{\partial \bar I_1}{\partial\mathbf F}
 +W_2\frac{\partial \bar I_2}{\partial\mathbf F}
 +W_JJ\mathbf F^{-T}.
$$

内核返回的是 $\mathbf s=\mathbf P^T$。一致切线是对该返回量继续求导：

$$
\mathbb C_{A i k L}=\frac{\partial s_{A i}}{\partial F_{kL}}.
$$

因此，各个不变量模型只需给出 $W$ 对 $\bar I_1$、$\bar I_2$ 和 $J$ 的一、二阶导数，就能统一组装应力和切线。以公共体积势能为例，

$$
W_{\mathrm{vol}}=\frac{\kappa}{2}(J-1)^2
\Longrightarrow
W_J=\kappa(J-1),
\qquad
\mathbf P_{\mathrm{vol}}=\kappa J(J-1)\mathbf F^{-T}.
$$

`NeoHookeanLnJ` 只替换体积项：

$$
W_{\mathrm{vol}}=\frac{\kappa}{2}(\ln J)^2
\Longrightarrow
W_J=\frac{\kappa\ln J}{J},
\qquad
\mathbf P_{\mathrm{vol}}=\kappa\ln J\,\mathbf F^{-T}.
$$

## 2. 现有材料模型

### 2.1 `NeoHookean`

接口：`NeoHookean(mu, kappa)`。

$$
W=\frac{\mu}{2}(\bar I_1-3)+\frac{\kappa}{2}(J-1)^2.
$$

其中 $\mu$ 是初始剪切模量，$\kappa$ 是体积模量。它是最简单的等体积超弹性模型，适合作为基准模型。

### 2.2 `NeoHookeanLnJ`

接口：`NeoHookeanLnJ(mu, kappa)`。

$$
W=\frac{\mu}{2}(\bar I_1-3)+\frac{\kappa}{2}(\ln J)^2.
$$

与 `NeoHookean` 具有相同的初始剪切和体积响应，但有限体积变化时的体积势能不同。两者不能仅凭参数数值相同就认为有限变形结果相同。

### 2.3 `MooneyRivlin`

接口：`MooneyRivlin(c10, c01, kappa)`。

$$
W=c_{10}(\bar I_1-3)+c_{01}(\bar I_2-3)+\frac{\kappa}{2}(J-1)^2.
$$

参考构形附近的初始剪切模量为

$$
\mu_0=2(c_{10}+c_{01}).
$$

因此可以用 $c_{10}=\mu_0/2,c_{01}=0$ 得到与 Neo-Hookean 相同的等体积能量；也可以取 $c_{10}=c_{01}=\mu_0/4$，使两阶不变量共同参与有限变形响应。

### 2.4 `Yeoh`

接口：`Yeoh(c1, c2, c3, kappa)`，令 $x=\bar I_1-3$。

$$
W=c_1x+c_2x^2+c_3x^3+\frac{\kappa}{2}(J-1)^2.
$$

初始剪切模量为

$$
\mu_0=2c_1.
$$

`c2`、`c3` 不改变参考构形处的线性模量，但会改变中、大应变下的硬化或软化趋势。使用时应确保给定变形范围内能量和切线保持物理可接受。

### 2.5 `Gent`

接口：`Gent(mu, Jm, kappa)`。

$$
W=-\frac{\mu J_m}{2}
  \ln\left(1-\frac{\bar I_1-3}{J_m}\right)
  +\frac{\kappa}{2}(J-1)^2.
$$

$J_m$ 控制网络的极限链伸长，必须满足

$$
\bar I_1-3<J_m.
$$

在参考构形处 $\mu_0=\mu$。接近锁定边界时切线会快速增大，Newton 迭代可能需要更小的加载步或更好的初始解。

### 2.6 `ArrudaBoyce`

接口：`ArrudaBoyce(mu, N, kappa)`。当前内核使用五项八链模型的级数截断：

$$
\begin{aligned}
W_{\mathrm{iso}}=\mu\bigg[&c_1(\bar I_1-3)
 +c_2(\bar I_1^2-3^2)
 +c_3(\bar I_1^3-3^3)\\
 &+c_4(\bar I_1^4-3^4)
 +c_5(\bar I_1^5-3^5)\bigg],
\end{aligned}
$$

其中

$$
(c_1,c_2,c_3,c_4,c_5)=
\left(\frac12,\frac{1}{20N},\frac{11}{1050N^2},
\frac{19}{7000N^3},\frac{519}{673750N^4}\right).
$$

`N` 是每条统计链的链段数。五项多项式只是在锁定伸长之前的近似，不能替代锁定边界附近的完整逆 Langevin 模型。

需要注意：当前接口中的 `mu` 是上述多项式的整体尺度参数，并不自动等于严格的初始剪切模量。令

$$
S_N=c_1+6c_2+27c_3+108c_4+405c_5,
$$

则要匹配目标初始剪切模量 $\mu_0$，应取

$$
\texttt{mu}=\frac{\mu_0}{2S_N}.
$$

这也是 `examples/material_test/test_linear_elastic.py` 中的参数换算方式。

### 2.7 `Ogden`

接口：`Ogden(mu, alpha, kappa)`，`mu` 和 `alpha` 可以是标量，也可以是一维多项式参数列表。

$$
W=\sum_{p=1}^{n}
\frac{2\mu_p}{\alpha_p^2}
\left(\sum_{i=1}^{3}\bar\lambda_i^{\alpha_p}-3\right)
 +\frac{\kappa}{2}(J-1)^2.
$$

在本实现的参数化下，初始剪切模量为

$$
\mu_0=\sum_{p=1}^{n}\mu_p.
$$

因此多项式模式只要满足 `sum(mu) = mu0` 即可与目标材料匹配小应变剪切模量。`alpha` 不能为零；主伸长必须保持为正。内核对三阶对称张量使用显式解析特征值/特征向量计算，避免调用通用批量特征分解带来的额外 GPU 工作空间。

若 $\mathbf F=\sum_i\lambda_i\,\mathbf n_i\otimes\mathbf N_i$ 为主方向分解，则 Ogden 模型的主伸长导数为

$$
\frac{\partial W}{\partial\lambda_i}
=\sum_p\frac{2\mu_p}{\alpha_p\lambda_i}
\left(\bar\lambda_i^{\alpha_p}
 -\frac13\sum_{j=1}^{3}\bar\lambda_j^{\alpha_p}\right)
 +W_J\frac{J}{\lambda_i}.
$$

相应的主方向一阶 Piola 应力可写为

$$
\mathbf P=\sum_i\frac{\partial W}{\partial\lambda_i}
\,\mathbf n_i\otimes\mathbf N_i.
$$

主伸长相等时，上式中的谱导数采用连续极限；这是实现中重复特征值分支存在的原因。

### 2.8 `LinearElastic`

接口：`LinearElastic(E, nu)`。

该类名为线弹性，但当前内核采用有限运动学下的 Saint-Venant–Kirchhoff（StVK）形式，而不是直接把小应变张量用于任意大变形：

$$
\mathbf E_G=\frac12(\mathbf C-\mathbf I),
$$

$$
W=\frac{\lambda}{2}[\operatorname{tr}(\mathbf E_G)]^2
  +\mu\,\mathbf E_G:\mathbf E_G,
$$

$$
\mathbf S=\lambda\operatorname{tr}(\mathbf E_G)\mathbf I+2\mu\mathbf E_G,
\qquad
\mathbf P=\mathbf F\mathbf S,
$$

其中

$$
\mu=\frac{E}{2(1+\nu)},
\qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
$$

在小应变范围内，它与上面按 $E,\nu$ 换算的超弹性模型具有相同的线性切线；大转动或大拉伸下，StVK 与 Neo-Hookean 等模型的响应不应视为等价。

## 3. 参数等价速查

给定目标 $E,\nu$，先计算 $\mu_0,\kappa$：

| 材料 | 参数设置 | 小应变等价条件 |
|---|---|---|
| Neo-Hookean | `mu=mu0, kappa=kappa` | 直接等价 |
| Neo-Hookean lnJ | `mu=mu0, kappa=kappa` | 直接等价 |
| Mooney–Rivlin | `c10+c01=mu0/2, kappa=kappa` | 初始剪切、体积等价 |
| Yeoh | `c1=mu0/2, c2/c3` 任意高阶参数 | 初始切线等价 |
| Gent | `mu=mu0, Jm` 足够大, `kappa=kappa` | 初始切线等价 |
| Arruda–Boyce | `mu=mu0/(2*S_N), kappa=kappa` | 按五项截断修正后等价 |
| Ogden | `sum(mu_p)=mu0, kappa=kappa` | 初始切线等价 |
| LinearElastic | `E=E, nu=nu` | 与上述 $E,\nu$ 等价 |

“小应变等价”只表示 $F\approx I$ 附近的一阶响应一致，不保证压力、大拉伸、剪切或接近锁定时的位移一致。比较材料模型时，建议使用同一网格、同一载荷、同一边界和同一单位制，并把第一个已收敛结果作为后续材料的 `GC0` 初始解。

## 4. Python 使用方式

从 INP 构建模型后，可直接替换某个单元块的材料：

```python
import torchfea

fem = torchfea.FEA_INP()
fem.read_inp("C3D4.inp")
fe = torchfea.from_inp(fem)

element = fe.assembly.get_part("final_model").elems["C3D4"]
material = torchfea.materials.Gent(mu=0.6, Jm=20.0, kappa=4.5)
element.set_materials({"Gent": material})

fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-8)
result = fe.solve()
```

同一个单元块可以按局部单元编号分配多个材料；每个材料区域不能重叠，并且应覆盖需要参与计算的单元：

```python
element.set_materials(
    torchfea.materials.MooneyRivlin(c10=0.3, c01=0.3, kappa=4.5),
    name="region-a",
    element_indices=[0, 1, 2],
)
```

直接使用新材料时，推荐调用 `torchfea.materials` 的公开类。当前 INP 解析器的材料关键字仍主要映射为 `*Hyperelastic -> NeoHookean` 和 `*Elastic -> LinearElastic`；Mooney–Rivlin、Yeoh、Gent、Arruda–Boyce 和 Ogden 应在 Python 中构造后分配。CAE UI 当前也只提供线弹性、Neo-Hookean 和 Neo-Hookean lnJ 三种编辑项。

相关示例：

- `examples/material_test/test_hyperelastic.py`：逐模型能量、应力和解析切线检查。
- `examples/material_test/test_linear_elastic.py`：C3D4 压力模型下的小应变等价和有限变形对比。
- [有限元使用说明](usage.md)：完整的建模、载荷、边界和求解流程。
