# 导出 Abaqus 静力作业

直接从 controller 导出完整作业，不需要先初始化或求解：

```python
controller.export_inp("model.inp")
```

整个流程统一经过现有的 inp.py：

```python
inp_model = controller.to_inp()
# inp_model 是 torchfea.FEA_INP，可检查和编辑后再写文件。
settings = inp_model.steps[0].static
settings.initial_increment = 0.05
settings.minimum_increment = 1e-7
settings.maximum_increment = 0.2
inp_model.write_inp("model.inp")
```

也可以直接使用 `FEA_INP.from_controller(controller)`。
controller.export_inp 只是 to_inp().write_inp() 的便捷入口。
几何和物理映射集中在 converters/inp.py，通用 INP 解析、模型表示和写出统一在 inp.py。

转换层包含两个明确的方向：

```python
from torchfea.converters import controller_to_inp, inp_to_controller

inp_model = controller_to_inp(controller)
# 仅对求解器支持的 Part/实体材料子集进行反向转换：
controller = inp_to_controller(inp_model)
```

inp_to_controller 的支持范围与 INP 文件保真读写是两件事：
例如平铺导出的完整 Abaqus 作业可以无损读写，但目前不能通过反向转换
恢复为含全部载荷、接触和分析步的 torchfea controller。
原有 torchfea.from_inp 是 inp_to_controller 的公开别名。
新增其他格式时，在 converters 下增加对应模块，避免把映射逻辑放进包入口或文件读写类。

## 完整读取与无损写回

```python
inp_model = torchfea.FEA_INP()
inp_model.read_inp("original.inp")
inp_model.write_inp("copy.inp")
```

未修改的文件按字节原样写回，包括注释、空行、大小写、参数顺序、
关键字和数据续行、混合换行、编码字节与文件末尾是否换行。
所有关键字均进入 records，包括当前 torchfea 无法求解的接触、摩擦、
用户材料、输出请求、多分析步等；读取和写回不会丢掉这些内容。
`*INCLUDE` 保留原始引用，不展开或自动拷贝外部文件。
把文件移到其他目录时，需要同时维护其外部依赖的路径。

结构按 Part、Material、Assembly、Step 分层，字段由带 slots 的 dataclass
固定定义，集合使用有类型的列表，不使用参数字典、getattr 或动态属性。
例如：

```python
inp_model.part[0].name
inp_model.part[0].elements[0].connectivity
inp_model.material[0].elastic.young_modulus
inp_model.assemble.instances[0].translation
inp_model.steps[0].boundaries[0].values[0].value
inp_model.steps[0].concentrated_loads[0].values[0].magnitude
inp_model.steps[0].static.initial_increment
```

这些属性引用 records 中相同的静态对象，修改属性会直接反映在写出的文件里。
新建记录时先加入 records，再调用 refresh_views() 建立分层引用。
平铺导出的几何放在 mesh；原有带 Part 定义的文件放在 part 列表。
未知关键字或暂不能数值解释的参数化记录使用固定的 RawRecord 类，
其 keyword_name 和 text 保存原始文本；不会转换成任意参数字典。
完整 INP 文件模型不意味着 torchfea 求解器已经支持其中所有物理类型。

在安装 Abaqus 的环境运行：

```bash
abaqus job=model input=model.inp datacheck interactive
abaqus job=model input=model.inp interactive
```

文件包含网格、材料、实体截面、参考点、边界、支持的载荷/耦合、
静力分析步和 U/RF/S/E 场输出。实例的旋转和平移直接写入未变形节点坐标，
采用整体编号的 flat assembly。每个实例的集合和表面独立编号，原名称写在注释里。
单位制沿用当前模型。

导出的是模型的最终载荷参数和初始几何；求解结果 GC、临时连续加载系数、
Newton 状态均不影响文件。连续加载和回退由 Abaqus 的静力自动增量完成。

## 分析步设置

```python
controller.export_inp(
    "model.inp",
    nlgeom=True,
    initial_increment=0.1,
    minimum_increment=1e-6,
    maximum_increment=0.1,
    maximum_increments=1000,
)
```

步长均相对于总步时间 1。当前只支持 Abaqus/Standard 静力步；
controller 可以尚未设置 solver，已设置时要求为 StaticImplicitSolver。
仅输出启用的对象，启用对象引用已禁用实例时会报错。

## 映射

| torchfea 对象 | Abaqus 定义 |
|---|---|
| C3D4、C3D6、C3D8、C3D10、C3D15、C3D20 | 同名实体单元，自动处理二十节点单元续行 |
| LinearElastic | *ELASTIC，保留 E、nu |
| NeoHookean | *HYPERELASTIC, NEO HOOKE；C10=mu/2，D1=2/kappa |
| MooneyRivlin | *HYPERELASTIC, MOONEY-RIVLIN；D1=2/kappa |
| Yeoh | *HYPERELASTIC, YEOH；D1=2/kappa，D2=D3=0 |
| Ogden | *HYPERELASTIC, OGDEN，保留各项 mu、alpha；D1=2/kappa，其余 Di=0 |
| 材料分区、逐单元材料参数/密度 | 按参数组合生成材料与 *SOLID SECTION |
| Boundary_Condition / Boundary_Condition_RP | *BOUNDARY，内部零基自由度转为 Abaqus 一基编号 |
| Couple | 节点表面 + *COUPLING / *KINEMATIC |
| Concentrate_Force / Moment | *CLOAD；同节点同自由度载荷自动累加 |
| Pressure | *DSLOAD, P；正值朝实体内部，与现有表面法向约定一致 |
| BodyForce | *DLOAD, BX/BY/BZ，直接使用单位体积的力 |
| Penalty_DoF | *SPRING1 刚度 k，加 *CLOAD=k×target |
| Spring_RP_RP / Spring_RP_Point | *SPRINGA，要求初始长度非零且等于自由长度；固定端增加固定节点 |
| Contact / ContactSelf | 默认输出无摩擦硬接触 general contact，并保留接触表面配对/自接触范围 |

接触无需设置额外选项，直接调用 controller.export_inp("model.inp") 即会替换为
无摩擦硬接触。自定义指数势能参数和自接触初始排除过滤不转换。

Penalty_DoF 的最终残差仍为 k×(u-target)，并保留有限罚刚度。
Abaqus 在加载过程中保持弹簧刚度 k，逐渐增加等效力 k×target；
因此其增量路径与 torchfea 逐渐增加罚函数整体系数的路径不同。

LinearElastic 在 torchfea 中采用 St Venant–Kirchhoff 有限应变本构。
Abaqus 的标准 *ELASTIC 在大应变下采用不同的本构表达：
nlgeom=True 时导出器会明确警告并把说明写入 INP 注释。
两者在小应变下对应相同的 E、nu，大应变下不能据此要求数值一致。
同样，Abaqus 旋转弹簧和旋转向量罚函数在有限旋转下也存在差别。

## 明确拒绝的情况

罚函数关节、NeoHookeanLnJ、Gent、
ArrudaBoyce 以及用户自定义单元/材料/载荷目前没有实现可靠映射，
会抛出 NotImplementedError，指出具体对象。它们不会被忽略，
也不会被自动替换为 Abaqus 的理想关节。

需要将关节替换为 Abaqus 理想连接器时，显式开启：

```python
controller.export_inp("native.inp", native_interactions=True)
```

此时球铰转换为理想 JOIN，纯旋转单自由度关节转换为 HINGE，
纯平移单自由度关节转换为 TRANSLATOR。连接点与参考点不重合时，
通过 BEAM MPC 保持连接点的刚性偏置，局部 1 轴对应指定的自由方向。
这些理想关节不保留原来的有限罚刚度；
同时含平移/旋转的六维螺旋运动向量仍会明确拒绝。
native_interactions 仅控制关节映射；Contact/ContactSelf 始终输出无摩擦硬接触。
这些映射差异都会发出警告并写入 INP 注释。

多个材料势能叠加、材料分区遗漏/重叠、高斯点间变化的材料参数、
预拉伸弹簧也会明确拒绝。边界或罚函数直接施加在 Couple 从属节点上的模型，
需要先在建模时把条件转换到参考点，避免 Abaqus 的从属自由度重复约束。

导出器先在内存中完成转换和校验，再打开目标文件；模型不支持时不会覆盖已有文件。
这项功能生成标准关键字作业文件，并不能保证任意欠约束、网格畸变或物理失稳模型收敛。

## 验证

测试覆盖完整作业结构、稀疏原始标签重编号、重复实例和变换、
材料分区/密度、各超弹性常数与多行输入、罚函数等价性、
所有六种实体单元、默认无摩擦硬接触和可选理想关节，以及导出失败时保留已有文件。
仓库内的所有示例 INP 都进行读入再写出的逐字节一致性验证，
另有未知关键字、参数化数据、混合换行、非 UTF-8 编码字节、
注释、关键字编辑和 controller→FEA_INP→文件的测试。

若系统能够找到 abaqus 命令，tests/test_abaqus_export.py 中的 datacheck
测试会提交一个小模型检查；没有 Abaqus 时该测试会跳过。

关键字参考：

- [Abaqus *COUPLING](https://docs.software.vt.edu/abaqusv2026/English/SIMACAEKEYRefMap/simakey-r-coupling.htm)
- [Abaqus *KINEMATIC](https://docs.software.vt.edu/abaqusv2026/English/SIMACAEKEYRefMap/simakey-r-kinematic.htm)
- [Abaqus *SPRING](https://docs.software.vt.edu/abaqusv2026/English/SIMACAEKEYRefMap/simakey-r-spring.htm)
- [Abaqus *HYPERELASTIC](https://docs.software.vt.edu/abaqusv2026/English/SIMACAEKEYRefMap/simakey-r-hyperelastic.htm)
- [Abaqus 体积力定义](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEPRCRefMap/simaprc-c-loaddistributed.htm)
