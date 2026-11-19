# LiteRT 源码笔记（四）：从算子注册到 Delegate 图改写

上一篇：[InterpreterBuilder 如何搭建执行图](03-interpreter-builder.md) · 下一篇：[源码跟读实验](05-reading-lab.md)

知道模型里有一个 ABS，不等于运行时已经知道怎样求绝对值；知道环境里有 XNNPACK，也不等于 ABS 一定会交给它。本篇顺着两条线看：一条是 **Resolver 为原节点找实现**，另一条是 **Delegate 把满足条件的一部分原节点换成新的执行节点**。这里讨论的是 `tflite/` 下的 Interpreter 路径，不是直接对上一篇模型 IR 调用这些方法。节选以原稿的接口形态为主，宏、默认委托、支持条件和错误处理应以手头的 checkout 为准。

## 1. 从 `OpResolver` 查到 `TfLiteRegistration`

| 需要回答的问题 | 对象 |
| --- | --- |
| 这个内建算子是什么版本？ | `FindOp(builtin_code, version)` |
| 这个自定义算子叫什么、是什么版本？ | `FindOp(custom_name, version)` |
| 可以提供哪些默认 Delegate？ | `GetDelegateCreators()` / `GetOpaqueDelegateCreators()` 返回工厂，不是已创建实例 |
| 是否可能含有用户自定义算子？ | `MayContainUserDefinedOps()`；基类默认保守返回 true，并不说明已找到一个自定义节点 |

Opaque 工厂用于不透明 C API 对象，不能仅按「是不是 Google Play services」来理解。两个 `FindOp` 返回的是注册表项，不是执行结果，也不凭名称或版本号推导向后兼容。

`MutableOpResolver` 在抽象接口之上维护 `(code, version) → registration` 和 `(name, version) → registration`。查找时先看本地表，再按顺序查链里的 Resolver：

```cpp
// 逻辑示意，省略类型声明和错误处理。
if (auto* reg = FindInLocalTable(key, version)) return reg;
for (auto* resolver : other_op_resolvers_) {
  if (auto* reg = resolver->FindOp(key, version)) return reg;
}
return nullptr;
```

- `AddBuiltin` 和 `AddCustom` 分别按枚举和名称注册。在所指定的版本或版本区间内复制并补齐注册信息；同键覆盖原条目。复制的是注册记录，函数指针并不会复制一份函数实现。
- `AddAll` 合入另一个 Mutable Resolver 的映射，并在原稿实现中带入其链式引用；`ChainOpResolver` 本身保存引用，未命中时才调用对方，不是复制条目。链中的对象必须活到查找结束。
- `may_directly_contain_user_defined_ops_` 是保守标记。查询还会考虑链上的 Resolver，不能拿它枚举模型使用的自定义节点。
- `BuiltinOpResolver` 在 Mutable Resolver 上注册标准内建实现，也可能提供默认 Delegate 工厂。它是常见选择，不是唯一选择；应用可以显式提供更小的注册集合，或选择不附自动默认委托的 Resolver。

## 2. Registration 是实现表，Node 是图中的一次使用

![图 4-1：Registration 与 Node：同一类实现，多次节点使用](figures/07-node-kernel.png)

*图 4-1　Registration 与 Node：同一类实现，多次节点使用。两条 ABS 只是示例；运行时可以为每个节点分别保存 Registration 记录，但回调可指向同一组函数。init 生成的节点状态彼此独立；prepare、invoke 和 free 的实际时机见正文。*

`TfLiteRegistration` 是 C 风格结构体，不是 C++ 抽象基类。源码保留若干用于兼容的版本结构（例如 `TfLiteRegistration_V3`）；不要将兼容结构的名字等同于当前完整结构的字段集。

| Registration 字段 | 如何理解 |
| --- | --- |
| `init(context, buffer, length)` | 节点初始化，返回内核私有状态，保存到 `node.user_data`。自定义算子通常读取 options 字节及长度；内建算子可通过长度为 0 的调用接收内建参数指针，具体以构图入口为准。无需状态的算子允许省略。 |
| `free(context, data)` | 清理的是 `init` 返回的状态，通常不是传给 `init` 的 options 指针。一次初始化对应一次清理；重建节点不是再次调用同一实例的生命周期。 |
| `prepare(context, node)` | 推导输出 shape、校验参数、准备临时存储；分配或动态尺寸变化等场景可能重跑，不是每一次 `Invoke` 都必定跑。合法已解析的算子可以没有 Prepare。 |
| `invoke(context, node)` | 真正执行本节点的计算。 |
| `profiling_string` / `async_kernel` | 前者提供 profiling 描述；后者提供受支持的异步执行接口。返回空不等于内核不能在线程池里使用并行计算。 |
| `registration_external` / `inplace_operator` | 前者连接不透明 `TfLiteOperator` 接口；后者只是可原地复用的声明，在生命周期等条件满足时才可能共享存储。 |

`TfLiteNode` 是节点实例。多个同类节点可以共享实现，但各有自己的 `inputs`、`outputs`（Subgraph 张量表下标）、`intermediates`、`temporaries`、`builtin_data`、`custom_initial_data` 和 `user_data`；委托节点还有 `delegate`，节点可能有副作用。所谓静态内存构建（`TF_LITE_STATIC_MEMORY`）影响可用的辅助 API，不能据此把常规节点理解为只有输入、输出和指针。

`Subgraph::OpPrepare` 把节点交给具体实现：若 external 包装引用已有内部节点，则调用该 registration；否则优先使用带 `user_data` 的 opaque 回调，再尝试不带 data 的回调；最后可使用普通 `op_reg.prepare`。**已解析且 Prepare 为空可以成功；未解析的 Custom/Flex 占位符如果一路留到准备阶段才报错。** 这也解释了为什么构建时可能暂时接受缺少普通 CPU 实现的节点：某个 Delegate 还可能接管它。

## 3. 用 ABS 跟一遍普通 CPU 内核

源码入口：`tflite/kernels/elementwise.cc` 中的 `Register_ABS`，再查 `AbsInit/Free` 对应的回调和 `ElementwisePrepare` / `AbsEval`；名称可随版本变化。把生成回调的宏展开来看，仍是普通函数指针调用。

1. **Init / Free**：公共 elementwise 内核用 `OpData` 保存量化用的 multiplier、shift、输入/输出零点、`needs_rescale` 以及 INT8 / INT16 查询表指针。Free 清理该状态及其拥有的表。查找表用于适合预计算的非线性操作（例如部分 LOG/SQRT 路径，具体随类型和实现），不是每个 ABS 都要建表。
2. **Prepare**：获取输入、输出张量，检查类型与量化参数。原稿 ABS 实现涉及 FLOAT32、INT8、INT16、INT32；不要把这些类型推广到所有一元算子。仿射量化时，从输入 / 输出 scale 得到比例，转换成固定点 multiplier 与 shift。ABS 的输出形状跟输入一致，调用 `context->ResizeTensor`；这是 Subgraph 填入的回调，不是 ABS 自己管理全部张量内存。
3. **Resize 的两层**：某些形状/存储等价条件下可以快速复用已有内存；需要变化时进入 `ResizeTensorImpl`。动态张量可即时调整存储，arena 张量调整尺寸/字节数并交给后续内存规划。字符串、Resource、Variant 有特殊存储规则，不能直接按维度乘固定字节数。
4. **Invoke**：浮点等非量化路径走直接逐元素计算；量化路径先减输入零点、取绝对值，必要时换算输出尺度，再加入输出零点并夹到目标范围（INT8 为 −128 到 127）。非量化 INT16 的中间值提升到 32 位，不代表最终结果天然能用 INT16 表示；极值要按所读内核的校验/转换处理。

```text
量化路径示意：
centered = input_q - input_zero_point
magnitude = abs(centered)
output_q = saturate(round(magnitude × input_scale / output_scale) + output_zero_point)
```

例如输入零点为 −3、输入 scale 为 0.25，输入编码 −11 表示 −2。输出 scale 为 0.5、输出零点为 1 时，ABS 的结果 2 对应输出编码 5。源码使用定点运算近似缩放，不在循环里逐元素做浮点除法。

## 4. Delegate 是什么，何时创建

![图 4-2：Delegate：显式应用与惰性应用的时机](figures/11-delegate-control.png)

*图 4-2　Delegate：显式应用与惰性应用的时机。两侧是可用的入口及触发时机，不是互斥的后端模式。注册工厂时还没有改图；首次触发的确切位置和错误处理以正文对应版本为准。*

`TfLiteDelegate` 也是 C 风格结构体。它的 `Prepare(context, delegate)` 不是某个 ABS 节点的 Prepare，而是检查图、选择可接管节点、请求改写执行计划。它可用 `CopyFromBufferHandle` 把后端存储同步回可由 CPU 读的张量，用 `CopyToBufferHandle` 做反方向，用 `FreeBufferHandle` 清理该句柄及后端为它管理的资源；释放细节由 Delegate 实现定义，不能理解成永远不释放资源。没有相关回调往往意味着该方向的句柄操作不受支持，不意味着后端完全没有私有分配。

| Flag | 读源码时看什么 |
| --- | --- |
| `AllowDynamicTensors` | 是否能处理动态张量及相应的输出形状变化。 |
| `RequirePropagatedShapes` | 需要先把形状沿原执行计划传播，再准备对应的委托 kernel；它不是对所有 Delegate 无条件启用。 |
| `PerOperatorProfiling` | 委托可以报更细粒度的内部算子事件，而不是只看整个分区。 |
| `HintFullyDelegatedToSingleDelegate` | 某些路径可省去为原 CPU 图做的准备/分配。不是「每个后端天然可以接管全图」；原稿分支显式核对 `IsFullyDelegated`，不能满足则返回错误。该函数遍历当前执行计划检查节点是否带 delegate。 |

创建期 `InterpreterBuilder::ApplyDelegates` 尝试显式传入的 Delegate，相关版本也处理自动获取的 Flex；不要把这段和 Resolver 默认工厂混在一起。Lazy Delegate 在 Builder 阶段保存 creator，通常由 Interpreter 首次 `AllocateTensors` 一类准备入口实际尝试（不是必须等到第一次 `Invoke`）。原稿的筛选条件是在启用 QS8 / QU8 构建时直接允许创建 provider，否则看是否有 FP32 张量；保存工厂仍不意味着创建成功，创建成功也不代表有节点可接管。

### 4.1 为什么 XNNPACK 工厂可能返回空

XNNPACK 是 CPU 推理内核库，XNNPACK Delegate 是其与 Interpreter 之间的适配层。原稿的 `MaybeCreateXNNPACKDelegate` 在未集成的构建里可能返回空；集成构建会合并默认 options，再按该版本的 QS8（有符号量化 8 位）/ QU8（无符号量化 8 位）支持配置构造对象。用带定制删除器的 `TfLiteDelegatePtr` 管理对象生命周期。

- **线程池**：区分外部借用和自行拥有。借用运行时 CPU 后端的线程池不应由 Delegate 析构，独有池按配置创建与销毁。Emscripten 无 pthread 的构建不走相同的多线程代码，线程数并不是永远取 Interpreter 或 options 的同一个值。
- **权重缓存**：权重重排（packing）把常量转换为更适合内核读取的布局；开启缓存可在其生命周期内复用重排结果。可借用用户传入的 cache/provider，也可按配置创建自有 provider。借用者要保证所有者寿命；文件路径、内存锁定、映射方式均看具体选项和平台。`LockMemory` 若实际启用，目标是尝试让相关页保持驻留，不能保证所有缺页都消失。

## 5. 改写从 Interpreter 下沉到 Subgraph

```text
Interpreter::ModifyGraphWithDelegateImpl
  → 每个适用的 Subgraph::ModifyGraphWithDelegate
      → 保存原执行计划 / 检查状态、动态形状与适用性
      → 切换为 DelegateContext
      → TfLiteDelegatePrepareInternal（opaque 或普通回调）
          → Delegate 选节点并调用 ReplaceNodeSubsetsWithDelegateKernels
      → 切回 KernelContext
      → 检查覆盖范围、更新形状/内存及状态
```

原稿的 Interpreter 级实现会先按下标建立 Subgraph 信息并校验，而后逐个处理适用子图；有失败清理和恢复路径。Subgraph 在准备原图时按 `execution_plan_` 的索引调用 `PrepareOpsStartingAt`，跟踪是否出现动态输出及准备进度。开始一轮与从中间继续不同，不能把每次的动态张量检测都当成独立的全图检查。Prepare 可能添加临时张量，所以会预留向量容量，避免底层存储意外移动；也不要自行长期持有会失效的元素指针。

不能处理动态张量且没有全委托提示时，原计划通常需要先准备，用于识别动态形状。全委托的优化路径允许 Delegate 自行完成准备；否则不能简单跳过。原来是 Immutable 时，有些路径暂置 Uninvokable 以执行再次准备；失败时按错误阶段和状态恢复原计划或移除委托，不能假定任何报错都等于「已经安全 CPU 回退」。

## 6. XNNPACK 如何决定接管哪些节点

`DelegatePrepare` 先找候选，再向 Subgraph 发出改写请求。判断不是只看 opcode，还依赖参数、张量类型、形状、常量属性及实际 kernel 覆盖。在原稿的实现中，常规节点与 MoE 节点分组提交。这里的 MoE 是特定自定义算子，不代表 XNNPACK 会识别任意写成通用子图的专家网络：路由输入决定选哪些专家，权重/scale 的数量和类型都属于匹配条件。原稿分支的匹配包括指定 custom 名称、可解析属性（专家数和隐藏/模型维度等）、单输出、浮点权重 7 输入或量化权重 10 输入、FP32 激活/权重系数/输出与 INT32 路由索引；专家权重静态且形状、类型、元素数正确，量化 scale 也须静态且符合要求。这些是该版本检查项，不是跨版本的 MoE 接口规范。

### 6.1 准静态张量：为什么一条计算边可以提前处理

「准静态」输出虽然有生产节点，但从常量输入出发，可以在准备期得到。例如 INT8 权重转 FP32，或者把常量稀疏数据展开成稠密布局。实现首先建立 tensor 到消费节点的映射，然后记录准静态张量、其生产节点以及委托端实际需要展开的集合。

- DEQUANTIZE：原稿检查单输入/输出、非稀疏的 FP16 或带有效量化参数的 INT8 输入、FP32 输出、只读常量或之前已识别的准静态来源。部分 FP16 权重在所有消费者都满足特定 FC/Conv/DepthwiseConv 权重槽位约束时可直用；若它是激活输入或被其他算子消费，就不能据此省略这步。
- DENSIFY：以符合条件的只读稀疏常量为源，单输入/输出，类型一致；原稿列出 FP16、INT8、FP32。还要看该分支其余元数据检查。
- `VisitNode` 在没传 XNNPACK 子图时通常只检查，真正构建时才定义对应后端节点。若某节点留在 CPU，需要重新检查原转换的其余消费者，不能把所有准静态转换一律从执行计划删除。源码按依赖执行必要展开，从原始静态数据或已展开缓存取输入，做反量化 / 稀疏展开并保存，再更新权重缓存所需的映射，将可跳过的转换并入委托候选。测试构建里强制全委托的路径用于暴露不支持项，不能作为默认行为。

用一个小图看这个限制：`常量 W → DEQUANTIZE → Wf → Conv`，若只有被委托的 Conv 需要 `Wf`，有机会在 Prepare 完成转换；若还有一个 CPU 节点读取 `Wf`，它必须继续得到合法数据。一个张量是不是常量，和能不能删除它的生产节点，是两个判断。

## 7. 从支持列表到新执行计划

![图 4-3：Delegate：先选择受支持的节点，再按图分区改写](figures/08-delegate-partition.png)

*图 4-3　Delegate：先选择受支持的节点，再按图分区改写。下面是示例，不是“所有支持节点必定合并”。原始图里 A、B、D 可委托，C 留给原内核。图分区允许时，A、B 合为一个委托分区，D 留作另一个；一个 Delegate 可负责多个分区，一个分区可包多个原节点。边界 Tensor 继续连接相邻分区或原生节点。*

`ReplaceNodeSubsetsWithDelegateKernels` 先建立 DELEGATE 的 registration；opaque 路径还要维护外部算子对象的生命周期。随后通过 `PartitionGraph` 按依赖划分分区：区分委托和非委托节点，依赖就绪才能被调度；可合并时尽可能聚合，不要求原计划里的下标天然相邻，也不能破坏数据依赖与必要的控制顺序。如果禁止节点融合，则使用更保守的分法。分区阶段本身仍未改写计划。

```text
改写前（示意）：[节点 0 CPU, 节点 1 委托, 节点 2 委托, 节点 3 CPU]
改写后        ：[节点 0 CPU, 委托节点 K(1,2), 节点 3 CPU]
```

非委托分区把原节点下标写回新执行计划；委托分区构造包含原节点列表、边界输入/输出以及 delegate 引用的 params，将其作为新节点初始化数据。新节点和对应 kernel registration 保存进 `nodes_and_registration_` 并进入执行计划。原节点未必从存储中消失，**执行计划里的下标不再指向它们才是不逐个执行的原因。** 真正 Invoke 时走委托节点的回调，委托内部根据分区搭建并运行 XNNPACK 图，与原始 TFLite Subgraph 不是同一个对象。

### 7.1 分配结束后如何理解三种状态

- **Uninvokable**：需要准备或重新分配，不能执行。
- **Invokable**：当前可执行，允许符合规则的后续调整。
- **InvokableAndImmutable**：可执行但图受到不可变约束，典型情形是应用了不支持动态张量的 Delegate。

原稿分支的尾部处理区分三类：当前委托不支持动态张量时通常确保内存分配并标记 Immutable（满足全委托提示可走优化分支）；当前委托支持动态、但旧图已 Immutable 时，重新 Prepare，若引入动态张量则恢复计划并报错；原图已 Invokable 且可动态时，刷新必要分配以保持一致。新图还未准备的普通路径可留待分配阶段。调试时按「Resolver 查找 → 普通内核 Prepare / 形状 → Delegate 选点 → 分区与边界 → 内存分配 → Invoke」顺序定位，不要把任何一个阶段的通过当成后续已经成功。

## 源码导航

- [OpResolver 接口](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/api/op_resolver.h) · [MutableOpResolver](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/mutable_op_resolver.cc) · [内建注册](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/kernels/register.cc)
- [C 数据结构与 Delegate flags](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/c/common.h) · [Elementwise 示例](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/kernels/elementwise.cc)
- [Interpreter](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/interpreter.cc) · [Subgraph](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/subgraph.cc) · [图分区](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/graph_info.cc) · [XNNPACK Delegate](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/delegates/xnnpack/xnnpack_delegate.cc)
