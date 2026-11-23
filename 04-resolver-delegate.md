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

`MutableOpResolver` 在抽象接口之上维护 `(code, version) → registration` 和 `(name, version) → registration`。查找时先看本地表，再按顺序查链里的 Resolver。


- `AddBuiltin` 和 `AddCustom` 分别按枚举和名称注册。在所指定的版本或版本区间内复制并补齐注册信息；同键覆盖原条目。复制的是注册记录，函数指针并不会复制一份函数实现。
- `AddAll` 合入另一个 Mutable Resolver 的映射，并在原稿实现中带入其链式引用；`ChainOpResolver` 本身保存引用，未命中时才调用对方，不是复制条目。链中的对象必须活到查找结束。
- `may_directly_contain_user_defined_ops_` 是保守标记。查询还会考虑链上的 Resolver，不能拿它枚举模型使用的自定义节点。
- `BuiltinOpResolver` 在 Mutable Resolver 上注册标准内建实现，也可能提供默认 Delegate 工厂。它是常见选择，不是唯一选择；应用可以显式提供更小的注册集合，或选择不附自动默认委托的 Resolver。

### 从 `FindOp()` 看本地表与链式查找

实际源码：[mutable_op_resolver.cc:28–56](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/mutable_op_resolver.cc#L28-L56)。

```text
查找 builtin 时以 (枚举, 版本) 为键，查找 custom 时以 (名称, 版本) 为键
本地表命中 → 返回表项
没命中 → 依注册顺序询问链上的 resolver
直到找到实现；全未命中才返回空
```

1. Builtin 查 `(枚举, version)`，custom 查 `(名称, version)`。同一个算子的不同版本可以落到不同记录，不能只按名字找。
2. 本地命中直接返回；未命中才按顺序问链上的 resolver，直到第一个有效指针。`nullptr` 表示没有实现，不是创建了一个空 Node。
3. 这一层只回答“由谁实现”。Builder 下一步才把注册项和一个具体节点配在一起。

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

源码入口：`tflite/kernels/elementwise.cc` 中的 `Register_ABS`，再追 `ElementWiseQuantizedInit` / `ElementWiseQuantizedFree`、`PrepareAbs` / `GenericPrepare` 以及 `AbsEval`。把生成回调的宏展开来看，仍是普通函数指针调用。

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

### 3.1 ResizeTensor：为什么没有马上分配所有输出

ABS 的 Prepare 将输入形状交给 Context 的 Resize 回调；回调通过 `impl_` 找到 Subgraph，不由 ABS 内核决定全图的内存布局。看 [ResizeTensor 回调与快路径](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1868-L1911) 和 [ResizeTensorImpl](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2150-L2199) 时，可以按以下顺序读：

1. 快路径先要求已有数据指针且 shape 完全不变；普通定长类型还须理论字节数等于当前 `bytes`。String、Resource、Variant 的大小不由这段公式判断。命中时只在需要时换掉 dims 对象，不重分配数据。同样是 12 个元素的 `[3,4]` 与 `[2,6]` 不满足 shape 相同这一条件。
2. 进入通用实现后，先看 allocation 类型能不能调整，再视类型计算所需字节。普通堆张量可以当场调存储；String、Resource、Variant 不走普通的元素数乘类型大小路径。
3. 更新维度；若属于 arena 管理的读写类型，则清空旧数据指针、留待内存规划。对固定只读权重，Resize 返回错误而不是直接改原文件。

### 把 ABS 的 `Prepare` 和 `Invoke` 分开看

实际源码：[elementwise.cc:542–550](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/elementwise.cc#L542-L550)、[elementwise.cc:212–214](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/elementwise.cc#L212-L214)、[elementwise.cc:309–330](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/elementwise.cc#L309-L330)。

```text
ABS 注册项 → Init / Free / PrepareAbs / AbsEval
PrepareAbs:
    检查数据类型和量化参数；令输出形状与输入相同
AbsEval:
    访问当前 Node 的第 0 个输入
    FLOAT32 / INT32 → 普通计算
    INT8 → 量化计算
    INT16 → 按有无量化参数选计算路径
    其他类型 → 返回错误
```

1. `Register_ABS()` 把 Init、Free、Prepare、Invoke 放到同一注册记录；Init/Free 管节点私有状态，不是每个输出元素调用一次。
2. `PrepareAbs` 由宏展开进 `GenericPrepare()`，完成类型和量化检查、需要时准备定点参数，最后通过 Context 让输出 shape 对齐输入。`ResizeTensor()` 调整的是运行时描述和相关内存状态，不是在这里求绝对值。
3. `AbsEval()` 的 `GetInput(..., 0)` 是**本节点的第 0 个操作数**，和主图第 0 个对外输入不是一回事。Invoke 再根据类型走浮点、普通整数或量化路径。

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

### 默认 XNNPACK 注册的是工厂

实际源码：[register.cc:414–426](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/kernels/register.cc#L414-L426)、[interpreter.cc:333–339](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter.cc#L333-L339)、[interpreter.cc:344–352](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter.cc#L344-L352)。

```text
BuiltinOpResolver 创建时：
    注册 builtin，再保存一个尝试创建 XNNPACK 的工厂
    不带默认委托的构造路径只注册 builtin
触发 lazy 应用时：
    如果已全委托或没有工厂，直接结束
    取走工厂列表，避免重复尝试
    逐项调用工厂；有对象才尝试改图
```

1. 普通 `BuiltinOpResolver` 先装入 builtin 注册项，再追加一个 XNNPACK creator；不带默认委托的构造仍会注册 builtin，但没有这一步。
2. 运行时才遍历 factory。没有 creator 或已完全委托就不做这一步；执行时先把列表 swap 出来，避免多次重复应用。
3. 工厂返回空表示此项未创建；返回对象后才尝试 `ModifyGraphWithDelegateImpl()`。因此只看到 Resolver 构造成功不能断言 XNNPACK 生效。

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

### 5.1 从选择 Prepare 入口到实际调用

[委托入口与动态能力判断](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2588-L2651)里，**应用 Delegate 之前也可能调用普通算子的 Prepare**。已经处于 `kStateInvokableAndImmutable` 的子图，先改回未就绪，给后续重新准备留出空间；否则，若新委托既不支持动态张量，又没有「单一委托接管全图」提示，就从位置 0 准备当前执行计划，提前探测动态输出。如果探测到动态张量，源码先让内存分配回到一致状态，再返回 `kTfLiteApplicationError`。这个探测不是运行推理，也不是委托自己的图编译。

#### 5.1.1 PrepareOpsStartingAt：循环一项一项看

[固定版本源码：逐节点准备](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1584-L1625)

```text
输入：开始位置 start、要读的计划 plan、用于写回的位置 last
仅当 start == 0：
    扫一遍子图输出，初始化动态标记（包括直接透传的输出）
从 start 开始逐项处理：
    plan[位置] → 节点编号 → (TfLiteNode, TfLiteRegistration)
    为可能新增的 Tensor 槽预留容量
    如启用编译期追踪，记录这次准备事件
    result = 按 registration 调用 OpPrepare
    若 result 既非 OK 也非 OutputShapeNotKnown：
        上报这一节点的失败并结束，不推进 last
    last = 当前计划位置
    若该节点有动态输出，或 result == OutputShapeNotKnown：
        标记子图存在动态张量；以 OK 结束这一轮
走完整个可准备区间后，以 OK 结束
```

1. **先扫边界，再扫节点。** 子图有时原样透传某个输入，没有算子产生该输出，单查节点的 `outputs` 会漏掉它。只有从 0 开始才重新扫描子图边界；从后续位置续跑不重置这段状态。
2. **区分三个编号。** 计划位置用于推进或恢复，节点编号用于找 `nodes_and_registration_`，Tensor 编号用于找张量表。比如计划为 `[7, 2, 9]`，从位置 1 开始准备的是节点 2，写回的进度仍是位置 1，不是节点 2。
3. **扩容不是分配数据。** [EnsureTensorsVectorCapacity](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2561-L2569)在回调之前按「当前大小加预留量」检查容量，需要时按所需容量与扩容比例的较大值 `reserve`，随后刷新 `context_.tensors`。它不增加张量数量、不为推理数据分配 arena，也不保证此前取到的地址不会失效。
4. **何时暂停。** 动态的是节点**输出**才会在这里截断准备；动态临时张量不影响其他节点的输出尺寸，因此不单独构成暂停条件。内核返回 `kTfLiteOutputShapeNotKnown` 即便没有在输出扫描里找到明确的动态 Tensor，也会暂停；不要把它和真正的 Prepare 失败混为一谈。暂停的节点已记为本轮最后准备位置，但并未执行。
5. **记录的对象不同。** `last_execution_plan_index_prepared` 是位置，`dynamic_tensor_index_` 是动态 Tensor 的编号。后者在张量扫描中更新，不能当成恢复游标；仅返回「形状未知」的场合也不一定提供一个新的编号。

#### 5.1.2 OpPrepare：选择哪一个回调

[固定版本源码：普通与不透明回调](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1419-L1491)

| 按顺序判断的情形 | 实际路径 |
| --- | --- |
| 有 `registration_external`，且 wrapper 记录了原节点编号 | 通过节点编号取原 `TfLiteRegistration`，调用原实现的 Prepare。 |
| 外部对象提供 `prepare_with_data` | 使用外部对象自己的 `user_data`；传入不透明 Context 和 Node。它不是 `TfLiteNode::user_data`。 |
| 外部对象只提供 `prepare` | 调用无 data 参数的不透明回调。 |
| 以上都未接管 | 回到普通 `registration.prepare`。 |
| 最终没有回调 | 已解析的算子可以成功跳过；未解析的 Custom/Flex 占位项返回 `kTfLiteUnresolvedOps`。 |

原笔记里「外部包装可以复用原生算子」的关键是第一行：有不透明对象不代表重写了一遍 ABS，也不代表每次都通过 ext 对象自身的回调。排查先看是不是 wrapper，再查两个回调和 user data 来源。

#### 5.1.3 从 AllocateTensors 暂停，再从 Invoke 继续

[准备与内存规划](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1627-L1700) · [Invoke 续跑](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1743-L1757) · [动态输出重置](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1828-L1856)

- **准备原计划与当前计划。** 若有已应用的委托要求先传播形状，并且保存了委托前的计划，例程先用原计划准备，再准备实际执行的计划。不要只根据字段名猜传参：该快照原计划分支写回 `next_original_execution_plan_index_to_prepare_`，实际传给调用的起点却是 `next_execution_plan_index_to_prepare_`。
- **已经准备到哪里，只分配到哪里。** 当前计划的 Prepare 游标更新为 `last + 1`。需要时建立内存规划器并规划，再对本轮已准备区间执行分配；核对自定义分配后推进分配游标。Prepare 与分配是两步，不是 `OpPrepare` 调完就等于所有数据已分配。
- **执行时继续。** 例如节点 7 的输出大小由输入内容决定，后续节点 2 依赖它。初始准备可先停在节点 7；Invoke 执行节点 7 后，走到计划位置 1 才继续准备节点 2。如果本轮执行让动态输出改变大小，准备起点重设为当前项之后；已超前规划的下游内存也会被重置。不会从计划位置 0 无条件重做整图。

### 5.2 Delegate Prepare 前后为何要切换 Context

实际源码：[subgraph.cc:2647–2662](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2647-L2662)。

```text
首次应用委托时保留原执行计划
Context 临时开放委托回调
调用 Delegate 的 Prepare，记录结果
恢复普通内核 Context
按 Prepare 结果继续或恢复原计划
```

1. 先保存首次改写前的 `execution_plan_`，为失败恢复保留原序列；前面的分支还会根据已应用委托、动态 Tensor 能力等决定是否先准备普通算子。
2. `SwitchToDelegateContext()` 临时提供读取计划、查询节点、替换分区等接口，委托通过这些回调请求改图，不直接拥有 TFLite 子图。
3. Prepare 返回后切回普通 kernel Context，再处理错误；不要把“某个节点不支持”与“整个 Delegate Prepare 失败”混为一谈。

## 6. XNNPACK 如何决定接管哪些节点

`DelegatePrepare` 先找候选，再向 Subgraph 发出改写请求。判断不是只看 opcode，还依赖参数、张量类型、形状、常量属性及实际 kernel 覆盖。在原稿的实现中，常规节点与 MoE 节点分组提交。这里的 MoE 是特定自定义算子，不代表 XNNPACK 会识别任意写成通用子图的专家网络：路由输入决定选哪些专家，权重/scale 的数量和类型都属于匹配条件。原稿分支的匹配包括指定 custom 名称、可解析属性（专家数和隐藏/模型维度等）、单输出、浮点权重 7 输入或量化权重 10 输入、FP32 激活/权重系数/输出与 INT32 路由索引；专家权重静态且形状、类型、元素数正确，量化 scale 也须静态且符合要求。这些是该版本检查项，不是跨版本的 MoE 接口规范。

#### XNNPACK 先枚举节点，再做能力检查

实际源码：[xnnpack_delegate.cc:7316–7336](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7316-L7336)、[xnnpack_delegate.cc:7517–7541](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7517-L7541)、[xnnpack_delegate.cc:7872–7915](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7872-L7915)。

```text
读取当前执行计划和对应节点
先把每个输入 Tensor 的消费者登记下来
逐节点做 VisitNode 检查：
    能接：登记候选和所需的准静态输入
    不能接：保留给普通内核，并保留它依赖的原转换
DelegatePrepare 拿到候选后：
    普通候选集合非空时，请 Runtime 按依赖替换
    MoE 候选在该快照逐个提交为单节点分区
```

1. `execution_plan->data[i]` 才是节点编号；`i` 只是执行顺序中的位置。第一遍顺便建立 `Tensor 编号 → 消费者节点列表`，供后续判断静态转换是否仍被普通节点使用。
2. `VisitNode()` 此时收到空的 XNNPACK subgraph，只检查算子、Tensor、参数能否支持。失败的节点不加入委托列表，普通内核继续负责；遇到仍被它消费的准静态结果，还要保留相应转换节点。
3. `DelegatePrepare()` 拿到普通节点候选后通过 Context 提交给 TFLite 分区；是返回空指针才表示这一阶段错误，空列表并不等于错误。这个快照另有 MoE 单节点处理，不能把所有节点都看成一次普通替换。

### 6.1 准静态张量：为什么一条计算边可以提前处理

「准静态」输出虽然有生产节点，但从常量输入出发，可以在准备期得到。例如 INT8 权重转 FP32，或者把常量稀疏数据展开成稠密布局。实现首先建立 tensor 到消费节点的映射，然后记录准静态张量、其生产节点以及委托端实际需要展开的集合。

- DEQUANTIZE：原稿检查单输入/输出、非稀疏的 FP16 或带有效量化参数的 INT8 输入、FP32 输出、只读常量或之前已识别的准静态来源。部分 FP16 权重在所有消费者都满足特定 FC/Conv/DepthwiseConv 权重槽位约束时可直用；若它是激活输入或被其他算子消费，就不能据此省略这步。
- DENSIFY：以符合条件的只读稀疏常量为源，单输入/输出，类型一致；原稿列出 FP16、INT8、FP32。还要看该分支其余元数据检查。
- 测试构建里还有用于暴露不支持算子的强制委托路径，不代表普通构建会禁用回退。

用一个小图看这个限制：`常量 W → DEQUANTIZE → Wf → Conv`，若只有被委托的 Conv 需要 `Wf`，有机会在 Prepare 完成转换；若还有一个 CPU 节点读取 `Wf`，它必须继续得到合法数据。一个张量是不是常量，和能不能删除它的生产节点，是两个判断。

准静态张量不是扫到就能删掉生产者。实际展开时，先按生产节点编号安排预展开，输入优先读已经展开的结果，否则读模型常量；再做反量化或稀疏转稠密，把结果放进 `static_unpacked_data`。只要还有没委托的分支要读这个结果，原转换节点仍要留给 Interpreter。比如 `INT8 权重 → DEQUANTIZE → FP32` 同时被委托和未委托的节点读取时，不能因为 XNNPACK 能预展开，就把原转换从执行计划中拿掉。[消费者与候选筛选](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7517-L7542) · [预展开](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7559-L7655)。

#### 准静态权重、线程池和缓存不是同一个层次

- **选择可提前做的转换。** `DEQUANTIZE` 在该路径要求一进一出、有效类型和已知的输入字节；部分 FP16 权重如果只被支持的权重槽消费，可直接使用，不是先造一份 FP32 再跳过。稀疏常量的 `DENSIFY` 还要看输入是否带 sparsity、输出是否与输入同类型。
- **展开并记账。** 用生产者索引安排展开，找到节点并核对一进一出，从已展开缓存或静态常量读取输入，按输出大小建立缓冲区；反量化或转稠密之后，由 `static_unpacked_data` 按 Tensor 编号保存。再处理权重缓存映射和可省略的原转换节点，最后排序候选提交分区。这里的“提前”是委托准备期，不是模型转换工具预先改写了文件。
- **执行资源的所有权另算。** XNNPACK 可借用 CPU Backend 的线程池，也可拥有自己的线程池；析构只释放自己拥有的。权重 cache 缓解的是 packing/重排成本，缓存 provider 负责查找和管理这些条目；它既不是 Thread Pool，也不是保存 Tensor 索引的表。外部 provider 与由 Delegate 创建的 provider 不能按同一套所有权处理。

[准静态识别与检查](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7413-L7542) · [展开与候选提交前的整理](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7559-L7800)

## 7. 从支持列表到新执行计划

![图 4-3：Delegate：先选择受支持的节点，再按图分区改写](figures/08-delegate-partition.png)

*图 4-3　Delegate：先选择受支持的节点，再按图分区改写。下面是示例，不是“所有支持节点必定合并”。原始图里 A、B、D 可委托，C 留给原内核。图分区允许时，A、B 合为一个委托分区，D 留作另一个；一个 Delegate 可负责多个分区，一个分区可包多个原节点。边界 Tensor 继续连接相邻分区或原生节点。*

`ReplaceNodeSubsetsWithDelegateKernels` 先建立 DELEGATE 的 registration；opaque 路径还要维护外部算子对象的生命周期。随后通过 `PartitionGraph` 按依赖划分分区：区分委托和非委托节点，依赖就绪才能被调度；可合并时尽可能聚合，不要求原计划里的下标天然相邻，也不能破坏数据依赖与必要的控制顺序。如果禁止节点融合，则使用更保守的分法。分区阶段本身仍未改写计划。

```text
改写前（示意）：[节点 0 CPU, 节点 1 委托, 节点 2 委托, 节点 3 CPU]
改写后        ：[节点 0 CPU, 委托节点 K(1,2), 节点 3 CPU]
```

非委托分区把原节点下标写回新执行计划；委托分区构造包含原节点列表、边界输入/输出以及 delegate 引用的 params，将其作为新节点初始化数据。新节点和对应 kernel registration 保存进 `nodes_and_registration_` 并进入执行计划。原节点未必从存储中消失，**执行计划里的下标不再指向它们才是不逐个执行的原因。** 真正 Invoke 时走委托节点的回调，委托内部根据分区搭建并运行 XNNPACK 图，与原始 TFLite Subgraph 不是同一个对象。

#### 拿到候选以后，真正改了哪张表

实际源码：[subgraph.cc:564–574](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L564-L574)、[subgraph.cc:641–653](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L641-L653)、[subgraph.cc:667–681](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L667-L681)、[subgraph.cc:1235–1241](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1235-L1241)。

```text
PartitionGraph 根据依赖关系与聚类配置生成分区
清空当前执行计划
依分区顺序：
    普通分区 → 将原节点编号放回计划
    委托分区 → 构造带边界张量的代理节点，将它加入计划
    委托分区的边界输出与代理节点记录所属 Delegate
```

1. `PartitionGraph()` 根据依赖、控制边和聚类配置算出分区。相邻不保证同一分区，非相邻也不代表绝不可能合并；边界上只列该分区对外需要的 Tensor。
2. 清空旧执行计划后，普通分区把原节点编号放回计划；由委托负责的分区用 `AddNodeWithParameters()` 追加代理节点。该函数在结束时将新节点编号放进计划。
3. 代理节点持有分区输入、输出和委托初始化参数，边界输出 Tensor 也会记录所属委托。原节点记录不等于全部删除；推理走的是改写后的计划，不是简单按节点表下标遍历。

把这段分成两次决策更容易看清：XNNPACK 决定**哪些节点能接**，TFLite 决定**哪些候选可以放在一个分区**。例如两个候选节点之间若隔着必须由普通内核执行、并且后一个候选依赖其输出的节点，就不能把两个候选合成一个代理节点；否则代理节点还没跑完，普通内核就等不到它所需的前半段结果。这个例子也解释了为什么只按原执行计划上连续的编号分组不够。

#### 分区边界怎样从依赖关系算出来

[PartitionGraph 的 epoch 逻辑](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/graph_info.cc#L90-L259)

1. 起始时把没有节点生产的 Tensor 看作可用，例如外部输入和常量；有生产者的先标成未就绪。不要按 Tensor 编号大小判断依赖。
2. 逐个考虑尚未安排的节点：输入的生产者以及控制前驱必须可用，候选与当前分区还必须是同一类型（委托或普通）。聚类/融合配置限制能否把不相邻节点吸收进来。
3. 消费者从更早分区或图外拿到某个 Tensor 时，它成为当前分区的输入；若来自更早分区，它也成为生产分区的输出。图对外输出还要另行标到生产分区，即使后面没有节点消费。
4. 先得到一组分区描述，才创建 Delegate 节点和刷新 `execution_plan_`。原节点及注册表仍用于保留依赖、恢复原计划等，不是划完分区就把整段对象删掉。

### 7.1 分配结束后如何理解三种状态

- **Uninvokable**：需要准备或重新分配，不能执行。
- **Invokable**：当前可执行，允许符合规则的后续调整。
- **InvokableAndImmutable**：可执行但图受到不可变约束，典型情形是应用了不支持动态张量的 Delegate。

原稿分支的尾部处理区分三类：当前委托不支持动态张量时通常确保内存分配并标记 Immutable（满足全委托提示可走优化分支）；当前委托支持动态、但旧图已 Immutable 时，重新 Prepare，若引入动态张量则恢复计划并报错；原图已 Invokable 且可动态时，刷新必要分配以保持一致。新图还未准备的普通路径可留待分配阶段。调试时按「Resolver 查找 → 普通内核 Prepare / 形状 → Delegate 选点 → 分区与边界 → 内存分配 → Invoke」顺序定位，不要把任何一个阶段的通过当成后续已经成功。

### 7.2 看失败发生在什么阶段，而不是只看“没用上委托”

| 观察到的现象 | 从哪里查 |
| --- | --- |
| creator 返回空 | 还没创建对象。看构建是否接入 XNNPACK，以及 resolver 工厂配置。 |
| 某个候选未通过 `VisitNode` | 正常的能力筛选；其他可用分区仍可委托，该节点仍可由原内核执行。 |
| 静态委托在预检查发现动态输出 | 在委托 Prepare 之前拒绝这次应用，返回 `kTfLiteApplicationError`。 |
| Delegate Prepare 自己失败 | 先切回普通 Context，再按该路径恢复原执行计划并返回委托错误；不能说所有错误都无条件静默回退。 |
| 声明整图委托提示，但仍有普通节点 | 应用时会验证 `IsFullyDelegated()`；不满足时报错，不把提示当成功结果。 |

查最后三项时对照 [预检查、改图、状态恢复](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2588-L2711)。把断点放在入口、`PrepareOpsStartingAt`、`TfLiteDelegatePrepareInternal` 和 `ReplaceNodeSubsetsWithDelegateKernels`，就能判断失败是在普通内核准备、后端选点、分区替换，还是分配这一段。

## 源码导航

代码对照版本：LiteRT [`e3c11f0`](https://github.com/google-ai-edge/LiteRT/tree/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62)；流程块是按固定版本整理的阅读稿；可以点对应行号看完整实现。

- [OpResolver 接口](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/api/op_resolver.h) · [MutableOpResolver](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/mutable_op_resolver.cc) · [内建注册](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/kernels/register.cc)
- [C 数据结构与 Delegate flags](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/c/common.h) · [Elementwise 示例](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/kernels/elementwise.cc)
- [Interpreter](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/interpreter.cc) · [Subgraph](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/core/subgraph.cc) · [图分区](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/graph_info.cc) · [XNNPACK Delegate](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/delegates/xnnpack/xnnpack_delegate.cc)
