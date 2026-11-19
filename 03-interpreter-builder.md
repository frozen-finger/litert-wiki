# LiteRT 源码笔记（三）：InterpreterBuilder 如何把模型变成运行时执行图

上一篇：[从文件到 LiteRT IR](02-load-model-ir.md)｜下一篇：[算子注册与 Delegate](04-resolver-delegate.md)

这一篇关心一件事：文件里只有算子编号、张量索引和一些参数，Interpreter 怎么知道该调用哪个函数、去哪里读权重、把结果写到哪里？

先区分两条路径：上一篇的 `LiteRtModelT` 是 LiteRT 用于表示和处理模型的 IR；本篇的 `InterpreterBuilder` 读取的是 **TFLite FlatBuffer 视图**，不是直接把 `LiteRtModelT` 强转成 `Interpreter`。在 LiteRT 的 CompiledModel 实现里，可以到 `compiled_model.cc` 查找 `tflite::InterpreterBuilder` 的实际调用点；上游版本和编译开关不同，转换步骤、构造参数或支持的特性也可能不同。

## 1. 从 Builder 到 Interpreter

```cpp
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder builder(model, resolver);
if (builder(&interpreter) != kTfLiteOk || !interpreter) {
  // 处理构建失败；此处不进入推理。
}
```

传给 `operator()` 的是一个用于接收结果的 `unique_ptr`，不是已经构造好的 Interpreter。Interpreter 并非不能直接构造；从模型建图时，把校验、子图导入和算子绑定交给 Builder。构建过程可概括为：

```text
校验输出、模型指针及 Model.version()
  → 为 Model.operator_codes 建立本地注册表
  → 创建 Interpreter 及各 Subgraph
  → 导入图的 I/O，解析 Tensor、Node 和变量
  → 导入 SignatureDef、Metadata
  → 应用显式 Delegate，登记满足条件的延迟工厂
  → 移交 Interpreter
```

顺序示意中，登记延迟工厂和应用显式 Delegate 是两个独立动作；原稿对应路径先登记工厂，再调用 `ApplyDelegates`。`Model.version()` 与 `TFLITE_SCHEMA_VERSION` 比较，是 TFLite 模型 schema 的兼容性检查，并非要求生成文件的 FlatBuffers 库版本等于某个版本。输出指针存在时，Builder 先 `reset`，临时 Interpreter 构建完成才 `move` 给调用者。检查通过也不代表可直接 `Invoke()`：通常还要填好输入形状并调用 `AllocateTensors()`。

常用的 `FlatBufferModel` 构造重载从模型对象取 `GetModel()`、错误报告器、元数据和 `allocation()`，传入的 `OpResolver` 负责查算子；另外也有直接接收 `const tflite::Model*` 并显式传入 ErrorReporter、options、Allocation 的重载。原始模型字节可被运行时张量或自定义参数借用；调用者应遵守该 API 的模型、Resolver 和外部对象生命周期约定，不可在构建成功后立刻销毁这些依赖。

## 2. 把 opcode 配对到 registration

![图 3-1：Builder：从文件算子编号到每节点运行时记录](figures/05-builder-mapping.png)

*图 3-1　Builder：从文件算子编号到每节点运行时记录。多条 Operator 可用同一个 opcode_index，也可共享同类注册信息。子图把所选 Registration 记录下来，并为每个节点保存独立的输入输出索引和状态；这不意味着多个节点共用一个 user_data。*

模型里一张全局 `operator_codes` 表描述算子类型，子图里的每个 `Operator.opcode_index` 用表索引指向它。一个 `ADD` 出现十次，十个节点可以共享同一个 opcode 表项，却各自有 I/O 和参数。Builder 调用 `BuildLocalIndexToRegistrationMapping()`，给每个表项缓存一次解析结果：

| 文件中的信息 | Resolver 按什么查 | 查不到时 |
|---|---|---|
| 内建 op | `FindOp(builtin_code, version)` | 该版本未注册，构建失败；只支持同名的其他版本不够 |
| 自定义 op | `FindOp(custom_code, version)` | 名称存在但未注册时，可放入未解析占位项，留待 Delegate 或 Prepare；没有名称仍是错误 |

`GetBuiltinCode` 是读取 schema 里内建编号的兼容入口，不应假定历史模型只使用新版编号字段。`flatbuffer_op_index_to_registration_` 是按原表顺序保存的 `const TfLiteRegistration*` 数组；原始 registration 由 Resolver 提供，Builder 自己管理的 `unresolved_custom_ops_` 则存占位记录。实现先为可能的自定义项预留空间，再把指针放进索引表，避免向量扩容使旧地址失效。模型里未命名的 CUSTOM、超出范围的内建 opcode、未注册的 built-in 版本都不该被当作“等待 Delegate”的普通情况。

`TfLiteRegistration` 不只是算子名和版本，它描述 init、free、prepare、invoke 等回调及相关元数据；外部 API 使用的 `TfLiteOperator` 是不透明句柄，由其创建和配置函数操作。**不透明**说明调用方不依赖字段布局，并不等于可以不核对具体发行版的 ABI/API 保证。带 `Flex` 前缀的自定义编码通常是转换时选择的 TensorFlow ops（Select TF Ops），须有相应 Flex 支持；不是看到前缀便能自动运行，也不是所有自定义 op 都属于 Flex。

## 3. 先搭子图，再导入内容

![图 3-2：Interpreter 的对象归属与执行入口](figures/06-interpreter-relations.png)

*图 3-2　Interpreter 的对象归属与执行入口。一个 Interpreter 管理多个 Subgraph；一个 Subgraph 管理自己的 Tensor 和节点记录。多个节点可读同一个 Tensor，一个节点也可读取多个 Tensor。Context 是子图交给 kernel 的接口；画出回调关系并不表示 kernel 拥有子图。*

Interpreter 初始化第 0 个 Subgraph；Builder 按模型子图数补齐其余对象，因此模型子图索引和运行时子图索引一一对应。主子图是默认接口使用的图，有 Signature 时也可以指定其他图。原稿中的 Subgraph 构造参数可以按是否共享理解：

| 每个图独有 | Interpreter 统一管理或提供 |
|---|---|
| 子图索引、`TfLiteContext`、张量、节点及执行计划 | ErrorReporter、外部 Context 数组、子图容器、资源表、名称到资源 ID 的映射、初始化状态映射 |

`TfLiteContext` 是交给 kernel 或 Delegate 的 C 风格接口表。`impl_` 记录实现对象，桥接函数把调用转回当前 Subgraph；它还暴露 `tensors`、`tensors_size` 等。Kernel 和 Delegate 模式不是两张计算图：是在合适时机切换 Context 可调用的接口。Delegate 准备时需要查看执行计划、查询节点和注册信息、替换受支持的节点分区；普通 kernel 不应任意改图。Interpreter 的常规 Context 指向主子图的 Context。原稿中的初始张量 reserve 容量不是张量上限，外部 Context 的枚举数也不是可用加速器数量。

`ExternalCpuBackendContext` 是供子图共享的外部 Context 包装；内部真正提供 CPU 后端状态的 `CpuBackendContext` 可按需创建，不能看到包装对象就认为线程池和矩阵缓存都已初始化。线程数通过 `recommended_num_threads` 等路径传给运行时，`-1` 表示没有显式指定，不代表无限线程。Interpreter options 可以影响零点压缩、委托聚类或大张量分配等行为，具体字段以当前 checkout 为准。

构图也可挂性能采集：`MaybeCreatePlatformProfiler` 是否产生 profiler 取决于平台、编译配置和实现；例如 Android 的 atrace 或 Apple 的 signpost，不能假定每个构建都同时提供同一套采集器。Telemetry profiler 是另一条接收事件的路径，可与平台 profiler 并存；未提供就不应期待出现对应遥测。部分版本会在构图时为遥测记录子图的张量量化信息。

## 4. ParseTensors：从描述到运行时张量

先 `AddTensors` 建张量槽位并让 Context 能定位它们，再设图的输入、输出索引，最后逐个配置这些槽位。AddTensors 扩的是 `TfLiteTensor` 描述对象数组，不等于为所有张量的数据分配内存。Builder 把 schema 的 `TensorType` 映射到运行时 `TfLiteType`（例如 `INT32 → kTfLiteInt32`），并统计浮点张量，供后面的默认委托工厂判断。

### 4.1 数据地址、形状和生命周期

- `Tensor.buffer` 指向 **模型级** `Model.buffers`，不是某个子图私有的 buffer 表。常规权重可来自 `Buffer.data`，也可由 offset/size 定位到 FlatBuffer 序列化对象以外、但仍在**同一个 Allocation** 内的字节。检查范围时用 `offset <= bytes && size <= bytes - offset`，不要先做可能溢出的 `offset + size`。
- `external_buffer` 则是另一个引用机制，必要时由支持此功能的环境提供单独资源；不能把所有“外部”误读成模型同包里的偏移。
- `shape` 给当前维度；`shape_signature` 用 `-1` 标出未固定的维度。例如当前 `[1, 4]`，签名 `[-1, 4]` 代表第一维可变，而不是“分配 −1 个元素”。还要看具体索引、kernel 和 Delegate 是否支持所请求的 resize。
- Tensor 的名称和序号分别回答“怎么辨认”和“节点怎么连”；变量 tensor 是跨调用需要保有状态的一类张量，不是“维度可变”的同义词。模型中收集的变量索引另行交给 Subgraph 管理。

| 导入路径 | 可能的分配类型 | 含义 |
|---|---|---|
| 可直接引用的常量数据 | `kTfLiteMmapRo` | 运行时将其视为只读；名字并不保证指针一定来自 mmap |
| 普通输入、中间值、输出 | `kTfLiteArenaRw` | 通常由 arena 按生命周期计划分配 |
| 可保持状态的定长变量 | `kTfLiteArenaRwPersistent` | 不按普通临时值复用，跨调用保留，仍可按规定重置 |
| string/resource/variant 等特例 | `kTfLiteDynamic` | 走动态分配路径；支持的变量组合以实现为准 |

原稿的只读路径对普通密集张量会核对理论字节数不大于实有 buffer；字符串、资源、Variant、稀疏和外部数据等不能照这个公式验，不是“跳过以后再用同一个公式验”。常量与 variable 冲突时不能简单把固定数据地址改为可写；见该版本的实际错误分支。地址不满足类型对齐时，原稿对应路径把数据复制到堆，清掉原 Allocation 引用，并用 `aligned_ro_buffers_` 管理副本。该路径额外补 16 字节供特定 SIMD 内核安全读取，不能据此推断所有 Tensor 都有相同 padding。只读是运行时的数据使用约定，与 OS 页权限不是一回事。

### 4.2 量化：从文件参数变成运行时结构

`ParseQuantization` **搬运和校验参数，不在构图阶段重新量化全部权重**。`TfLiteQuantization.type` 决定如何解释 `params` 指向的结构。没有量化信息时为 `kTfLiteNoQuantization`。

| 参数形式 | 解释 |
|---|---|
| Affine per-tensor | 一组 scale、zero point，反量化近似 `x = scale × (q − zero_point)` |
| Affine per-axis | 沿 `quantized_dimension` 的坐标选择参数。例如形状 `[2, 3]` 的 axis=1 可选三组参数，每列一组，不是给三种维度各一组 |
| Blockwise | 老形式按指定轴用 `block_size` 划分；含 `block_shape` 的形式逐维描述每块占多少元素，检查长度等于 rank 且各项大于零 |
| MultiAxis | 按多个量化轴的坐标组选择参数，原稿涉及 scale/zero-point Tensor 引用、轴次序及末轴分块；`0` 不分块，不能当作普遍必带的格式 |

Affine 分支检查参数个数、量化轴和 scale 个数与对应轴长度的关系。实验选项打开时，如果 zero point 全部相同，运行时可只保留一个值；读取方必须识别这种表示，尤其是自定义或外部 Delegate。Blockwise/MultiAxis 的 scale 和 zero point 可通过子图内的 Tensor 索引引用而非直接存成数组，可选引用可能用 `-1`。检查索引有效不代表每个内核都接受这种类型或布局；还须看 Prepare 与硬件后端的支持。量化格式详见[第一篇](01-model-format.md)。

### 4.3 稀疏张量：字节短了，索引不能少

`ParseSparsity` 处理的是“权重值以压缩形式保存，怎么找回坐标”。`traversal_order` 指定遍历维度顺序；分块时附加块内维度，`block_map` 把这些维度映射回原张量轴。`dim_metadata` 逐层描述 DENSE（坐标全存在，按 dense size 遍历）或 SPARSE_CSR（按 segments 划出每个父位置对应的 indices 范围）。例如一层的 `segments=[0, 2, 3]` 与 `indices=[1, 3, 2]`，表示第一个父位置下有坐标 1、3，第二个父位置下有坐标 2。索引序列可有 8、16、32 位编码，原稿的解析器统一转为 `TfLiteIntArray`。这里没有把稀疏权重展开成密集矩阵；运行时保有压缩值和元数据。原稿路径的稀疏导入用于固定数据，不应假设可写路径能接受同样的 sparsity 参数。

量化结构和稀疏元数据由临时 RAII 封装而不是因为“局部变量不能 delete”。这类结构内部可能有单独申请的内存；只有在所有权成功交接时才 `release`，失败路径让相应 deleter 清理，模型字节借用与这些结构的所有权不能混为一谈。

## 5. ParseNodes：把张量索引和算子实现接起来

每个序列化 Operator 给出 I/O 索引，Builder 通过 `opcode_index` 取得 registration，再为节点准备参数。简化逻辑如下：

```cpp
// 逻辑示意；检查和错误处理从略。
const TfLiteRegistration* registration = mapping[op->opcode_index()];
if (is_custom) {
  // 获取 custom_options，或者经边界检查后引用 Allocation 中的偏移。
  // 保存 init_data、init_data_size。
} else {
  // ParseOpData(..., &builtin_data)，按算子解析专用参数。
}
subgraph->AddNodeWithParameters(inputs, outputs, intermediates,
                                init_data, init_data_size,
                                builtin_data, registration);
```

- **Builtin**：`ParseOpData` 调用对应算子的解析器。以 ADD 为例，`TfLiteAddParams` 携带融合激活类型以及定点路径相关的 `pot_scale_int16` 配置；转换函数把 schema 的激活枚举映射为运行时枚举。没有执行加法，只是准备参数。
- **Custom**：`custom_options` 或者经校验的 `large_custom_options_offset/size` 是交给注册实现解释的字节。判断 CUSTOM 以 opcode 为准，不用“有 custom options”反推类型；没有自定义 options 也可以是合法 custom op。
- `BuiltinDataAllocator` 抽象参数结构的申请与释放；placement new 负责在已有存储中构造对象，不会让 `free` 自动调用析构。具体实现采用适合其参数结构的约定，并借助作用域对象清理失败时尚未转交的参数。若启用标准 aligned_alloc，其 size 必须满足 alignment 的倍数要求。Micro 的静态内存构建和逐算子解析入口不是桌面动态 Interpreter 的通用行为。

`nodes_and_registration_` 每个元素包含 `TfLiteNode` 和对应的 `TfLiteRegistration`：Node 持有输入、输出、临时/中间索引与参数等，registration 才告诉它如何 Prepare/Invoke。原始执行计划收录新加入的节点索引；计划可因 Delegate 而改写，所以“所有节点的存储数组”和“当前要依次执行的节点索引”是两回事。以 `inputs=[0,1], outputs=[2]` 的 ADD 为例，槽位 0/1 指向输入，槽位 2 对应输出，执行时由该版本 ADD kernel 实现读取和写入，不用按字符串名称找 tensor。

## 6. 签名、默认委托与分配入口

`ParseSignatureDefs` 从文件的 signature key、subgraph_index 和具名 I/O 映射建立外部接口。这些名字用于调用签名，不等于必须覆盖原始 Tensor 名。构图结束前还导入 Metadata，按条件保存 resolver 的 lazy delegate creators，并执行调用方显式指定的委托。原稿中的 QS8/QU8 编译开关或 FP32 数量只决定是否获取默认工厂，不保证 XNNPACK 已创建或必然接管该图。显式 Delegate 被调用后用途仍可能未定，采用拥有型重载时实现会保留委托对象的所有权，不能因状态失败就假定马上释放，详见[第四篇](04-resolver-delegate.md)。

原稿末尾再次应用动态大张量分配选项，是为了按最终图状态决定哪些张量可从 arena 改为 dynamic；所示分支会跳过图输入，只处理达到阈值的 `kTfLiteArenaRw` 张量，并清空旧数据指针，不是“在 Builder 里立即分配所有大张量”。实际运行通常经历 **建图 → AllocateTensors（Prepare/分配）→ 填输入 → Invoke**。排查错误时先定位失败阶段：注册失败查 opcode/版本，Prepare 失败查形状、量化或 unresolved custom，Invoke 失败再追内核执行。动态形状、委托和多次 resize 可能让准备工作在后续时机重跑。

## 源码入口

- [LiteRT CompiledModel 中的 InterpreterBuilder 使用](https://github.com/google-ai-edge/LiteRT/blob/main/litert/runtime/compiled_model.cc)
- [TensorFlow Lite InterpreterBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/interpreter_builder.cc)
- [运行时 Subgraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/subgraph.cc) · [OpResolver 实现](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.cc)
- [FlatBuffer 参数转换](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/flatbuffer_conversions.cc)

核对具体分支时，优先搜索本文列出的函数名。多轴量化、外部 buffer、遥测与委托工厂等是容易随 checkout 或编译开关变化的部分。
