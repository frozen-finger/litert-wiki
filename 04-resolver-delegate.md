# Interpreter 执行（一）：Resolver、Registration 与 ABS

上一篇：[Builder（三）](03c-interpreter-nodes.md) · [目录](README.md) · 下一篇：[执行（二）](04b-interpreter-prepare.md)

回到 [Builder（一）的构造入口](03-interpreter-builder.md#1-builder-的构造与入口检查)：调用方传入 `model` 和 `resolver`。模型给出「我要哪个算子、哪一版」，Resolver 给出「本次构建用哪套实现」。[Builder（三）](03c-interpreter-nodes.md#1-parsenodes把-tensor-和注册项接起来)已经把查到的 Registration 放进节点记录；本篇沿这条链往回看一次查找，再往前看 ABS 怎样执行。

## 1. Resolver 查找什么

### 先和 InterpreterBuilder 对上

先延续 [上篇的 ADD](03c-interpreter-nodes.md#2-以-add-走完参数交接)：T0、T1 作为输入，T2 作为输出。在这版源码中，[`Register_ADD`](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/add.cc#L496-L504)提供 ADD 的注册项，[`Prepare`](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/add.cc#L90)用输入形状确定输出形状，[`Eval`](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/add.cc#L415)才真正读取输入、计算结果。下面换成 ABS，查表和绑定的步骤不变；换例子是为了看清单个内核的量化和 Resize 分支。

以一个序列化 ABS 节点为例，设它的 `opcode_index=0`，读 T3、写 T7。读代码时依次看这几个地方：

| 时机 | 谁在做 | 交给下一步什么 |
| --- | --- | --- |
| 构造 Builder | 调用方传入 Resolver，Builder 保存引用 | `op_resolver_` |
| 建立本地映射 | `GetRegistrationFromOpCode()` 读 `Model.operator_codes[0]`；内建 ABS 按类型和版本 `FindOp` | `flatbuffer_op_index_to_registration_[0]` |
| 导入节点 | `ParseNodes()` 取本地表第 0 项，连同 T3/T7 等节点信息传给 `AddNodeWithParameters()` | Subgraph 中的节点和 Registration 副本；有 Init 回调时，节点保存它返回的状态 |
| 准备与执行 | Subgraph 从执行计划找到节点及 Registration | Prepare 推导形状，Invoke 读 T3 并写 T7 |

文件里第 0 项可以被多个 Operator 引用。若第二个 ABS 读的是 T5，就仍可用同一组回调，但它不会借用第一个 ABS 的 I/O 索引或 `user_data`。查找不到时也要分开看：未注册的内建版本会让 Builder 失败；有名称的自定义算子可以先成为占位项，等待适配的 Delegate 接管，否则会在准备阶段报错。

另有一条独立支路：同一个 Resolver 可提供**默认 Delegate 工厂**。Builder 会在符合条件时保存工厂，供后续尝试改图；给 ABS 找到普通 Registration 不意味着选了 XNNPACK，更不意味着该节点已经被委托。[创建与应用 Delegate](04c-delegate-setup.md#3-显式应用与-lazy-delegate)会接着讲这条线。

| 需要回答的问题 | 对象 |
| --- | --- |
| 模型要求的内建算子版本有对应实现吗？ | `FindOp(builtin_code, version)` |
| 这个名称和版本的自定义算子有对应实现吗？ | `FindOp(custom_name, version)` |
| 可以提供哪些默认 Delegate？ | `GetDelegateCreators()` / `GetOpaqueDelegateCreators()` 返回工厂，不是已创建实例 |
| 是否可能含有用户自定义算子？ | `MayContainUserDefinedOps()`；基类默认保守返回 true，并不说明已找到一个自定义节点 |

Opaque 工厂用于不透明 C API 对象，不能仅按「是不是 Google Play services」来理解。两个 `FindOp` 返回的是注册表项，不是执行结果，也不凭名称或版本号推导向后兼容。

`MutableOpResolver` 在抽象接口之上维护 `(code, version) → registration` 和 `(name, version) → registration`。查找时先看本地表，再按顺序查链里的 Resolver。


- `AddBuiltin` 和 `AddCustom` 分别按枚举和名称注册。在所指定的版本或版本区间内复制并补齐注册信息；同键覆盖原条目。复制的是注册记录，函数指针并不会复制一份函数实现。
- `AddAll` 合入另一个 Mutable Resolver 的映射，并在原稿实现中带入其链式引用；`ChainOpResolver` 本身保存引用，未命中时才调用对方，不是复制条目。链中的对象必须活到查找结束。
- `may_directly_contain_user_defined_ops_` 是保守标记。查询还会考虑链上的 Resolver，不能拿它枚举模型使用的自定义节点。
- `BuiltinOpResolver` 在 Mutable Resolver 上注册标准内建实现，也可能提供默认 Delegate 工厂。它是常见选择，不是唯一选择；应用可以显式提供更小的注册集合，或选择不附自动默认委托的 Resolver。

对照：[OpResolver 实现](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/api/op_resolver.cc)。

`OpResolver` 是抽象基类。前两个 `FindOp` 是 Builder 做普通算子绑定时调用的接口；工厂与能力查询由不同的代码路径使用：

- `FindOp`：按内建枚举或自定义名称，加上版本，返回 Registration。
- `GetDelegateCreators()`：返回工厂函数。工厂提供潜在后端，不逐个返回某个算子的实现。
`GetOpaqueDelegateCreators()` 面向不透明 C Delegate API；不能把它等同于某个特定发行渠道。
- `MayContainUserDefinedOps()`：用于保守判断当前 Resolver 是否可能提供用户自定义算子，默认 `true` 不表示某个模型实际用了它们。

还包括一些重要的属性：

```cpp
  using TfLiteDelegatePtrVector =
      std::vector<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>>;
```

这是拥有若干 Delegate 实例的容器，元素带对应的删除器。

```cpp
  using TfLiteDelegateCreator =
      std::function<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
          TfLiteContext* /*context*/)>;

  using TfLiteDelegateCreators = std::vector<TfLiteDelegateCreator>;
```

这是工厂函数及其集合。函数接收 Context，返回拥有型 Delegate 指针；保存工厂和执行工厂是两步。

## 2. Registration 与 Node 的职责

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

`TfLiteRegistration` 是 C 风格结构体。以下保留原稿关注的回调，并用当前完整结构来讲；兼容类型 `TfLiteRegistration_V3` 不等于最新完整结构。

- `init(context, buffer, length)` 为一个节点创建私有状态。Builtin 传类型化参数时长度可为 0；Custom 通常取得原始字节和长度。无状态实现可省略。

- `free(context, data)` 清理由 Init 返回的私有状态，不一定等于序列化参数地址。

- `prepare(context, node)` 验证参数、传播输出形状、建立临时需求；尺寸变化后可能重新 Prepare，但不是每次 Invoke 都要重做。

- invoke，调用算子，参数为context和TfliteNode

- profiling_string，将对应于一个TfliteNode的操作在profile总结时集成在一起

- 异步内核接口可提供 `TfLiteAsyncKernel`。没有异步接口，并不代表实现无法在内部使用线程池。

包含以下成员变量

- builtincode

- custom op name

- version

- registration_external TfLiteOperator，一个指向c风格对象的指针，用于提供功能给C风格的api调用者使用

- inplace_operator 代表了输出向量是否可以安全覆盖输入向量

`TfLiteNode` 是图中的一次算子调用，不是抽象基类。静态内存宏影响辅助 API 与部分成员，不能把桌面路径直接推到 Micro 上。

```cpp
typedef struct TfLiteNode {
  TfLiteIntArray* inputs;

  TfLiteIntArray* outputs;

  TfLiteIntArray* intermediates;

  TfLiteIntArray* temporaries;

  void* user_data;

  void* builtin_data;

  const void* custom_initial_data;
  int custom_initial_data_size;

  struct TfLiteDelegate* delegate;

  bool might_have_side_effect;
} TfLiteNode;
```

主要包含了input，output以及指向delegate的指针

如果要看这些回调在何时分派与续跑，接着看 [Interpreter 执行（二）](04b-interpreter-prepare.md)。

## 3. MutableOpResolver 与内建注册

对照：[MutableOpResolver](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/mutable_op_resolver.cc) · [BuiltinOpResolver](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/kernels/register.cc)。

multableopresolver继承了opresolver，并额外添加了以下的成员

- AddBuiltin, 接受builtin code，registration，以及version的方法，会替代掉对应版本范围内之前使用的registration

- `AddCustom` 用名称与版本，而不是 builtin code，写入自定义注册表。

- AddAll，接受mutableopresolver的指针作为参数，向本resolver添加所有指针指向的resolver中的算子

- `ChainOpResolver` 保存对另一个 Resolver 的引用。本地找不到才查链；本地已有相同 key 时，不会去链上替换实现。与 `AddAll()` 直接合入注册表区分开。

- builtins, 保存opcode到registration的映射

- custom_ops_, 间接保存了custom op的name到registration映射

- may_directly_contain_user_defined_ops_，代表了当前的resolver是否已经包含了用户自定义的op

介绍完了额外内容，现在来看关键函数的实现, 由于有些函数功能相同只是参数不太一样不做特别区别与说明：

```cpp
const TfLiteRegistration* MutableOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  auto it = builtins_.find(std::make_pair(op, version));
  if (it != builtins_.end()) {
    return &it->second;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    const TfLiteRegistration* result = other->FindOp(op, version);
    if (result != nullptr) {
      return result;
    }
  }
  return nullptr;
}
```

findop从builtins按照opcode和version直接查找有没有对应的registration，如果没有再从chainresolver接口传入的额外resolver做链式查找

```cpp
void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   const TfLiteRegistration* registration,
                                   int version) {
  if (registration == nullptr) {
    return;
  }
  TfLiteRegistration new_registration = *registration;
  new_registration.custom_name = nullptr;
  new_registration.builtin_code = op;
  new_registration.version = version;
  auto op_key = std::make_pair(op, version);
  builtins_[op_key] = new_registration;
  may_directly_contain_user_defined_ops_ = true;
}
```

做了一份浅拷贝后，按照code和version配置到了Builtin中

```cpp
void MutableOpResolver::AddCustom(const char* name,
                                  const TfLiteRegistration* registration,
                                  int version) {
  TfLiteRegistration new_registration = *registration;
  new_registration.builtin_code = BuiltinOperator_CUSTOM;
  new_registration.custom_name = name;
  new_registration.version = version;
  auto op_key = std::make_pair(name, version);
  custom_ops_[op_key] = new_registration;
  may_directly_contain_user_defined_ops_ = true;
}
```

内建和自定义查找的顺序相同；不同的是前者以枚举为 key，后者以名称为 key，两者都带版本。

```cpp
void MutableOpResolver::AddAll(const MutableOpResolver& other) {
  for (const auto& other_builtin : other.builtins_) {
    builtins_[other_builtin.first] = other_builtin.second;
  }
  for (const auto& other_custom_op : other.custom_ops_) {
    custom_ops_[other_custom_op.first] = other_custom_op.second;
  }
  other_op_resolvers_.insert(other_op_resolvers_.begin(),
                             other.other_op_resolvers_.begin(),
                             other.other_op_resolvers_.end());
}
```

除了builtin和custom正常复制外，也会复制other_op_resolver

```cpp
void MutableOpResolver::ChainOpResolver(const OpResolver* other) {
  other_op_resolvers_.push_back(other);
}

bool MutableOpResolver::MayContainUserDefinedOps() const {
  if (may_directly_contain_user_defined_ops_) {
    return true;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    if (OpResolverInternal::MayContainUserDefinedOps(*other)) {
      return true;
    }
  }
  return false;
}
```

在判断是否包含自定义op时也会查找otheropresolver

默认注册集合常由 `BuiltinOpResolver` 提供；应用也可以直接用 `MutableOpResolver` 构建更小的集合或选择不登记默认委托。

```cpp
BuiltinOpResolver::BuiltinOpResolver() {
  RegisterBuiltinOps();
  delegate_creators_.push_back([](TfLiteContext* context) {
    return tflite::MaybeCreateXNNPACKDelegate(context,
                                              XNNPackQS8Options::default_value);
  });
}
```

其中register执行了所有builtin op的注册

```cpp
void BuiltinOpResolver::RegisterBuiltinOps() {
  AddBuiltin(BuiltinOperator_ABS, Register_ABS(), /* min_version = */ 1,
             /* max_version = */ 5);
  AddBuiltin(BuiltinOperator_HARD_SWISH, Register_HARD_SWISH());
  AddBuiltin(BuiltinOperator_RELU, Register_RELU(), /* min_version = */ 1,
             /* max_version = */ 3);
  ...
```

以ABS为例：

到这里已经回答「框架怎样找到函数」。下面换一个比 ADD 更容易看清量化细节的一元算子 ABS。看的是同一份 Registration 如何分别参与节点初始化、形状准备和实际计算；ABS 切换了示例，没有改变此前介绍的建图方式。

## 4. 以 ABS 跟一次 CPU 算子

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

对照：[ABS 的注册、Prepare 和 Invoke](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/kernels/elementwise.cc)。

```cpp
TfLiteRegistration* Register_ABS() {
  static TfLiteRegistration r = {elementwise::ElementWiseQuantizedInit,
                                 elementwise::ElementWiseQuantizedFree,
                                 PrepareAbs, elementwise::AbsEval};
  return &r;
}
```

按 `Init → Prepare → Invoke` 的顺序跟三次调用；`Free` 对应节点状态清理。它们来自同一份 Registration，但触发时机不同：Init 通常在加节点时，Prepare 可以因重新分配或动态形状再次运行，Invoke 才读取本次输入。

```cpp
void* ElementWiseQuantizedInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
  return new OpData();
}

struct OpData {
  int32_t multiplier;
  int32_t shift;
  int input_offset;
  int output_offset;
  bool needs_rescale;
  union {
    int8_t* lut_int8;
    int16_t* lut_int16;
  };
};
```

创建了一个名叫opdata的结构体，这个结构被用来储存算子的参数，主要用于量化算子

- multiplier 将浮点缩放系数表示成定点乘法系数
- shift 配合multiplier，通过二进制移位实现缩放
- input_offset, 输入tensor的量化零点
- output_offset, 输出tensor的量化零点
- needs_rescale, 是否要将结果从输入量化尺度转换到输出量化尺度
- lut_int8, INT8算子使用的查找表
- lut_int16, INT16使用的查找表

有一些计算形式的代价较高比如log，但是输入的范围又有限，所以存在直接的查找表，可以从中找到对应输入的输出

```cpp
void ElementWiseQuantizedFree(TfLiteContext* context, void* buffer) {
  OpData* data = static_cast<OpData*>(buffer);
  if (data && data->lut_int8) {
    delete[] data->lut_int8;
  }
  delete data;
}
```

free相对应的清除了data的数据以及data中保留的查找表

```cpp
#define GENERIC_PREPARE(function_name, is_supported_type_function, type_name)  \
  static TfLiteStatus function_name(TfLiteContext* context,                    \
                                    TfLiteNode* node) {                        \
    return elementwise::GenericPrepare(context, node,                          \
                                       is_supported_type_function, type_name); \
  }

GENERIC_PREPARE(PrepareAbs, elementwise::IsAbsSupportedType,
                elementwise::kAbsName)
```

LiteRT使用了宏来辅助生成prepare函数，最后还是执行到elementwise中

```cpp
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node,
                            IsSupportedType is_supported_type,
                            const char* op_name) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!is_supported_type(input->type)) {
    TF_LITE_UNSUPPORTED_TYPE(context, input->type, op_name);
  }
```

对一系列的值做了检查，其中支持的类型包含以下四种type == kTfLiteFloat32 || type == kTfLiteInt8 ||
         type == kTfLiteInt16 || type == kTfLiteInt32

```cpp
  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 &&
       input->quantization.type != kTfLiteNoQuantization)) {
    TfLiteTensor* output = GetOutput(context, node, 0);
    auto* op_data = static_cast<OpData*>(node->user_data);
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);
```

当前只支持最简单的仿射量化

```cpp
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);
    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);
    op_data->input_offset = input_params->zero_point->data[0];
    op_data->output_offset = output_params->zero_point->data[0];
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, op_data->input_offset, 0);
      TF_LITE_ENSURE_EQ(context, op_data->output_offset, 0);
    }
    const float input_scale = input_params->scale->data[0];
    const float output_scale = output_params->scale->data[0];
    op_data->needs_rescale = input_scale != output_scale;
    if (op_name == kAbsName && op_data->needs_rescale) {
      SetAbsOutputMultiplier(input_scale, output_scale, &op_data->multiplier,
                             &op_data->shift);
```

计算一个multiplier用于转换scale

```cpp
    } else if (op_name == kRsqrtName) {
      if (input->type == kTfLiteInt16) {
        const void* lut_func_params = static_cast<const void*>(&output_scale);
        const auto lut_func = [](float value, const void* lut_func_params) {
          if (value <= 0.0f) {
            const float output_scale =
                *static_cast<const float*>(lut_func_params);
            return std::numeric_limits<int16>::max() * output_scale;
          }
          return 1.0f / std::sqrt(value);
        };
        if (!op_data->lut_int16) {
          op_data->lut_int16 = new int16_t[LUTSize<int16_t>()];
        }
        LUTPopulate<int16_t>(input_scale, input_params->zero_point->data[0],
                             output_scale, output_params->zero_point->data[0],
                             lut_func, lut_func_params, op_data->lut_int16);
      } else {
        SetRsqrtOutputMultiplier(input_scale, output_scale,
                                 &op_data->multiplier, &op_data->shift);
      }
    } else if (op_name == kLogName) {
      LogLUTPrepare(input->type, op_data, input_scale, op_data->input_offset,
                    output_scale, op_data->output_offset);
    }
  }
```

LOG、RSQRT 等部分量化路径可借助查找表，不能推广成每个 ABS 都预计算查找表。

```cpp
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}
```

resizetensor来自与subgraph创建时传入的静态函数指针，在之前的环节漏掉了

### ResizeTensor 的快慢路径

对照：[ResizeTensor](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1868-L1911) · [ResizeTensorImpl](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2150-L2199)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus Subgraph::ResizeTensor(TfLiteContext* context,
                                    TfLiteTensor* tensor,
                                    TfLiteIntArray* new_size) {
  const bool can_reuse_allocation = [tensor, new_size, context] {
    if (tensor->data.raw == nullptr) {
      return false;
    }
    if (!EqualArrayAndTfLiteIntArray(tensor->dims, new_size->size,
                                     new_size->data)) {
      return false;
    }
    if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
        tensor->type == kTfLiteVariant) {
      return true;
    }
    size_t new_bytes = 0;
    tflite::BytesRequired(tensor->type, tensor->dims->data, tensor->dims->size,
                          &new_bytes, context);
    return new_bytes == tensor->bytes;
  }();

  if (can_reuse_allocation) {
    if (new_size != tensor->dims) {
      TfLiteIntArrayFree(tensor->dims);
      tensor->dims = new_size;
    }
    return kTfLiteOk;
  }

  return static_cast<Subgraph*>(context->impl_)
      ->ResizeTensorImpl(tensor, new_size);
}
```

</details>

快路径要求已有数据指针且 shape 完全相同；定长类型还需字节数匹配。命中只更新等价的尺寸对象，不进入 `ResizeTensorImpl()`。

```cpp
TfLiteStatus Subgraph::ResizeTensorImpl(TfLiteTensor* tensor,
                                        TfLiteIntArray* new_size) {
  if (tensor->allocation_type == kTfLiteArenaRw ||
      tensor->allocation_type == kTfLiteDynamic ||
      tensor->allocation_type == kTfLiteArenaRwPersistent ||
      tensor->allocation_type == kTfLitePersistentRo ||
      tensor->allocation_type == kTfLiteCustom ||
      tensor->allocation_type == kTfLiteNonCpu) {
    tensor_resized_since_op_invoke_ |=
        TfLiteIntArrayEqual(tensor->dims, new_size) == 0;
    if (tensor->type != kTfLiteString && tensor->type != kTfLiteResource &&
        tensor->type != kTfLiteVariant) {
      size_t bytes_required;
      TfLiteStatus status =
          tflite::BytesRequired(tensor->type, new_size->data, new_size->size,
                                &bytes_required, &context_);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return kTfLiteError;
      }

      status = TfLiteTensorResizeMaybeCopyWithAllocator(bytes_required, tensor,
                                                        false, allocator_);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return status;
      }
      tensor->bytes = bytes_required;
    }
```

对普通类型先计算所需字节，再按 allocation 策略调整堆存储。String、Resource 和 Variant 不采用同一套字节数公式。

```cpp
    if (tensor->dims && tensor->dims != new_size) {
      TfLiteIntArrayFree(tensor->dims);
    }
    tensor->dims = new_size;

    if (tensor->allocation_type == kTfLiteArenaRw ||
        tensor->allocation_type == kTfLiteArenaRwPersistent) {
      tensor->data.raw = nullptr;
    }
```

其他情况下如果现有的维度不等则更新维度，如果需要arena memory planner申请内存就将data清空

```cpp
  } else {
    TfLiteIntArrayFree(new_size);
    ReportError("Attempting to resize a fixed-size tensor.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}
```

Resize 与最终分配不一定同一步发生：堆可立即调整，Arena 由后续内存规划分配；固定只读字节不允许在这里调整。

### Invoke：普通、量化和 INT16 路径

```cpp
TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteType type = input->type;
  switch (type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(
          context, node, [](float f) { return std::abs(f); }, type);
    case kTfLiteInt8:
      return AbsEvalQuantized<int8_t>(context, node, type);
    case kTfLiteInt16:
      return input->quantization.type == kTfLiteNoQuantization
                 ? AbsInt16EvalImpl(context, node, type)
                 : AbsEvalQuantized<int16_t>(context, node, type);
    case kTfLiteInt32:
      return EvalImpl<int32_t>(
          context, node, [](int32_t i) { return std::abs(i); }, type);
    default:
      TF_LITE_KERNEL_LOG(context, "Current data type %s is not supported.",
                         TfLiteTypeGetName(type));
      return kTfLiteError;
  }
}
```

到了 `Invoke` 才从节点的第一个输入索引拿到本次实际数据。先看非量化路径怎样逐元素取绝对值，再和量化路径对照：量化结果可能还需要换算到输出的 scale 和零点。
对于float32，int32的情况下会直接做evalimpl，evalimpl只是对原来最简单的std::abs方式做了一层封装，在外部额外检查了数据是否合法

```cpp
template <typename T>
TfLiteStatus AbsEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteType type) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const int kMin = std::numeric_limits<T>::min();
  const int kMax = std::numeric_limits<T>::max();

  std::function<T(T)> func = [&](T i) {
    const int32_t value = std::abs(i - op_data->input_offset);
    if (!op_data->needs_rescale) {
      return static_cast<T>(
          std::min(std::max(value + op_data->output_offset, kMin), kMax));
    }
    const int32_t output = MultiplyByQuantizedMultiplier(
                               value, op_data->multiplier, op_data->shift) +
                           op_data->output_offset;
    return static_cast<T>(std::min(std::max(output, kMin), kMax));
  };

  return EvalImpl<T>(context, node, func, type);
}
```

INT8 路径可以分三步读：减输入零点再取绝对值；必要时用整数乘数和移位换算尺度；加输出零点并限制到 `[-128,127]`。所以不能直接把存储的 INT8 字节当成真实数值取绝对值。

```cpp
inline TfLiteStatus AbsInt16EvalImpl(TfLiteContext* context, TfLiteNode* node,
                                     TfLiteType expected_type) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const int64_t num_elements = NumElements(input);
  const int16_t* in_data = GetTensorData<int16_t>(input);
  int16_t* out_data = GetTensorData<int16_t>(output);
  for (int64_t i = 0; i < num_elements; ++i) {
    out_data[i] = static_cast<int16_t>(
        std::abs<int32_t>(static_cast<int32_t>(in_data[i])));
  }
  return kTfLiteOk;
}
```

普通 INT16 路径扩大中间运算位宽。中间值扩大并不自动保证写回 INT16 可表示所有结果，边界行为要看对应内核。

ABS 的 `Prepare` 只是单个节点的回调。下一篇 [Prepare、分配与动态 Invoke](04b-interpreter-prepare.md)回到 Subgraph，解释这些回调按什么顺序调用，以及有的输出为什么要等到 Invoke 才知道尺寸。
