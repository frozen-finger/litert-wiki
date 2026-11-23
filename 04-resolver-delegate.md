# Interpreter 执行（一）：Resolver、Registration 与 ABS

上一篇：[Builder（三）](03c-interpreter-nodes.md) · [目录](README.md) · 下一篇：[执行（二）](04b-interpreter-prepare.md)

上一主题的 Builder 把节点接到了 Registration。这一篇先看 Resolver 怎样提供实现，再以 ABS 跟完 Init、Prepare、Resize 和 Invoke。Delegate 的创建、选点和分区另在后面的主题里按发生顺序展开。

## 1. Resolver 查找什么

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

对照：[OpResolver 实现](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/api/op_resolver.cc)。

op_resolver是一个抽象基类，其中主要有以下的几个函数：

- findop 通过提供的op信息找到对应的registration，有两种，一种可以通过builin的enum code找到对应的算子实现，还有一种可以通过op的name找到算子实现
- GetDelegateCreators 返回delegatec的创建方法，可以用来针对flatbuffer中的op创建对应的delgate
`GetOpaqueDelegateCreators()` 面向不透明 C Delegate API；不能把它等同于某个特定发行渠道。
- MayContainUserDefinedOps 返回是否有builtin之外的用户自定义op，目前默认返回为true，代表了可能会存在builtin之外的op但不代表一定存在

还包括一些重要的属性：

```cpp
  using TfLiteDelegatePtrVector =
      std::vector<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>>;
```

代表了一系列delegate

```cpp
  using TfLiteDelegateCreator =
      std::function<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
          TfLiteContext* /*context*/)>;

  using TfLiteDelegateCreators = std::vector<TfLiteDelegateCreator>;
```

delegate的创建函数

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

- ChainOpResolver, 与addall功能类似，但是实现上只保存了指向resolver的指针，当需要时链式查找保存的指针指向的resolver有没有适合的算子，所以如果原本resolver中已有了对应算子的注册则不会查找到传入的resolver

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

与上者类似，只是匹配改用name而不是op code

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

这里的四个参数即为init，free, prepare和invoke的指针，一次看一下每个方法的实现

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

最后是invoke方法，根据tensor的数据类型主要包含两种实现方式，这里取tensor时是取的input中的第一个tensor。
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

这里完成三件事，首先计算了量化输入消除zeropoint后的绝对值，之后判断输入与输出的scale是否不同，如果不同就通过整数运算重新量化，最后将结果限制到给定范围内，对INT8就是-128，127

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
