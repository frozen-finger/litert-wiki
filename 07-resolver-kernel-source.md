# LiteRT 源码精读（七）：Resolver、ABS 与 XNNPACK 创建

[上一篇：InterpreterBuilder](06-interpreter-builder-source.md) · [Delegate 概览](04-resolver-delegate.md) · [下一篇：准备与图改写](08-prepare-delegate-source.md)

沿原稿保留 Resolver 的注册方法、Registration 的回调、ABS 的一次计算，以及 XNNPACK 适配层的创建。这篇先回答“节点由谁实现”，下一篇再回答“哪些节点能换成委托执行”。

本文是逐段整理的原稿。长代码可展开，短代码直接显示；不同代码块可能位于同一函数的不同位置，不应拼成独立示例编译。上游链接用于对照固定快照，少数原稿中的版本差异以实际 checkout 为准。

opresolver决定了所有的operator会被如何实现

## 1. OpResolver 的接口及两类查找

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

### 1.1 TfLiteRegistration 的生命周期回调

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

### 1.2 TfLiteDelegate 与它负责的存储

同样是一个抽象结构体，有以下的方法：

- prepare, 接受context和TfLiteDelegate两个参数，在modifygraphwithdelegate时会被调用，会查看整体的context中的计算图并通知tensorflowlite runtime替换掉其中对应的算子为本delegate实现

- CopyFromBufferHandle，接受context，TfLiteDelegate，TfLiteBufferHandle, TfLiteTensor四个参数，从delegate的bufferhandle中复制数据到对应的tensor中 （转移到cpu能直接访问的内存地址中）

- CopyToBufferHandle，与上述效果相反

- `FreeBufferHandle` 按 Delegate 契约释放句柄及其负责的资源；没有该回调不证明后端完全不拥有私有存储。

以下为成员对象：

- flags，使用二进制存储delegate的一些特征
  1. kTfLiteDelegateFlagsAllowDynamicTensors = 1 true代表delegate可以处理dynamic tensor，也要求delegate处理动态的输出
  2. kTfLiteDelegateFlagsRequirePropagatedShapes = 2，true代表对于dynamic tensor，在调用delegate kernel的prepare之前，自动完成相关tensor的形状传播，即通过输入的tensor形状修改对应的之后的tensor
  3. kTfLiteDelegateFlagsPerOperatorProfiling - 4，调整性能的监控到更细的级别，不为整个delegate记录事件，而是会记录delegate内部的算子事件，最后将数据汇总到operator-wise区域（delegate内部可能包含多个算子）
  4. kTfLiteDelegateFlagsHintFullyDelegatedToSingleDelegate = 8，内存分配的优化提示，保证subgraph中所需要的所有算子都可以由一个delegate代理，可以让LiteRT runutime跳过原本用于准备CPU执行路径的内存分配功能做，是提示但不保证delegate有这个功能

### 1.3 TfLiteNode：一个节点保存什么

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

## 2. MutableOpResolver 与内建注册

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

## 3. ABS：从注册到计算

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

### 3.1 ResizeTensor 的快慢路径

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

### 3.2 Invoke：普通、量化和 INT16 路径

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

## 4. XNNPACK 工厂、创建与资源所有权

对照：[XNNPACK Delegate](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc)。

```cpp
  delegate_creators_.push_back([](TfLiteContext* context) {
    return tflite::MaybeCreateXNNPACKDelegate(context,
                                              XNNPackQS8Options::default_value);
  });
```

builtinopresolver创建的最后，还会向delegate_createors_中推入了一个lambda表达式用于创建delegate
MaybeCreateXNNPACKDelegate受到编译期宏TFLITE_BUILD_WITH_XNNPACK_DELEGATE影响（即是否集成XNNPACK），XNNPACK时一个google提供的高性能CPU神经网络推理计算库，XNNPackQS8Options时LiteRT中控制XNNPACK部分量化算子支持情况的配置枚举，此处的XNNPACK delegate也就是XNNPACK和LiteRT之间的适配层，基于集成或不集成分为两种情况。

集成的情况：

```cpp
TfLiteDelegatePtr MaybeCreateXNNPACKDelegate(
    TfLiteContext* context, XNNPackQS8Options xnnpack_qs8_options) {
  auto opts = TfLiteXNNPackDelegateOptionsDefault();
  switch (xnnpack_qs8_options) {
    case XNNPackQS8Options::enabled:
      opts.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
      break;
    case XNNPackQS8Options::disabled:
      opts.flags &= ~(1UL << (TFLITE_XNNPACK_DELEGATE_FLAG_QU8 - 1));
      break;
    case (XNNPackQS8Options::default_value):
    default:
      break;
  }

  return TfLiteDelegatePtr(
      TfLiteXNNPackDelegateCreateWithThreadpool(&opts, context),
      TfLiteXNNPackDelegateDelete);
}
```

defaultoptions为XNNPACK根据编译期宏配置的一些默认参数
QS8代表是否支持quantizad signed 8-bit，QU8代表是否支持quantizad unsigned 8-bit

```cpp
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;
```

TfliteDelegatePtr是一个指向TfLiteDelegate的指针，TfLiteXNNPackDelegateCreateWithThreadpool返回了一个TfLiteDelegate对象

```cpp
TfLiteDelegate* TfLiteXNNPackDelegateCreateWithThreadpool(
    const TfLiteXNNPackDelegateOptions* options, TfLiteContext* context) {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  xnn_workspace_t workspace = nullptr;
  if (xnn_create_workspace(&workspace) != xnn_status_success) {
    return nullptr;
  }

  auto* xnnpack_delegate =
      new ::tflite::xnnpack::Delegate(options, workspace, context);
  return xnnpack_delegate ? xnnpack_delegate->tflite_delegate() : nullptr;
}
```

xnn_initialize等都属于XNNPACK包中的内容，在此不做赘述，只说明Delegate的构造过程

### 4.1 线程池：复用还是自己创建

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
  explicit Delegate(const TfLiteXNNPackDelegateOptions* options_ptr,
                    xnn_workspace_t workspace, TfLiteContext* context = nullptr)
      : options_(options_ptr ? *options_ptr
                             : TfLiteXNNPackDelegateOptionsDefault()) {
    int num_subgraphs = 1;
    if (context) {
      tflite::Subgraph* this_subgraph =
          reinterpret_cast<tflite::Subgraph*>(context->impl_);
      num_subgraphs = this_subgraph->GetSubgraphs()->size();
    }

    static_unpacked_data_.resize(num_subgraphs);
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
    pthreadpool_t threadpool = nullptr;
#ifdef TFLITE_KERNEL_USE_XNNPACK
    if (context != nullptr) {
      threadpool =
          CpuBackendContext::GetFromContext(context)->get_xnnpack_threadpool();
    }
#endif
    if (threadpool != nullptr) {
      threadpool_.reset(threadpool);
      own_threadpool_ = false;
    } else {
      own_threadpool_ = true;
      if (options_.num_threads > 1) {
        threadpool_.reset(
            pthreadpool_create(static_cast<size_t>(options_.num_threads)));
        threadpool = threadpool_.get();
      }
    }

#endif
```

</details>

这一段主要是针对XNNPACK的多线程处理，先是通过!defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)判断了当前是否为Emscripten的单线程环境，如果是则直接跳出，#ifdef TFLITE_KERNEL_USE_XNNPACK代表了原生平台或者Emscripten多线程可以从Context中获取线程池.
CpuBackendContext继承自TfLiteInternalBackendContext, 代表了一系列可供tflite使用的环境工具，如创建线程池等。own_threadpool被用来区分线程池是由外部传入的还是自己创建的从而在销毁时决定是否释放线程池。如果是XNNPACK下的线程池会默认拥有context中设置的最大线程数，否则会对应options中设置的线程数

### 4.2 权重缓存与 provider

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite XNNPACK delegate for CPU.");

    delegate_.flags = GetXNNPackDelegateFlags();
    workspace_.reset(workspace);

    if (!options_.weights_cache) {
      if (options_.weight_cache_provider) {
        weight_cache_provider_ = maybe_unique_ptr<MMapWeightCacheProvider>(
            reinterpret_cast<MMapWeightCacheProvider*>(
                options_.weight_cache_provider),
            kNotOwned);
      }
      if (!weight_cache_provider_->IsActive() &&
          (options_.weight_cache_file_path ||
           options_.weight_cache_file_descriptor > 0)) {
        FileDescriptor fd;
        if (options_.weight_cache_file_descriptor > 0) {
          fd.Reset(options_.weight_cache_file_descriptor);
        }
        if (!weight_cache_provider_->LoadOrStartBuild(
                options_.weight_cache_file_path, std::move(fd))) {
          TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                          "XNNPack weight cache could neither be loaded from "
                          "or saved to '%s'. Check that this location is "
                          "readable and writable.",
                          options_.weight_cache_file_path);
        }
      } else if (!weight_cache_provider_->IsActive() &&
                 !weight_cache_provider_.owning()) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_ERROR,
            "XNNPack weight cache was manually overridden but not loaded and "
            "no file path or file descriptor was provided.");
      }
      if (weight_cache_provider_->IsActive()) {
        if (options_.weight_cache_lock_memory) {
          if (!weight_cache_provider_->LockMemory()) {
            TFLITE_LOG_PROD(
                tflite::TFLITE_LOG_ERROR,
                "XNNPack weight cache could not be locked in memory.");
          }
        }

        options_.weights_cache =
            reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(
                &weight_cache_provider_->GetCacheProvider());
        options_.weight_cache_file_path =
            weight_cache_provider_->GetFilePath().data();
      } else {
        TFLITE_LOG(tflite::TFLITE_LOG_VERBOSE,
                   "XNNPack weight cache not enabled.");
      }
    }
  }
```

</details>

weights_cache_是XNNPACK的权重缓存，weight_cache_provider是负责提供，查找，保存以及管理缓存内容的对象，在XNNPACK中模型权重通常需要经过weight packing权重重排，转换成更适合CPU读取的内存布局，但转换比较耗时所以XNNPACK可以缓存转换后的权重供后续runtime或者interpreter使用。
若配置提供外部 provider，Delegate 使用但不接管其所有权；若传入适用的缓存路径且没有外部 provider，可以建立自己的。锁页用于减少换出，并不消除所有缺页。

对于没有启用XNNPACK的情况下，则会直接返回空指针

```cpp
TFLITE_ATTRIBUTE_WEAK TfLiteDelegatePtr AcquireXNNPACKDelegate() {
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}
```
