# InterpreterBuilder（三）：算子节点、Signature 与构建收尾

上一篇：[Builder（二）](03b-interpreter-tensors.md) · [目录](README.md) · 下一篇：[执行（一）](04-resolver-delegate.md)

这一篇从 Tensor 表追到 `TfLiteNode`，再跟 Signature、Metadata 和 Delegate 入口。目标是看清哪些数据由 Builder 填入，哪些工作留到 `AllocateTensors()` 或 `Invoke()`。

## 1. ParseNodes：把 Tensor 和注册项接起来

每个序列化 Operator 给出 I/O 索引，Builder 通过 `opcode_index` 取得 registration，再为节点准备参数。


- **Builtin**：`ParseOpData` 调用对应算子的解析器。以 ADD 为例，`TfLiteAddParams` 携带融合激活类型以及定点路径相关的 `pot_scale_int16` 配置；转换函数把 schema 的激活枚举映射为运行时枚举。没有执行加法，只是准备参数。
- **Custom**：`custom_options` 或者经校验的 `large_custom_options_offset/size` 是交给注册实现解释的字节。判断 CUSTOM 以 opcode 为准，不用“有 custom options”反推类型；没有自定义 options 也可以是合法 custom op。
- `BuiltinDataAllocator` 抽象参数结构的申请与释放；placement new 负责在已有存储中构造对象，不会让 `free` 自动调用析构。具体实现采用适合其参数结构的约定，并借助作用域对象清理失败时尚未转交的参数。若启用标准 aligned_alloc，其 size 必须满足 alignment 的倍数要求。Micro 的静态内存构建和逐算子解析入口不是桌面动态 Interpreter 的通用行为。

`nodes_and_registration_` 每个元素包含 `TfLiteNode` 和对应的 `TfLiteRegistration`：Node 持有输入、输出、临时/中间索引与参数等，registration 才告诉它如何 Prepare/Invoke。原始执行计划收录新加入的节点索引；计划可因 Delegate 而改写，所以“所有节点的存储数组”和“当前要依次执行的节点索引”是两回事。以 `inputs=[0,1], outputs=[2]` 的 ADD 为例，槽位 0/1 指向输入，槽位 2 对应输出，执行时由该版本 ADD kernel 实现读取和写入，不用按字符串名称找 tensor。

对照：[ParseNodes](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L335-L409)。

```cpp
TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph) {
  TfLiteStatus status = kTfLiteOk;

  subgraph->ReserveNodes(operators->size());
```

subgraph中用于存储node的结构为

```cpp
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>>
      nodes_and_registration_;
```

其中node保存了所有运算中需要的数据inputs, outputs, intermediates，包括了一个指向delegate的指针. Registration保存了该运算的实际执行逻辑, Registration的详细结构留到resolver以及registration的相关内容说明

```cpp
  for (int i = 0; i < operators->size(); ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Missing registration for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[index];
    if (registration == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_, "Skipping op for opcode_index %d\n",
                           index);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Found builtin operator %s with custom options.\n",
                           EnumNameBuiltinOperator(op_type));
    }
```

先用 `opcode_index` 取 Registration，再从注册项确定 builtin code。不能反过来用 `custom_options` 是否为空推断 CUSTOM。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
    void* builtin_data = nullptr;
    const char* init_data = nullptr;
    size_t init_data_size = 0;
    if (op_type == BuiltinOperator_CUSTOM) {
      if (op->custom_options()) {
        init_data = reinterpret_cast<const char*>(op->custom_options()->data());
        init_data_size = op->custom_options()->size();
      } else if (allocation_) {
        const uint64_t offset = op->large_custom_options_offset();
        if (offset > 1) {
          const size_t alloc_bytes = allocation_->bytes();
          const uint64_t size = op->large_custom_options_size();
          if (offset > alloc_bytes || size > alloc_bytes - offset) {
            TF_LITE_REPORT_ERROR(
                error_reporter_,
                "Custom Option Offset for opcode_index %d is out of bound\n",
                index);
            return kTfLiteError;
          }
          init_data =
              reinterpret_cast<const char*>(allocation_->base()) + offset;
          init_data_size = size;
        }
      }
    } else {
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
    }
    subgraph->AddNodeWithParameters(
        FlatBufferIntArrayToVector(op->inputs()),
        FlatBufferIntArrayToVector(op->outputs()),
        FlatBufferIntArrayToVector(op->intermediates()), init_data,
        init_data_size, builtin_data, registration);
  }

  return status;
}
```

</details>

Custom 参数可能来自内嵌字节，也可能来自经边界检查的同文件追加区。Builtin 参数使用 `BuiltinDataAllocator`；placement new 在已有存储上构造对象，`free()` 不会自动调用析构函数。按平台使用的 aligned allocation 必须满足对齐与大小约束。

## 2. 以 ADD 走完参数交接

顺着 [ParseOpData 的构建分支](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/api/flatbuffer_conversions.cc#L3023-L3057) 和 [ParseAdd](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/api/flatbuffer_conversions.cc#L1126-L1149) 可核对原稿提出的两个问题：

1. **为什么同时有两个参数对象？** 文件里的 `AddOptions` 是 FlatBuffer 视图，内核接收的 `TfLiteAddParams` 是 Runtime 布局。转换把融合激活枚举映射到 Runtime 枚举，并带上 `pot_scale_int16`；没有在这里执行 ADD，也没有改输入张量列表。
2. **分配后失败由谁收尾？** `SafeBuiltinDataAllocator` 先把对象置于带 deleter 的作用域句柄中；成功才通过 `release()` 交给 `builtin_data`。之后由节点构造路径接管，不是 `release` 当场释放，也不能把 placement new 与析构混为一谈。
3. **继续跟到 Init。** Custom 走原始 `init_data` 字节与长度；Builtin 可传类型化参数指针与约定的零长度。Init 的返回对象单独放在 `node.user_data`，与 `node.builtin_data` 分开。静态内存构建要查专用解析入口，不能直接照搬动态 Builder 的通用分派。

对照：[ParseAdd](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/api/flatbuffer_conversions.cc#L1126-L1149)。

```cpp
TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_type,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
#ifdef TF_LITE_STATIC_MEMORY
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "ParseOpData is unsupported on TfLiteMicro, please use the operator "
      "specific parse functions (e.g. ParseAdd etc.).\n");
  return kTfLiteError;
#else
  return ParseOpDataTfLite(op, op_type, error_reporter, allocator,
                           builtin_data);
#endif
}
```

STATIC_MEMEORY为一个编译期宏，用于支持TensorFlow Lite Micro这类以静态内存管理为主的嵌入式运行环境，如果定义了该宏则不走ParseOpData，而是需要调用者使用具体的paarseAdd等算子解析函数。Micro主要为了应对条件十分受限的嵌入式领域，只有一块固定容量的内存，一般情况下走ParseOpDataTfLite路径

```cpp
TfLiteStatus ParseOpDataTfLite(const Operator* op, BuiltinOperator op_type,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  ...

  SafeBuiltinDataAllocator safe_allocator(allocator);
  *builtin_data = nullptr;
  switch (op_type) {
    case BuiltinOperator_ABS: {
      return ParseADD(op, error_reporter, allocator, builtin_data);
    }
  ...
```

ParseOpDataTfLite中对allocator又做了一次封装，主要是为了cpp的RAII机制，保证allocator及其构造的数据可以被正确的析构，接着按照type的类型调用每个op所对应的解析函数，以add操作为例

```cpp
TfLiteStatus ParseAdd(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteAddParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteAddParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const AddOptions* schema_params = op->builtin_options_as_AddOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}
```

TfLiteAddParams是属于add的参数，不同算子参数不一，主要区分了激活函数和pot_scale。ActivationFunction是一个int32位数据，代表了不同的激活函数。ConvertAction只是做了一次内部数值的简单转换。最后使用AddNodeWithParameters创建新的node并添加到graph的nodes_and_registration_数组中，同时还会将node的index添加到execution_plan_中。

---
最后对所有的variables类型的tensor做处理，统一添加到subgraph中，整个流程就结束了

## 3. Signature、Metadata 与 Delegate 入口

`ParseSignatureDefs` 从文件的 signature key、subgraph_index 和具名 I/O 映射建立外部接口。这些名字用于调用签名，不等于必须覆盖原始 Tensor 名。构图结束前还导入 Metadata，按条件保存 resolver 的 lazy delegate creators，并执行调用方显式指定的委托。原稿中的 QS8/QU8 编译开关或 FP32 数量只决定是否获取默认工厂，不保证 XNNPACK 已创建或必然接管该图。显式 Delegate 被调用后用途仍可能未定，采用拥有型重载时实现会保留委托对象的所有权，不能因状态失败就假定马上释放，详见[第四篇](04c-delegate-setup.md)。

原稿末尾再次应用动态大张量分配选项，是为了按最终图状态决定哪些张量可从 arena 改为 dynamic；所示分支会跳过图输入，只处理达到阈值的 `kTfLiteArenaRw` 张量，并清空旧数据指针，不是“在 Builder 里立即分配所有大张量”。实际运行通常经历 **建图 → AllocateTensors（Prepare/分配）→ 填输入 → Invoke**。排查错误时先定位失败阶段：注册失败查 opcode/版本，Prepare 失败查形状、量化或 unresolved custom，Invoke 失败再追内核执行。动态形状、委托和多次 resize 可能让准备工作在后续时机重跑。

**Signature 的名字何时会改到 Runtime Tensor 上？** 一般的签名映射先记录参数名和 Tensor 编号；当前 Builder 只有在 `GetUseSignatureTensorNames()` 打开时，才遍历映射把名字写到对应子图的 Tensor 描述上。读原稿的「设置 Tensor 名称」时要保留这个条件；按 Signature 查名字和改写 Tensor 自己的名字不是同一件事。[Builder 名称分支](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L1026-L1058)

对照：[签名别名和默认工厂](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L1026-L1084)。

```cpp
  if (ParseSignatureDefs(model_->signature_defs(), tmp_interpreter.get()) !=
      kTfLiteOk) {
    return kTfLiteError;
  }
```

signaturedef是tflite模型对外提供的具名调用接口描述，主要包含四项，key代表调用什么功能，index代表了功能所对应的subgraph，input，output为两个map，保存参数名与对应tensor的映射

```cpp
  if (options_.GetUseSignatureTensorNames()) {
    for (const auto& signature_def : tmp_interpreter->signature_defs_) {
      Subgraph* subgraph =
          tmp_interpreter->subgraph(signature_def.subgraph_index);
      if (subgraph == nullptr) {
        continue;
      }
      for (const auto& [name, tensor_index] : signature_def.inputs) {
        TfLiteTensor* tensor = subgraph->tensor(tensor_index);
        if (tensor != nullptr) {
          tensor->name = name.c_str();
        }
      }
      for (const auto& [name, tensor_index] : signature_def.outputs) {
        TfLiteTensor* tensor = subgraph->tensor(tensor_index);
        if (tensor != nullptr) {
          tensor->name = name.c_str();
        }
      }
    }
  }
```

只有打开 `GetUseSignatureTensorNames()`，Builder 才会把签名映射中的名字写回对应 Runtime Tensor；按名字找到签名参数本身不要求打开这个选项。

```cpp
  if (tmp_interpreter->SetMetadata(metadata_) != kTfLiteOk) {
    return kTfLiteError;
  }

  if (ShouldCreateLazyDelegateProviders(num_fp32_tensors_)) {
    tmp_interpreter->lazy_delegate_providers_ =
        op_resolver_.GetDelegateCreators();
  }
```

```cpp
inline bool ShouldCreateLazyDelegateProviders(int num_fp32_tensors) {
#if defined(XNNPACK_DELEGATE_ENABLE_QS8) || defined(XNNPACK_DELEGATE_ENABLE_QU8)
  return true;
#else
  return num_fp32_tensors > 0;
#endif
}
```

如果开启了QS8,QU8或者存在fp32类型的tensor，则开启lazydelegatepoviders功能。lazydelegate指的是构建interpreter时，不直接创建delegate，而是将创建的函数作为指针添加到interpreter中，在需要创建delegate时再创建

```cpp
  if (telemetry_registered) {
    ParseConversionMetadata(telemetry_settings.get());
    tmp_interpreter->SetTelemetrySettings(std::move(telemetry_settings));
    tmp_interpreter->ReportTelemetrySettings(kTelemetryBuilderEventName);
  }

  if (TfLiteStatus status = ApplyDelegates(tmp_interpreter.get());
      status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to apply delegates.\n");
    return status;
  }
```

applydelegates本身比较复杂

### 显式委托与 Flex 的所有权差异

对照：[ApplyDelegates](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L898-L915)。

```cpp
TfLiteStatus InterpreterBuilder::ApplyDelegates(Interpreter* interpreter) {
  if (has_flex_op_) {
    if (Interpreter::TfLiteDelegatePtr flex_delegate = AcquireFlexDelegate()) {
      TF_LITE_ENSURE_STATUS(interpreter->ModifyGraphWithDelegateImpl(
          std::move(flex_delegate)));
    }
  }
  for (TfLiteDelegate* delegate : delegates_) {
    TF_LITE_ENSURE_STATUS(interpreter->ModifyGraphWithDelegateImpl(delegate));
  }
  return kTfLiteOk;
}
```

Flex 和普通外部委托最终都进入改图逻辑，但所有权不同：这里 Flex 经拥有型重载交给 Interpreter；外部裸指针不移交所有权。

```cpp
  template <typename Delegate, typename Deleter>
  inline TfLiteStatus ModifyGraphWithDelegateImpl(
      std::unique_ptr<Delegate, Deleter>&& delegate) {
    Deleter deleter = std::move(delegate.get_deleter());

    owned_delegates_.emplace_back(
        delegate.release(), [deleter](TfLiteDelegate* delegate_to_delete) {
          deleter(
              static_cast<typename std::unique_ptr<Delegate, Deleter>::pointer>(
                  delegate_to_delete));
        });
    return ModifyGraphWithDelegateImpl(owned_delegates_.back().get());
  }

    TfLiteStatus Interpreter::ModifyGraphWithDelegateImpl(
        TfLiteDelegate* delegate) {
    std::vector<int> all_subgraph_indices(subgraphs_.size());
    for (int i = 0; i < subgraphs_.size(); ++i) {
        all_subgraph_indices[i] = i;
    }
    return ModifyGraphWithDelegateImpl(delegate, all_subgraph_indices);
    }
```

显式 `delegates_` 为空时不会进入裸指针循环。Resolver 提供的默认工厂另走延迟创建，不要把它们混进这个列表。

```cpp
  if (options_.GetDynamicAllocationForLargeTensors()) {
    tmp_interpreter->ApplyOptionsImpl(&options_);
  }

  *interpreter = std::move(tmp_interpreter);
  return kTfLiteOk;
}
```

interpreteroptions在之前已经设置过一次，但是当初设置时还没有做parsetensor，所以对large tensor的优化无法执行，此处再调用一次执行优化，具体会调用到每个subgraph中的优化

### 大张量优化为何在构图末尾再做

对照：[OptimizeMemoryForLargeTensors](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2202-L2217)。

```cpp
void Subgraph::OptimizeMemoryForLargeTensors(
    int threshold) {

  for (size_t tensor_index = 0;
       tensor_index < context_.tensors_size;
       tensor_index++) {

    TfLiteTensor* tensor =
        &context_.tensors[tensor_index];

    if (tensor->bytes >= threshold &&
        tensor->allocation_type == kTfLiteArenaRw &&
        std::find(inputs_.begin(), inputs_.end(),
                  tensor_index) == inputs_.end()) {

      tensor->allocation_type = kTfLiteDynamic;
      tensor->data.raw = nullptr;
    }
  }
}
```

对于大小以及type符合要求tensor，type会从arenarw改为dynamic，使得large tensor在执行时可以动态分配内存。分配内存涉及到arena等相关内容，在相关章节说明。

至此，interpreter已经构建完成。

## 源码入口

代码对照版本：LiteRT [`e3c11f0`](https://github.com/google-ai-edge/LiteRT/tree/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62)；流程块是按固定版本整理的阅读稿；可以点对应行号看完整实现。

- [LiteRT CompiledModel 中的 InterpreterBuilder 使用](https://github.com/google-ai-edge/LiteRT/blob/main/litert/runtime/compiled_model.cc)
- [TensorFlow Lite InterpreterBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/interpreter_builder.cc)
- [运行时 Subgraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/subgraph.cc) · [OpResolver 实现](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.cc)
- [FlatBuffer 参数转换](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/flatbuffer_conversions.cc)

核对具体分支时，优先搜索本文列出的函数名。多轴量化、外部 buffer、遥测与委托工厂等是容易随 checkout 或编译开关变化的部分。
