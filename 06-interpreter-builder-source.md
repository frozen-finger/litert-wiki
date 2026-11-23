# LiteRT 源码精读（六）：InterpreterBuilder 原稿整理

[构建概览](03-interpreter-builder.md) · [下一篇：Resolver 与 CPU 内核](07-resolver-kernel-source.md)

沿原稿从构造入口跟到 Subgraph、Tensor、Node，再到 Signature 与 Delegate 入口。把 `Allocation`（文件字节）和 Runtime 张量内存分开看。

本文是逐段整理的原稿。长代码可展开，短代码直接显示；不同代码块可能位于同一函数的不同位置，不应拼成独立示例编译。上游链接用于对照固定快照，少数原稿中的版本差异以实际 checkout 为准。

已有 `.tflite` 时，通常不手工填充 `Interpreter`。`InterpreterBuilder` 负责校验模型、搭建子图、绑定 Tensor 和节点；真正的推理在分配并填入输入之后才开始。

## 1. Builder 的构造与入口检查

对照：[InterpreterBuilder 实现](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L220-L252)。

从 `FlatBufferModel` 构建 `InterpreterBuilder` 时，先把模型视图和创建运行时所需的依赖传给 Builder。

```cpp
InterpreterBuilder::InterpreterBuilder(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    const InterpreterOptions* options_experimental)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(model.error_reporter())),
      metadata_(model.ReadAllMetadata()),
      allocation_(model.allocation()) {
  if (options_experimental) {
    options_ = *options_experimental;
  }
}
```

创建interpreterbuilder需要之前传入以下的参数：

- model： 即load时获得的flatbuffermodel对象，一个用来代表tflite文件的RAII对象
- op_resolver: 接口，用来将文件中的op映射到实际可执行的算子
- error——reporter： 处理报错故障信息
其它依赖包括错误报告器、元信息和 `Allocation`。`Allocation` 是模型字节的底层存储，不是运行时输入的分配器。

初始化完成后，调用重载的 `operator()`，传入一个接收结果的 `std::unique_ptr<Interpreter>` 地址。不是先手动构造 Interpreter 再传进去。

```cpp
TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter) {
  if (!interpreter) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Null output pointer passed to InterpreterBuilder.");
    return kTfLiteError;
  }
  interpreter->reset();

  if (!model_) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Null pointer passed in as model.");
    return kTfLiteError;
  }

  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model_->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  if (BuildLocalIndexToRegistrationMapping() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Registration failed.\n");
    return kTfLiteError;
  }
```

先排除空的输出指针和空模型，再比较 `Model.version()` 与 `TFLITE_SCHEMA_VERSION`。这里检查的是模型 schema 的版本，不是生成它的 FlatBuffers 库版本。

## 2. 建立 opcode 到 Registration 的索引

对照：[注册索引与占位项](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L253-L295)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping() {
  TfLiteStatus status = kTfLiteOk;
  flatbuffer_op_index_to_registration_.clear();
  unresolved_custom_ops_.clear();

  auto opcodes = model_->operator_codes();
  if (!opcodes) {
    return status;
  }
  int num_custom_ops = 0;
  for (const OperatorCode* opcode : *opcodes) {
    if (GetBuiltinCode(opcode) == BuiltinOperator_CUSTOM) {
      num_custom_ops++;
    }
  }
  unresolved_custom_ops_.reserve(num_custom_ops);
  for (const OperatorCode* opcode : *opcodes) {
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver_, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      if (GetBuiltinCode(opcode) != BuiltinOperator_CUSTOM) {
        return status;
      }
      if (!opcode->custom_code()) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Operator with CUSTOM builtin_code has no custom_code.\n");
        return status;
      }
      const auto* op_name = opcode->custom_code()->c_str();
      unresolved_custom_ops_.push_back(CreateUnresolvedCustomOp(op_name));
      registration = &unresolved_custom_ops_.back();
      has_flex_op_ |= IsFlexOp(op_name);
      status = kTfLiteOk;
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}
```

</details>

这张映射表的下标与文件 `operator_codes` 保持一致，值为 `const TfLiteRegistration*`；`unresolved_custom_ops_` 则保存 Builder 自己拥有的 CUSTOM 占位对象。Registration 描述内核的版本、类型和生命周期回调。

`TfLiteOperator` 是对 C API 暴露的不透明对象。调用者通过创建与配置函数操作，而不是依赖结构体布局。具体 ABI 保证仍以 API 文档为准。

```cpp
TfLiteOperator* op = TfLiteOperatorCreate(
    kTfLiteBuiltinCustom,
    "MyCustomOp",
    1,
    nullptr);

TfLiteOperatorSetPrepareWithData(op, MyPrepare);
```

```cpp
  std::vector<const TfLiteRegistration*> flatbuffer_op_index_to_registration_;
  std::vector<TfLiteRegistration> unresolved_custom_ops_;
```

不过flatbuffer_op_index_to_registration_由于对应了已有的registration所有权并不在interpreterbuilder内所以使用了const指针而unresolved_custom_ops_的所有权属于interpreterbuilder
`OperatorCode` 对应 schema 中的 table。先统计 CUSTOM 数量，为占位容器 `reserve`，而不是提前 `resize`；后面会存放元素地址，避免扩容使地址失效。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus GetRegistrationFromOpCode(
    const OperatorCode* opcode, const OpResolver& op_resolver,
    ErrorReporter* error_reporter, const TfLiteRegistration** registration) {
  TfLiteStatus status = kTfLiteOk;
  *registration = nullptr;
  auto builtin_code = GetBuiltinCode(opcode);
  int version = opcode->version();

  if (builtin_code > BuiltinOperator_MAX) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Op builtin_code out of range: %d. Are you using old TFLite binary "
        "with newer model?",
        builtin_code);
    status = kTfLiteError;
  } else if (builtin_code != BuiltinOperator_CUSTOM) {
    *registration = op_resolver.FindOp(builtin_code, version);
    if (*registration == nullptr) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Didn't find op for builtin opcode '%s' version '%d'. "
          "An older version of this builtin might be supported. "
          "Are you using an old TFLite binary with a newer model?\n",
          EnumNameBuiltinOperator(builtin_code), version);
      status = kTfLiteError;
    }
  } else if (!opcode->custom_code()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Operator with CUSTOM builtin_code has no custom_code.\n");
    status = kTfLiteError;
  } else {
    const char* name = opcode->custom_code()->c_str();
    *registration = op_resolver.FindOp(name, version);
    if (*registration == nullptr) {
      status = kTfLiteError;
    }
  }
  return status;
}
```

</details>

GetRegistrationFromOpCode定义在op_resolver中，如果不为custom op则通过实现了op_resolver接口的对象的findop方法返回一个registration对象，如果是custom op则使用名称去resolver中查找有无对应的registration，在实际实践时，可能在interpreter构造前就已经通过resolver注册了一个自己实现的registration。关于resolver的具体实现放到resolver的章节之中

找不到有名称的 CUSTOM 时，可先保存占位 Registration。这给后续 delegate 处理留下机会，不代表该算子已有 CPU 实现。`Flex` 前缀标识 Select TF Ops 路径，仍须有相应的 Flex Delegate。

```cpp
  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();

  if (subgraphs->empty()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No subgraph in the model.\n");
    return kTfLiteError;
  }

  if (!buffers) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No buffers in the model.\n");
    return kTfLiteError;
  }

  auto tmp_interpreter = std::make_unique<Interpreter>(error_reporter_);
  if (subgraphs->size() > 1) {
    tmp_interpreter->AddSubgraphs(subgraphs->size() - 1);
  }
```

## 3. 建立 Interpreter 与子图

对照：[Interpreter 创建和子图导入](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L972-L1024)。

这部分创建了临时的interpreter并通过addsubgraphs向interpreter中注册对应数量的subgraph，interpreter的构造函数如下：

```cpp
Interpreter::Interpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
#if defined(TFLITE_IS_MOBILE_PLATFORM)
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#else
  TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#endif

  AddSubgraphs(1);
  context_ = primary_subgraph().context();

  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  own_external_cpu_backend_context_ =
      std::make_unique<ExternalCpuBackendContext>();
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();
}
```

### 3.1 子图初始化：哪些对象共享，哪些独有

对照：[Interpreter AddSubgraphs](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter.cc#L199-L216) · [Subgraph 构造](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L301-L380)。

addsubgraph会添加一个新构建的主subgraph，subgraph的构造函数如下：

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
Subgraph::Subgraph(ErrorReporter* error_reporter,
                   TfLiteExternalContext** external_contexts,
                   std::vector<std::unique_ptr<Subgraph>>* subgraphs,
                   resource::ResourceMap* resources,
                   resource::ResourceIDMap* resource_ids,
                   resource::InitializationStatusMap* initialization_status_map,
                   int subgraph_index)
    : external_contexts_(external_contexts),
      registration_externals_(new internal::OperatorsCache),
      error_reporter_(error_reporter),
      next_execution_plan_index_to_prepare_(0),
      next_execution_plan_index_to_plan_allocation_(0),
      subgraphs_(subgraphs),
      subgraph_index_(subgraph_index),
      resources_(resources),
      resource_ids_(resource_ids),
      initialization_status_map_(initialization_status_map),
      options_(nullptr) {
  context_.impl_ = static_cast<void*>(this);
  context_.ResizeTensor = ResizeTensor;
  context_.ReportError = ReportErrorC;
  context_.AddTensors = AddTensors;
  context_.tensors = nullptr;
  context_.tensors_size = 0;
  context_.allow_fp32_relax_to_fp16 = false;
  context_.recommended_num_threads = -1;
  context_.GetExternalContext = GetExternalContext;
  context_.SetExternalContext = SetExternalContext;
  context_.profiler = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = nullptr;
  context_.GetModelMetadata = GetModelMetadata;
#if defined(_WIN32)
  context_.TfLiteIntArrayCreate = TfLiteIntArrayCreate;
  context_.TfLiteIntArrayFree = TfLiteIntArrayFree;
#endif  // defined(_WIN32)

  tensors_.reserve(kTensorsReservedCapacity);
  nodes_and_registration_.reserve(kTensorsReservedCapacity);
  SwitchToKernelContext();
}
```

</details>

子图构造函数收到的七项参数：

| 参数 | 对应对象 | 用途 |
| --- | --- | --- |
| `error_reporter` | Interpreter 的错误报告器 | 统一报告运行时错误 |
| `external_contexts` | 外部 Context 数组 | 共享 CPU Backend 等运行环境 |
| `subgraphs` | Interpreter 的子图表 | 允许访问其它子图 |
| `resources` | 资源表 | 管理 Resource Variable 等资源 |
| `resource_ids` | 资源名字到 ID 的映射 | 按名字找到资源 |
| `initialization_status_map` | 初始化状态表 | 记录某个初始化过程是否完成 |
| `subgraph_index` | 当前编号，主图为 0 | 标识自身，不是资源表索引 |

前六项由 Interpreter 提供；子图还拥有自己的 `TfLiteContext`。`impl_` 保存当前子图地址，使 kernel 可通过 C 风格回调转回 C++ 实现。张量向量的初始预留容量不是模型能容纳的 Tensor 上限。
同一个 Context 可切换普通内核与委托所用的回调集；Delegate 准备期间，临时开放查计划和替换节点等回调。主图 Context 也是 Interpreter 常用的默认 Context，但其他子图有自己的 Context。外部 Context 枚举上限不是硬件数目。

ExternalCpuBackendContext继承自TfLiteExternalContext，用于标识和管理LiteRT runtime的外部共享资源，本身不保存CPU线程池，矩阵缓存或者硬件资源，真正管理CPU计算状态的为CpuBackendContext，有External内部的internal_backend_context指针指向，但在interpreter创建时实际还未拥有。

```cpp
void Interpreter::AddSubgraphs(int subgraphs_to_add,
                               int* first_new_subgraph_index) {
  const size_t base_index = subgraphs_.size();
  if (first_new_subgraph_index) *first_new_subgraph_index = base_index;

  subgraphs_.reserve(base_index + subgraphs_to_add);
  for (int i = 0; i < subgraphs_to_add; ++i) {
    Subgraph* subgraph = new Subgraph(
        error_reporter_, external_contexts_, &subgraphs_, &resources_,
        &resource_ids_, &initialization_status_map_, subgraphs_.size());
    subgraphs_.emplace_back(subgraph);
  }
}
```

Interpreter 初始化时已经创建主图。Builder 按序列化模型子图数补齐，其索引按模型顺序对应；支持特化的版本还可能建附加运行时子图。共享依赖不意味着各子图共享张量和节点。

```cpp
tmp_interpreter->SetNumThreads(num_threads_);

tmp_interpreter->ApplyOptionsImpl(&options_);

tmp_interpreter->SetProfilerImpl(
    tflite::profiling::MaybeCreatePlatformProfiler());
```

之后对创建出的interpreter中的一系列参数做了设置
线程数在子图建立后配置；`-1` 表示没有显式给出建议值，由运行时路径处理。
接着是options, 这个值同样默认为空，再不为空的情况下，每个subgraph会持有一个指向同一个option的指针，option内有以下的参数：

```cpp
preserve_all_tensors false 不强制保留全部中间 Tensor
ensure_dynamic_tensors_are_released false 不强制提前释放全部已不再使用的动态 Tensor
optimize_memory_for_large_tensors 0 不启用大 Tensor 的特殊动态分配优化
disable_delegate_clustering false 不主动禁用 Delegate 聚类优化
```

MaybeCreatePlatformProfiler按照平台创建了性能跟踪的工具

```cpp
std::unique_ptr<tflite::Profiler> MaybeCreatePlatformProfiler() {
#if defined(__ANDROID__)
  return MaybeCreateATraceProfiler();
#elif defined(SIGNPOST_PLATFORM_PROFILER)
  return MaybeCreateSignpostProfiler();
#elif defined(ENABLE_TFLITE_PERFETTO_PROFILER)
  return std::make_unique<tflite::profiling::PerfettoProfiler>();
#else
  return nullptr;
#endif
}
```

可选后端受平台和编译宏影响。这里展示的是平台 profiler 入口；不能从源码同时出现多个名字推断某个构建都启用了它们。

```cpp
  bool telemetry_registered = telemetry_profiler_ != nullptr;
  std::unique_ptr<TfLiteTelemetryInterpreterSettings> telemetry_settings;
  if (telemetry_registered) {
    tmp_interpreter->AddProfiler(std::move(telemetry_profiler_));
    telemetry_settings = std::make_unique<TfLiteTelemetryInterpreterSettings>();
    telemetry_settings->subgraph_infos.resize(subgraphs->size());
  }
```

与atrace等不同，telemetry_profiler属于遥测事件采集器，用于接收模型构建，执行状态，错误，算子执行时间等信息，addprofiler也不会像setprofilerimpl一样替代原有的profiler而是添加一个新的。如果外部调用方没有自建一个telemetry_profiler的情况下，同样该值会默认为空。

### 3.2 把文件子图导入已有运行时子图

对照：[BuildSubgraph](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L1092-L1139)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
    tflite::Subgraph* modified_subgraph =
        tmp_interpreter->subgraph(subgraph_index);
    modified_subgraph->allocation_ = allocation_;
    auto* subgraph_info =
        telemetry_registered
            ? &telemetry_settings->subgraph_infos[subgraph_index]
            : nullptr;
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    if (!tensors) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Did not get tensors in subgraph %d.\n",
                           subgraph_index);
      return kTfLiteError;
    }
    if (modified_subgraph->AddTensors(tensors->size()) != kTfLiteOk) {
      return kTfLiteError;
    }
    modified_subgraph->SetInputs(
        FlatBufferIntArrayToVector(subgraph->inputs()));
    modified_subgraph->SetOutputs(
        FlatBufferIntArrayToVector(subgraph->outputs()));

    if (ParseTensors(buffers, tensors, modified_subgraph, subgraph_info) !=
        kTfLiteOk)
      return kTfLiteError;
    if (operators && ParseNodes(operators, modified_subgraph) != kTfLiteOk)
      return kTfLiteError;

    std::vector<int> variables;
    for (int i = 0; i < modified_subgraph->tensors_size(); ++i) {
      auto* tensor = modified_subgraph->tensor(i);
      if (tensor->is_variable) {
        variables.push_back(i);
      }
    }
    modified_subgraph->SetVariables(std::move(variables));
    if (subgraph->name()) {
      modified_subgraph->SetName(subgraph->name()->c_str());
    }
  }
```

</details>

此处将model即flatbuffer中的subgraph信息导入到了之前创建的subgraph对象之中，如果telemetry不为空的情况下，subgraphinfo中会包含tensor对应的量化信息。

```cpp
TfLiteStatus Subgraph::AddTensors(int tensors_to_add,
                                  int* first_new_tensor_index) {
  const size_t base_index = tensors_.size();
  if (first_new_tensor_index) *first_new_tensor_index = base_index;
  if (tensors_to_add < 0) return kTfLiteError;
  tensors_.resize(tensors_.size() + tensors_to_add);
  for (size_t i = base_index; i < tensors_.size(); i++) {
    memset(&tensors_[i], 0, sizeof(tensors_[i]));
    tensors_[i].buffer_handle = kTfLiteNullBufferHandle;
  }
  context_.tensors = tensors_.data();
  context_.tensors_size = tensors_.size();
  return kTfLiteOk;
}
```
addtensors对subgraph中的tensors_大小和初始化内存大小以及指向buffer的指针做了初始化，这些值也会被存到每个subgraph的context中。

`SetInputs/SetOutputs` 先检查下标，随后移动有序的 Tensor 编号，而不是搬走 Tensor 本身。`ParseTensors` 与 `ParseNodes` 才配置已有张量槽和计算节点。

## 4. ParseTensors：从文件描述到运行时 Tensor

对照：[ParseTensors](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L766-L895)。

```cpp
TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Subgraph* subgraph, TfLiteTelemetrySubgraphInfo* subgraph_info) {
  TfLiteStatus status = kTfLiteOk;

  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };

  if (subgraph_info) {
    subgraph_info->quantizations.resize(tensors->size());
  }
```

以上为特殊情况的一些处理，如果telemetry不为空，这里会更新其中对应tensor量化的集合的大小，接下来的循环中会对所有的tensor一一进行构造

```cpp
  num_fp32_tensors_ = 0;
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());

    TfLiteType type;
    if (ConvertTensorType(tensor->type(), &type, error_reporter_) !=
        kTfLiteOk) {
      status = kTfLiteError;
      continue;
    }
    if (type == kTfLiteFloat32) {
      ++num_fp32_tensors_;
    }
```

将tensor的参数类型转换为了LiteRT对象使用的类型，表示为kTfLite+数值类型如Int32，同时用一个值记录所有float32的变量

### 4.1 查找常量字节与底层 Allocation

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      if (tensor->buffer() >= buffers->size()) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            i, tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        auto offset = buffer->offset();
        if (auto* array = buffer->data()) {
          *buffer_size = array->size();
          *buffer_data = reinterpret_cast<const char*>(array->data());
          return kTfLiteOk;
        } else if (offset > 1 && allocation_) {
          const size_t alloc_bytes = allocation_->bytes();
          const uint64_t size = buffer->size();
          if (offset > alloc_bytes || size > alloc_bytes - offset) {
            TF_LITE_REPORT_ERROR(
                error_reporter_,
                "Constant buffer %d specified an out of range offset.\n",
                tensor->buffer());
            return kTfLiteError;
          }
          *buffer_size = size;
          *buffer_data =
              reinterpret_cast<const char*>(allocation_->base()) + offset;
          return kTfLiteOk;
        }
      }
      return kTfLiteOk;
    };

    size_t buffer_size = 0;
    const char* buffer_ptr = nullptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));
```

</details>

`tensor->buffer()` 查模型级的 Buffer 表，不是 Subgraph 的表。这个 Builder 分支先看内嵌 `data()`，没有内嵌数据才尝试同文件追加区，并同时用 offset 和剩余字节验证范围。拿到的是字节地址，独立外部权重是另一种情况。

```cpp
    const auto* src_quantization = tensor->quantization();
    TfLiteQuantization quantization{};
    if (ParseQuantization(src_quantization, &quantization, dims,
                          tensors->size()) != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Tensor %d has invalid quantization parameters.", i);
      status = kTfLiteError;
    }
    if (subgraph_info) subgraph_info->quantizations[i] = quantization;
```

parseQuantization本身的行为也非常复杂

### 4.2 量化：MultiAxis、Blockwise 与 Affine

从完整解析分支读，先清空默认量化描述，再判断 details union；最后才走普通 Affine。记录下 scale 在各分支里到底是 Tensor 编号还是数值数组，这是理解 `void* params` 的前提。

对照：[ParseQuantization](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L411-L662)。

```cpp
TfLiteStatus InterpreterBuilder::ParseQuantization(
    const QuantizationParameters* src_quantization,
    TfLiteQuantization* quantization, const std::vector<int>& dims,
    size_t num_tensors) {
  quantization->type = kTfLiteNoQuantization;
  quantization->params = nullptr;
  if (!src_quantization) {
    return kTfLiteOk;
  }

  auto validate_tensor_index = [&](int32_t tensor_index, const char* name,
                                   bool optional) {
    if (optional && tensor_index == -1) {
      return kTfLiteOk;
    }
    if (tensor_index < 0 || static_cast<size_t>(tensor_index) >= num_tensors) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "MultiAxisQuantization %s tensor index must be in range [0, %d)%s. "
          "Was %d.",
          name, static_cast<int>(num_tensors),
          optional ? ", or -1 if optional" : "", tensor_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  };
```

首先做了次检查，判断当前的tensor是不是合法
首先判断使用MultiAxis量化的情况
MultiAxis 用轴列表描述参数如何对应多维切片；比例和零点是其它 Tensor 的编号，不是这里直接存放的数值数组。该实现接受非空的轴列表，不要求至少两个轴；block size 不能为负。

```cpp
  if (src_quantization->details_type() ==
      QuantizationDetails_MultiAxisQuantization) {
    auto* src_quant = src_quantization->details_as_MultiAxisQuantization();
    if (!src_quant || !src_quant->quantized_dimensions()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Invalid MultiAxisQuantization parameters.");
      return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(
        validate_tensor_index(src_quant->scales(), "scales", false));
    TF_LITE_ENSURE_STATUS(
        validate_tensor_index(src_quant->zero_points(), "zero_points", true));
    if (src_quant->block_size() < 0) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "MultiAxisQuantization block_size must be non-negative. Was %d.",
          src_quant->block_size());
      return kTfLiteError;
    }

    const auto* quantized_dimensions = src_quant->quantized_dimensions();
    if (quantized_dimensions->empty()) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "MultiAxisQuantization must have at least one quantized dimension.");
      return kTfLiteError;
    }
```

判断是否有需要的量化维度数据，scale, zero_points, block是否合法

```cpp
    for (int i = 0; i < quantized_dimensions->size(); ++i) {
      const int32_t quantized_dimension = quantized_dimensions->Get(i);
      if (quantized_dimension < 0 ||
          (!dims.empty() &&
           static_cast<size_t>(quantized_dimension) >= dims.size())) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "MultiAxisQuantization quantized_dimensions must be in range [0, "
            "%d). Value %d at index %d is invalid.",
            static_cast<int>(dims.size()), quantized_dimension, i);
        return kTfLiteError;
      }
    }
```

所有的维度数据必须是[0, 维度数量）之间

```cpp
    auto* multi_axis_quantization =
        reinterpret_cast<TfLiteMultiAxisQuantization*>(
            malloc(sizeof(TfLiteMultiAxisQuantization)));
    if (!multi_axis_quantization) {
      return kTfLiteError;
    }
    multi_axis_quantization->scales = src_quant->scales();
    multi_axis_quantization->zero_points = src_quant->zero_points();
    multi_axis_quantization->blocksize = src_quant->block_size();
    multi_axis_quantization->quantized_dimensions =
        TfLiteIntArrayCreate(static_cast<int>(quantized_dimensions->size()));
    if (!multi_axis_quantization->quantized_dimensions) {
      free(multi_axis_quantization);
      return kTfLiteError;
    }
    for (int i = 0; i < quantized_dimensions->size(); ++i) {
      multi_axis_quantization->quantized_dimensions->data[i] =
          quantized_dimensions->Get(i);
    }
    quantization->type = kTfLiteMultiAxisQuantization;
    quantization->params = reinterpret_cast<void*>(multi_axis_quantization);
    return kTfLiteOk;
  }
```

将multiaxis需要的参数配置到了TfliteQuantization对象中，最后保存为了void*指针

接着处理blockwise量化的情况，blockwise的具体参数和内容可以参考模型加载部分的文档

```cpp
  if (src_quantization->details_type() ==
      QuantizationDetails_BlockwiseQuantization) {
    auto* src_quant = src_quantization->details_as_BlockwiseQuantization();
    if (!src_quant) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Blockwise quantization details are missing.");
      return kTfLiteError;
    }
```

quantization不合法的情况

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
    const auto* block_shape = src_quant->block_shape();
    if (block_shape && !block_shape->empty()) {
      if (!dims.empty() &&
          static_cast<size_t>(block_shape->size()) != dims.size()) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "BlockwiseQuantization block_shape must have one entry per "
            "dimension. Expected %d entries, got %d.",
            static_cast<int>(dims.size()), block_shape->size());
        return kTfLiteError;
      }
      for (int i = 0; i < block_shape->size(); ++i) {
        if (block_shape->Get(i) <= 0) {
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "BlockwiseQuantization block_shape entries must "
                               "be positive. Value %d at index %d is invalid.",
                               block_shape->Get(i), i);
          return kTfLiteError;
        }
      }
      auto* blockwise_quantization_v2 =
          reinterpret_cast<TfLiteBlockwiseQuantizationV2*>(
              malloc(sizeof(TfLiteBlockwiseQuantizationV2)));
      if (!blockwise_quantization_v2) {
        return kTfLiteError;
      }
      blockwise_quantization_v2->scale = src_quant->scales();
      blockwise_quantization_v2->zero_point = src_quant->zero_points();
      blockwise_quantization_v2->quantized_dimension =
          src_quantization->quantized_dimension();
      blockwise_quantization_v2->blocksize = src_quant->block_size();
      blockwise_quantization_v2->block_shape =
          TfLiteIntArrayCreate(static_cast<int>(block_shape->size()));
      if (!blockwise_quantization_v2->block_shape) {
        free(blockwise_quantization_v2);
        return kTfLiteError;
      }
      for (int i = 0; i < block_shape->size(); ++i) {
        blockwise_quantization_v2->block_shape->data[i] = block_shape->Get(i);
      }
      quantization->type = kTfLiteBlockwiseQuantizationV2;
      quantization->params = reinterpret_cast<void*>(blockwise_quantization_v2);
      return kTfLiteOk;
    }
```

</details>

第一种情况为使用blockshape的情况
blockshape参数的形状应该与tensor的shape的形状一致，每一个位置对应了block在该维度下占有多少个元素，这个数必须为正数。同样对应的数值会被保存到以void*指针形式指向的TfLiteBlockwiseQuantizationV2对象下

```cpp
    auto* blockwise_quantization_v1 =
        reinterpret_cast<TfLiteBlockwiseQuantization*>(
            malloc(sizeof(TfLiteBlockwiseQuantization)));
    if (!blockwise_quantization_v1) {
      return kTfLiteError;
    }
    blockwise_quantization_v1->scale = src_quant->scales();
    blockwise_quantization_v1->zero_point = src_quant->zero_points();
    blockwise_quantization_v1->quantized_dimension =
        src_quantization->quantized_dimension();
    blockwise_quantization_v1->blocksize = src_quant->block_size();
    quantization->type = kTfLiteBlockwiseQuantization;
    quantization->params = reinterpret_cast<void*>(blockwise_quantization_v1);
    return kTfLiteOk;
  }
```

第二种为使用blcoksize无blockshape的情况，blocksize根据quantizaddimension制定的维度为基准，设定了每个block共有多少个元素

```cpp
  if (!src_quantization->scale() || src_quantization->scale()->empty()) {
    return kTfLiteOk;
  }
  if (!src_quantization->zero_point()) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Quantization parameters has non-null scale but null zero_point.");
    return kTfLiteError;
  }

  if (src_quantization->scale()->size() !=
      src_quantization->zero_point()->size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "QuantizationParam has %d zero_point values and %d scale values. "
        "Must "
        "have same number.",
        src_quantization->zero_point()->size(),
        src_quantization->scale()->size());
    return kTfLiteError;
  }
```

普通 Affine 既可能是单 Tensor 的比例，也可能按选定的轴给每个通道一个比例。处理时核对数组与轴长度，不应把 scale 与 zero point 当成同一含义；相同 zero-point 的压缩另有条件。

```cpp
  const size_t num_scales = src_quantization->scale()->size();
  const size_t num_zero_points = src_quantization->zero_point()->size();
  bool all_zero_points_same = false;
  int32_t zero_point = 0;
  if (options_.GetCompressQuantizationZeroPoints() && num_zero_points > 0) {
    all_zero_points_same = true;
    zero_point = src_quantization->zero_point()->data()[0];
    for (int i = 1; i < num_zero_points && all_zero_points_same; ++i) {
      all_zero_points_same &=
          (src_quantization->zero_point()->data()[i] == zero_point);
    }
  }
```

如果事先开启了压缩量化zeropoints的功能，则会比较是否所有的zeropoint值都一样

```cpp
  if (src_quantization->quantized_dimension() < 0 ||
      (!dims.empty() &&
       src_quantization->quantized_dimension() >= dims.size())) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "quantized_dimension must be in range [0, %d). Was %d.", dims.size(),
        src_quantization->quantized_dimension());
    return kTfLiteError;
  }

  if (num_scales != 1 &&
      (!dims.empty() &&
       num_scales != dims[src_quantization->quantized_dimension()])) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "num_scales must be 1 for per-layer quantization, or %d for per-axis "
        "quantization, but got %d.",
        dims[src_quantization->quantized_dimension()], num_scales);
    return kTfLiteError;
  }
```

一些合法性检查

```cpp
  quantization->type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(num_scales);
  for (size_t i = 0; i < num_scales; ++i) {
    affine_quantization->scale->data[i] = src_quantization->scale()->Get(i);
  }
  if (all_zero_points_same) {
    affine_quantization->zero_point = TfLiteIntArrayCreate(1);
    affine_quantization->zero_point->data[0] = zero_point;
  } else {
    affine_quantization->zero_point = TfLiteIntArrayCreate(num_scales);
    for (size_t i = 0; i < num_scales; ++i) {
      affine_quantization->zero_point->data[i] =
          src_quantization->zero_point()->Get(i);
    }
  }
  affine_quantization->quantized_dimension =
      src_quantization->quantized_dimension();
  quantization->params = reinterpret_cast<void*>(affine_quantization);
  return kTfLiteOk;
}
```

将参数都配置到了quantization中
至此parsequantization的流程全部结束了，回到parsetensor之中。

### 4.3 shape_signature、只读和读写分叉

```cpp
    std::vector<int> dims_signature = {};
    if (tensor->shape_signature()) {
      dims_signature = FlatBufferIntArrayToVector(tensor->shape_signature());
    }
```

shape_signature被用于说明tensor某一维是固定的或是不固定的，不固定由-1表示

```cpp
    bool is_variable = tensor->is_variable();
    if (buffer_ptr) {
      if (is_variable || tensor->external_buffer() != 0) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Tensor %d is a variable tensor with buffer or external buffer. "
            "It's not supported now.\n",
            i);
        status = kTfLiteError;
      }

      const auto* src_sparsity = tensor->sparsity();
      TfLiteSparsity* sparsity = nullptr;
      if (ParseSparsity(src_sparsity, &sparsity) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d has invalid sparsity parameters.", i);
        status = kTfLiteError;
      }

      if (subgraph->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_, sparsity,
              /*buffer_identifier=*/tensor->buffer(),
              /*external_buffer_id=*/tensor->external_buffer()) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d is invalidly specified in schema.\n",
                             i);
        status = kTfLiteError;
      }
```

这里匹配了tensor具体对内存的使用方式，当buffer_ptr存在时即有固定的数据，则可变的tensor不可接受.对于稀疏tesnor来说，目前也仅接受有固定数据的情况，parseSparsity的细节如下：

### 4.4 解析稀疏 Tensor

对照：[ParseSparsity](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L664-L729)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus InterpreterBuilder::ParseSparsity(
    const SparsityParameters* src_sparsity, TfLiteSparsity** sparsity_ptr) {
  if (!src_sparsity) {
    return kTfLiteOk;
  }

  if (src_sparsity->traversal_order() == nullptr ||
      src_sparsity->dim_metadata() == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Invalid sparsity parameter.");
    return kTfLiteError;
  }

  auto* sparsity =
      reinterpret_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  memset(sparsity, 0, sizeof(TfLiteSparsity));
  *sparsity_ptr = sparsity;

  const size_t traversal_order_size = src_sparsity->traversal_order()->size();
  sparsity->traversal_order = TfLiteIntArrayCreate(traversal_order_size);
  for (int i = 0; i < traversal_order_size; i++) {
    sparsity->traversal_order->data[i] =
        src_sparsity->traversal_order()->Get(i);
  }

  if (src_sparsity->block_map()) {
    const size_t block_map_size = src_sparsity->block_map()->size();
    sparsity->block_map = TfLiteIntArrayCreate(block_map_size);
    for (int i = 0; i < block_map_size; i++) {
      sparsity->block_map->data[i] = src_sparsity->block_map()->Get(i);
    }
  }

  const size_t dim_metadata_size = src_sparsity->dim_metadata()->size();
  sparsity->dim_metadata_size = dim_metadata_size;
  sparsity->dim_metadata = reinterpret_cast<TfLiteDimensionMetadata*>(
      malloc(dim_metadata_size * sizeof(TfLiteDimensionMetadata)));
  memset(sparsity->dim_metadata, 0,
         dim_metadata_size * sizeof(TfLiteDimensionMetadata));

  for (int i = 0; i < dim_metadata_size; i++) {
    const auto* src_metadata = src_sparsity->dim_metadata()->Get(i);
    if (src_metadata->format() != DimensionType_DENSE &&
        src_metadata->format() != DimensionType_SPARSE_CSR) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "The %dth dimension has unknown type: %d.", i,
                           src_metadata->format());
      return kTfLiteError;
    }
    auto* tgt_metadata = &sparsity->dim_metadata[i];

    tgt_metadata->format =
        static_cast<TfLiteDimensionType>(src_metadata->format());

    if (tgt_metadata->format == kTfLiteDimDense) {
      tgt_metadata->dense_size = src_metadata->dense_size();
    } else {
      if (ParseSparseIndexVector(src_metadata, tgt_metadata) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "The %dth sparse dimension has invalid parameters.", i);
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}
```

</details>

处理了blocksparsity的相关参数，blocksparsity会将原始的tensor进行按块进行划分，blockmap决定了分块之后traversal_order中多出的维度如何与原始的维度相对应（多出的维度为块内维度），dim_metadata则决定了每个维度下是否有由于空值而被裁去的block，分为两种情况，如果为dense则代表当前维度下没有被裁掉的空block且该维度大小为densesize，否则为CSR形式，记录了该维度下有哪些坐标是存在的。ParseSparseIndexVector负责解析CSR的两种数组，针对数组中可能使用Int8, 16, 32的三种可能形式转化为同样的TfLiteIntArray(一般为int32)

有可用的常量字节时按只读方式配置，否则按读写描述配置。这里是 Runtime 的存储和生命周期分类，不是给任意地址设置操作系统页权限。

### 4.5 只读参数的所有权、字节数和对齐

对照：[SetTensorParametersReadOnly](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1972-L2093)。

```cpp
TfLiteStatus Subgraph::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t ndims,
    const int* dims, TfLiteQuantization quantization, const char* buffer,
    size_t bytes, const Allocation* allocation, TfLiteSparsity* sparsity,
    const size_t buffer_identifier, const size_t external_buffer_id) {
  ScopedTfLiteQuantization scoped_quantization(&quantization);
  ScopedTfLiteSparsity scoped_sparsity(sparsity);
```

此处用带释放器的作用域句柄管理量化、稀疏参数所指向的堆对象。错误退出由句柄清理，成功时交给 Tensor；被跟踪的是堆资源，不是对栈变量执行 `delete`。

```cpp
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadOnly is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);

  const bool has_external_buffer =
      external_buffer_id != 0 &&
      external_buffer_id != kTfLiteNoBufferIdentifier;
  if (type != kTfLiteString && type != kTfLiteResource &&
      type != kTfLiteVariant && sparsity == nullptr && !has_external_buffer) {
    size_t required_bytes;
    TF_LITE_ENSURE_OK(
        &context_,
        tflite::BytesRequired(type, dims, ndims, &required_bytes, &context_));
    TF_LITE_ENSURE(&context_, required_bytes <= bytes);
  }
```

对于tensor类型为可变大小的情况，由于内存大小不能确定都会跳过，但会检查buffersize是否满足按照dim和数据类型计算出的理论内存大小

```cpp
  if (buffer != nullptr && bytes > 0 &&
      reinterpret_cast<uintptr_t>(buffer) % GetTypeAlignment(type) != 0) {
    TFLITE_LOG_PROD_ONCE(
        tflite::TFLITE_LOG_WARNING,
        "Read-only tensor #%d (%s) has an unaligned buffer (%p, required "
        "alignment %zu); copying to an aligned heap buffer. Re-export the "
        "model to align buffers in the FlatBuffer and enable zero-copy mmap.",
        tensor_index, name ? name : "<unnamed>", buffer,
        GetTypeAlignment(type));
    constexpr size_t kExtraAllocationBytes = 16;
    if (bytes > std::numeric_limits<size_t>::max() - kExtraAllocationBytes) {
      return kTfLiteError;
    }
    std::unique_ptr<char, FreeDeleter> aligned_buffer(
        static_cast<char*>(malloc(bytes + kExtraAllocationBytes)));
    TF_LITE_ENSURE(&context_, aligned_buffer != nullptr);
    memcpy(aligned_buffer.get(), buffer, bytes);
    memset(aligned_buffer.get() + bytes, 0, kExtraAllocationBytes);
    buffer = aligned_buffer.get();
    allocation = nullptr;
    aligned_ro_buffers_.push_back(std::move(aligned_buffer));
  }
```

未对齐的常量在这个快照中会复制到堆缓冲区，尾部额外补 16 个零字节，配合 XNNPACK 的 SIMD 读取约定。副本由 `aligned_ro_buffers_` 持有，Tensor 原来的 Allocation 关联被清掉。即使仍打 `kTfLiteMmapRo` 标签，也不能说它仍从 mmap 零拷贝读取。对齐的作用可以从三方面理解：

- 满足 CPU 对特定数据类型的内存访问要求，避免某些架构上发生异常。
- 满足 C/C++ 的对象对齐规则，避免未定义行为（Undefined Behavior）。
- 提高内存访问效率，减少某些情况下额外的内存读取操作

```cpp
  TfLiteTensor& tensor = context_.tensors[tensor_index];
  if (type == tensor.type &&
      EqualArrayAndTfLiteIntArray(tensor.dims, ndims, dims)) {
    TfLiteTensorDataFreeWithAllocator(&tensor, allocator_);
    TfLiteQuantizationFree(&tensor.quantization);
    tensor.data.raw = const_cast<char*>(buffer);
    if (!tensor.dims) tensor.dims = ConvertArrayToTfLiteIntArray(ndims, dims);
    tensor.params = GetLegacyQuantization(quantization);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
    tensor.allocation_type = kTfLiteMmapRo;
    tensor.allocation = allocation;
  } else {
    state_ = kStateUninvokable;
    TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(ndims, dims),
                      GetLegacyQuantization(quantization),
                      const_cast<char*>(buffer), bytes, kTfLiteMmapRo,
                      allocation, false, &tensor);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
  }
```

类型和维度一致时可走快速配置，否则将图标记为 `kStateUninvokable` 后重设 Tensor。`release()` 交接参数对象的所有权，不是立即释放对象；`kStateUninvokable` 是图的状态，而不是某个 Tensor 的状态。

```cpp
  if (buffer_identifier != kTfLiteNoBufferIdentifier) {
    tensor_buffer_identifiers_[tensor_index] = buffer_identifier;
  }
  if (external_buffer_id != kTfLiteNoBufferIdentifier &&
      external_buffer_id != 0) {
    tensor_external_buffer_ids_[tensor_index] = external_buffer_id;
  }
  return kTfLiteOk;
}
```

将tensor index与buffer的对应关系保存到subgraph中，此处的bufferid指的是tflite中的buffers数组的索引

### 4.6 读写 Tensor 的分配策略

对照：[SetTensorParametersReadWrite](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2094-L2149)。

```cpp
    } else {
      if (subgraph->SetTensorParametersReadWrite(
              i, type, get_name(tensor), dims, quantization, is_variable,
              dims_signature, tensor->external_buffer()) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d is invalidly specified in schema.\n",
                             i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}
```

readwrite与readonly相比主要只有下面一段有明显区别，除此之外bufferid只有外部id以及没有对sparsity的处理

```cpp
  TfLiteAllocationType allocation_type = kTfLiteArenaRw;
  if (type == kTfLiteString || type == kTfLiteResource ||
      type == kTfLiteVariant) {
    if (is_variable) {
      ReportError("String variable tensor isn't supported.");
      return kTfLiteError;
    }
    allocation_type = kTfLiteDynamic;
  } else if (is_variable) {
    allocation_type = kTfLiteArenaRwPersistent;
  }
```

这里决定了不同类型的tensor在runtime时会怎么选择内存生命周期策略，persistent代表了tensor的内存区域需要被保留不能被覆盖使用，具体留到arena相关得到部分

到此为止tensor的解析就已经结束了，进入node也就是operator的解析环节

## 5. ParseNodes：解析参数并登记执行节点

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

### 5.1 跟一遍 ADD 的参数转换

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

## 6. Signature、Metadata 与委托

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

### 6.1 显式委托与 Flex 的所有权差异

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

### 6.2 大张量优化为何在构图末尾再做

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
