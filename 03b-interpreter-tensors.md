# InterpreterBuilder（二）：Tensor、权重与量化

上一篇：[Builder（一）](03-interpreter-builder.md) · [目录](README.md) · 下一篇：[Builder（三）](03c-interpreter-nodes.md)

上一节已经建立 Tensor 槽位和图的输入、输出编号。这一篇接着看 `ParseTensors` 怎样把每份描述填到槽位里，并区分常量数据、可写张量和真正分配好的运行时缓冲区。

跟一张 Tensor 时可以依次看三件事：**描述是什么**（类型、形状、量化），**数据从哪来**（模型常量还是运行时填入），**什么时候才能拿到地址**（导入时借用，或运行时另行分配）。这几个问题分别出现在 `ParseTensors()`、只读/读写分支和后续内存规划中。

仍用 [ADD 的三个槽位](01-model-format.md#32-用一个-add-把索引走通)对照：T0/T1/T2 都引用空 Buffer 0，因此此时只能填入 float32 和形状等描述；只有把 Y 改成文件常量，T1 才改走下文的只读配置。

## 1. ParseTensors 怎样逐项处理

先 `AddTensors` 建张量槽位并让 Context 能定位它们，再设图的输入、输出索引，最后逐个配置这些槽位。AddTensors 扩的是 `TfLiteTensor` 描述对象数组，不等于为所有张量的数据分配内存。Builder 把 schema 的 `TensorType` 映射到运行时 `TfLiteType`（例如 `INT32 → kTfLiteInt32`），并统计浮点张量，供后面的默认委托工厂判断。

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

若提供 telemetry，先为每张 Tensor 的量化信息预留记录位置。接下来才遍历文件中的 Tensor，逐项填入运行时槽位。

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

## 2. 常量字节从哪来

**常量地址是怎样找到的。** 在 [`ParseTensors()` 里面的 `get_readonly_data`](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L800-L834) 单独跟一遍：先把数据指针置空，编号 0 不取常量；其余编号查的是**模型级** Buffer 表，不是子图的 Tensor 表。此处优先用 `Buffer.data`；没有内嵌向量，才在 `offset > 1` 且有 Allocation 时尝试同文件追加区，用 `size <= 总长 - offset` 防止越界。第二篇 LiteRT Loader 的 `ReadBuffer()` 则优先判断非零 offset。两条路径不要合写成一段规则。

`shape` 给出当前形状，例如 `[1, 224, 224, 3]`；`shape_signature=[-1,224,224,3]` 说明 batch 维允许调整。两者一起看才能解释为什么刚加载有一个明确大小，之后还可以 Resize。解析过程中统计 FP32 Tensor 数还有一个后续用途：决定是否登记 lazy Delegate 工厂，计数不是 XNNPACK 的逐节点支持结果。

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

## 3. MultiAxis、Blockwise 与 Affine 量化

[MultiAxis 校验与参数对象](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L411-L502) · [Blockwise 两套运行时结构](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L504-L577)

| 分支 | 先检查什么 | 最终写入什么 |
| --- | --- | --- |
| 无量化描述 | 无描述时直接沿默认类型返回 | `kTfLiteNoQuantization`、空参数 |
| MultiAxis | Scale Tensor 编号必须有效；可选的 zero-point Tensor 允许用 `-1`；block size 不能为负；维度列表不能为空，逐项核对轴编号 | `TfLiteMultiAxisQuantization` 保存两个 Tensor 编号、block size 和复制后的轴列表。语义可覆盖多个轴，不要把语义写成这里强制至少两个轴。 |
| Blockwise V2 | 非空 `block_shape` 优先；已知 rank 时长度须与 rank 相符，每项必须大于零；有问题时报错，不退回 V1 | `TfLiteBlockwiseQuantizationV2` 保存比例/零点 Tensor 编号、量化维度、旧 block size 字段与 block shape。 |
| Blockwise V1 | 没有非空 `block_shape` 才走旧路径 | 保存比例/零点编号、量化维度和 block size。 |
| 普通仿射 | 不属于以上分支才读通用 scale/zero-point 列表 | 详见下面按单 tensor 或单轴量化的检查。 |

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

接下来是 Blockwise Quantization。沿用 [模型格式里的分块例子](01-model-format.md#42-blockwisequantization)：先问每组参数覆盖哪些元素，再看运行时怎么保存参数 Tensor 的编号。不要把编号当作已经读出的 scale 数值。

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

若启用了零点压缩，并且所有零点相等，可以只存一个零点。这省的是重复的量化元数据，不会改写权重数值或张量形状。

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

## 4. shape_signature 与读写两条路径

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

这一步按能否取得常量数据指针分流。已有固定字节的 Tensor 走只读配置，不能同时标为 variable；该版本的 sparsity 导入也只支持固定数据。下面的 `ParseSparsity()` 读的是压缩值对应的坐标描述。

```text
读取可能存在的常量地址和大小
解析量化参数与 shape_signature
if 有常量地址:
    核对 variable 和 external buffer 的限制
    解析稀疏元数据；配置只读描述和原始 Allocation
else:
    配置读写描述，并记录 variable、shape_signature 和外部 Buffer ID
遇到无效参数时记录构建错误
```

1. `get_readonly_data()` 先解析模型字节。随后 `ParseQuantization()` 解析量化参数，`shape_signature` 单独读入；已知形状与允许动态调整的维度不是同一个字段。
2. 有常量字节走 `SetTensorParametersReadOnly()`，传入字节范围、底层 Allocation、稀疏信息和 Buffer 标识。前面还会检查与 variable、external buffer 等组合是否兼容。
3. 没有这段常量字节走 `SetTensorParametersReadWrite()`，登记形状、variable 和动态维度等描述。配置张量不等于所有数据已分配；继续到 `AllocateTensors()` 看 Prepare 和实际分配。

这里的「只读」是 Runtime 按常量管理数据，不等于读取地址一定来自 `mmap`。一个容易漏掉的例外是：如果常量首地址不满足类型的对齐，`SetTensorParametersReadOnly()` 会复制到堆缓冲区，由 `aligned_ro_buffers_` 持有，并清掉原 `allocation` 关联；这条分支仍把 Tensor 标为 `kTfLiteMmapRo`。该快照还在尾部补 16 个零字节，配合 XNNPACK 的 SIMD 读取约定。要判断某个权重是否实际零拷贝，不能只看 allocation 类型，还得看指针和这个分支。[对齐与只读标记](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2023-L2068)。

可用「数据要活多久」来区分读写 Tensor：普通定长 Tensor 默认走 `kTfLiteArenaRw`；支持的定长 variable 走 `kTfLiteArenaRwPersistent`，不能像一步用完的中间结果那样复用它的存储；String / Resource / Variant 则另走动态分支。这里挑选的是分配策略，不是在 Builder 里已经给每个 Tensor 分配到最终地址。[读写分配分支](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2112-L2133)。

## 5. 稀疏 Tensor：保存的是编码结构

`ParseSparsity` 处理的是“权重值以压缩形式保存，怎么找回坐标”。`traversal_order` 指定遍历维度顺序；分块时附加块内维度，`block_map` 把这些维度映射回原张量轴。`dim_metadata` 逐层描述 DENSE（坐标全存在，按 dense size 遍历）或 SPARSE_CSR（按 segments 划出每个父位置对应的 indices 范围）。例如一层的 `segments=[0, 2, 3]` 与 `indices=[1, 3, 2]`，表示第一个父位置下有坐标 1、3，第二个父位置下有坐标 2。索引序列可有 8、16、32 位编码，原稿的解析器统一转为 `TfLiteIntArray`。这里没有把稀疏权重展开成密集矩阵；运行时保有压缩值和元数据。原稿路径的稀疏导入用于固定数据，不应假设可写路径能接受同样的 sparsity 参数。

量化结构和稀疏元数据由临时 RAII 封装而不是因为“局部变量不能 delete”。这类结构内部可能有单独申请的内存；只有在所有权成功交接时才 `release`，失败路径让相应 deleter 清理，模型字节借用与这些结构的所有权不能混为一谈。

从 [`ParseSparsity()`](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L664-L729) 往回看，`traversal_order` 是遍历顺序，`block_map` 是**块内维度与原维度的对应表**，各维的 `dim_metadata` 则回答「这一维怎么编码」。Dense 只保存大小；CSR 分别保存 segments（每个父位置的索引区间）与 indices（实际出现的子坐标）。例如 segments=[0,2,3]、indices=[1,4,2]，表示第一个父位置有子坐标 1、4，第二个只有 2，不是三项数据值。[稀疏索引的类型转换](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter_builder.cc#L105-L158)把 UInt8、UInt16 或 Int32 的编码整理为运行时的 `TfLiteIntArray`。Builder 解析的是已有压缩描述；这段不会现场删掉零元素。

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

## 6. 只读内存、对齐与所有权

普通密集常量会核对理论字节数不大于实际 Buffer；字符串、Resource、Variant、稀疏和外部数据不能套同一公式。

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

子图在这里记录 Tensor 编号到文件级 `Model.buffers` 编号的映射。这个 `buffer_id` 不是上篇 LiteRT IR 的 `BufferManager` 句柄。

## 7. 读写 Tensor 的分配策略

| 导入路径 | 可能的分配类型 | 含义 |
|---|---|---|
| 可直接引用的常量数据 | `kTfLiteMmapRo` | 运行时将其视为只读；名字并不保证指针一定来自 mmap |
| 普通输入、中间值、输出 | `kTfLiteArenaRw` | 通常由 arena 按生命周期计划分配 |
| 可保持状态的定长变量 | `kTfLiteArenaRwPersistent` | 不按普通临时值复用，跨调用保留，仍可按规定重置 |
| string/resource/variant 等特例 | `kTfLiteDynamic` | 走动态分配路径；支持的变量组合以实现为准 |

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

读写分支没有内置常量地址，也不解析只读路径的 sparsity 元数据；调用它时传的是 `external_buffer` 编号。代码接着选择日后怎么分配，下面三类 Tensor 的区别主要在数据寿命。

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

`Persistent` 的重点是这块存储还要留给后续调用，不能按普通中间值处理。这时只是选定分配类型；地址如何落实，见 [Prepare 与内存分配](04b-interpreter-prepare.md#3-allocatetensors-暂停后invoke-怎样继续)。

现在 Tensor 槽里已经有了数据描述和分配策略。接着读 [算子节点与 Signature](03c-interpreter-nodes.md)，看 `ParseNodes()` 怎样把这些槽接进执行计划。
