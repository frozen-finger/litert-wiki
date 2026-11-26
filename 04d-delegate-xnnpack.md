# Delegate（二）：XNNPACK 如何选点与展开权重

上一篇：[Delegate（一）](04c-delegate-setup.md) · [目录](README.md) · 下一篇：[Delegate（三）](04e-delegate-partition.md)

在 Context 已开放委托回调后，XNNPACK 先枚举可接管节点，再处理仅依赖常量的转换。这一篇说明“选了谁”，下一篇再解释“怎么合成分区与代理节点”。

上一章把 Context 切换到允许查看和改写图的接口。这里交回给 Subgraph 的是**节点编号列表**，不是新执行计划。为了读懂候选检查，先看普通路径；MoE 与静态权重展开是两条附加分支。

## 1. 枚举候选并交给 Runtime

`DelegatePrepare` 先找候选，再向 Subgraph 发出改写请求。判断不是只看 opcode，还依赖参数、张量类型、形状、常量属性及实际 kernel 覆盖。在原稿的实现中，常规节点与 MoE 节点分组提交。这里的 MoE 是特定自定义算子，不代表 XNNPACK 会识别任意写成通用子图的专家网络：路由输入决定选哪些专家，权重/scale 的数量和类型都属于匹配条件。原稿分支的匹配包括指定 custom 名称、可解析属性（专家数和隐藏/模型维度等）、单输出、浮点权重 7 输入或量化权重 10 输入、FP32 激活/权重系数/输出与 INT32 路由索引；专家权重静态且形状、类型、元素数正确，量化 scale 也须静态且符合要求。这些是该版本检查项，不是跨版本的 MoE 接口规范。

### XNNPACK 先枚举节点，再做能力检查

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

对照：[DelegatePrepare](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7872-L7915)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> moe_ops_to_replace(
      nullptr, TfLiteIntArrayFree);
  TfLiteIntArray* moe_ops_to_replace_raw = nullptr;
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context, &moe_ops_to_replace_raw);
  moe_ops_to_replace.reset(moe_ops_to_replace_raw);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  TfLiteStatus status = kTfLiteOk;
  if (ops_to_replace->size != 0) {
    status = context->ReplaceNodeSubsetsWithDelegateKernels(
        context, kSubgraphRegistration, ops_to_replace, delegate);
  }
  TfLiteIntArrayFree(ops_to_replace);
  if (status != kTfLiteOk) {
    return status;
  }

  if (moe_ops_to_replace != nullptr) {
    for (int i = 0; i < moe_ops_to_replace->size; ++i) {
      TfLiteIntArray* singleton_moe_op = TfLiteIntArrayCreate(1);
      singleton_moe_op->size = 1;
      singleton_moe_op->data[0] = moe_ops_to_replace->data[i];
      status = context->ReplaceNodeSubsetsWithDelegateKernels(
          context, kSubgraphRegistration, singleton_moe_op, delegate);
      TfLiteIntArrayFree(singleton_moe_op);
      if (status != kTfLiteOk) {
        break;
      }
    }
  }
  return status;
}
```

</details>

`PrepareOpsToDelegate()` 负责选点，`ReplaceNodeSubsetsWithDelegateKernels()` 负责改执行计划。此处 `data` 指向 XNNPACK Delegate 自身。普通节点和受支持的 MoE（混合专家）自定义节点分别提交；MoE 的路由输入用来决定这次计算选哪些专家，而不是让 Runtime 事后把任意算子组合认成 MoE。

## 2. 消费者、MoE 与普通候选

对照：[PrepareOpsToDelegate](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7292-L7830)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteIntArray* Delegate::PrepareOpsToDelegate(
    TfLiteContext* context, TfLiteIntArray** moe_ops_to_delegate) {
  if (moe_ops_to_delegate != nullptr) {
    *moe_ops_to_delegate = nullptr;
  }
  int subgraph_index = 0;
  if (context) {
    tflite::Subgraph* this_subgraph =
        reinterpret_cast<tflite::Subgraph*>(context->impl_);
    subgraph_index = this_subgraph->GetSubgraphIndex();
  }
  if (subgraph_index >= static_unpacked_data_.size()) {
    static_unpacked_data_.resize(subgraph_index + 1);
  }
  std::unordered_map<int, Buffer>& static_unpacked_data =
      static_unpacked_data_[subgraph_index];
  static_unpacked_data.clear();
  static_unpack_nodes_.clear();
  static_sparse_weights_.clear();
  f16_input_tensor_for_dequant_f32_tensor_.clear();
  local_id_to_resources_.clear();
  ClearTempScales();

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  std::unordered_map<int, std::vector<int>> node_ids_for_input_tensor;
  for (int i = 0; i < execution_plan->size; i++) {
    const int node_index = execution_plan->data[i];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      continue;
    }
    for (int i = 0; i < node->inputs->size; i++) {
      node_ids_for_input_tensor[node->inputs->data[i]].push_back(node_index);
    }
  }
```

</details>

`node_ids_for_input_tensor` 的方向是 Tensor 编号 → 读取它的节点列表。一个权重可被多个消费者使用。

```cpp
  std::unordered_map<int, int> quasi_static_tensors_producers;
  std::unordered_set<int> quasi_static_tensors;
  std::unordered_set<int> quasi_static_tensors_to_unpack;
```

准静态 Tensor 是图上的算子输出，但其输入来自常量，因此结果可以在准备时算出。`quasi_static_tensors` 记候选，`quasi_static_tensors_producers` 记生产它的节点，`quasi_static_tensors_to_unpack` 只记 XNNPACK 需要提前展开的子集。三个集合或映射回答的是三个问题，不是重复存同一份权重。

```cpp
  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
  TfLiteIntArray* moe_nodes_to_delegate = nullptr;
  if (moe_ops_to_delegate != nullptr) {
    moe_nodes_to_delegate = TfLiteIntArrayCreate(execution_plan->size);
    moe_nodes_to_delegate->size = 0;
  }
  auto cleanup_and_return_null = [&]() -> TfLiteIntArray* {
    TfLiteIntArrayFree(nodes_to_delegate);
    TfLiteIntArrayFree(moe_nodes_to_delegate);
    if (moe_ops_to_delegate != nullptr) {
      *moe_ops_to_delegate = nullptr;
    }
    return nullptr;
  };
```

创建了nodes_to_delegate, 以及moe_nodes_to_delegate用来记录对应的要被替换的node

### MoE 候选与普通候选

```cpp
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }

    if (moe_nodes_to_delegate != nullptr && registration != nullptr &&
        node != nullptr &&
        MoeExpertsDelegateKernel::IsMoeExpertsNode(registration, node)) {
      if (MoeExpertsDelegateKernel::IsSupported(context, node, registration,
                                                node_index) == kTfLiteOk) {
        moe_nodes_to_delegate->data[moe_nodes_to_delegate->size++] = node_index;
      }
      continue;
    }
```

先检查node是否可以被moe替代，这里判断的条件有两个

```cpp
  static bool IsMoeExpertsNode(const TfLiteRegistration* registration,
                               const TfLiteNode* node) {
    (void)node;
    return registration->builtin_code == kTfLiteBuiltinCustom &&
           registration->custom_name != nullptr &&
           std::strcmp(registration->custom_name, kMoeCustomOp) == 0;
  }
```

检查了builtin_code和customname是不是符合moe的要求
isSupported本身比较复杂，浓缩成以下的内容：

1. 算子类型	必须是 kTfLiteBuiltinCustom，且 custom_name == kMoeCustomOp
2. 算子属性	ReadAttributes() 能够成功解析 MoE 属性，如 weight_type、hidden_dim、model_dim、num_experts
3. 输入输出数量	FP32 权重要求 7 个输入；INT8/INT4 权重要求 10 个输入；均要求 1 个输出
4. 输入输出数据类型	src、top_weights、output 必须为 FP32；top_indices 必须为 INT32
5. 权重 Tensor 类型必须匹配 weight_type；必须是 kTfLiteMmapRo 静态常量；维度非空；元素数量必须符合预期
6. Scale Tensor	必须是静态 FP32 Tensor；检查量化 Scale 的元素数量及 per_expert_scale 的长度

在符合要求的情况下，这里记录了nodeindex

## 3. 准静态权重为什么可以提前处理

「准静态」输出虽然有生产节点，但从常量输入出发，可以在准备期得到。例如 INT8 权重转 FP32，或者把常量稀疏数据展开成稠密布局。实现首先建立 tensor 到消费节点的映射，然后记录准静态张量、其生产节点以及委托端实际需要展开的集合。

- DEQUANTIZE：原稿检查单输入/输出、非稀疏的 FP16 或带有效量化参数的 INT8 输入、FP32 输出、只读常量或之前已识别的准静态来源。部分 FP16 权重在所有消费者都满足特定 FC/Conv/DepthwiseConv 权重槽位约束时可直用；若它是激活输入或被其他算子消费，就不能据此省略这步。
- DENSIFY：以符合条件的只读稀疏常量为源，单输入/输出，类型一致；原稿列出 FP16、INT8、FP32。还要看该分支其余元数据检查。
- 测试构建里还有用于暴露不支持算子的强制委托路径，不代表普通构建会禁用回退。

用一个小图看这个限制：`常量 W → DEQUANTIZE → Wf → Conv`，若只有被委托的 Conv 需要 `Wf`，有机会在 Prepare 完成转换；若还有一个 CPU 节点读取 `Wf`，它必须继续得到合法数据。一个张量是不是常量，和能不能删除它的生产节点，是两个判断。

准静态张量不是扫到就能删掉生产者。实际展开时，先按生产节点编号安排预展开，输入优先读已经展开的结果，否则读模型常量；再做反量化或稀疏转稠密，把结果放进 `static_unpacked_data`。只要还有没委托的分支要读这个结果，原转换节点仍要留给 Interpreter。比如 `INT8 权重 → DEQUANTIZE → FP32` 同时被委托和未委托的节点读取时，不能因为 XNNPACK 能预展开，就把原转换从执行计划中拿掉。[消费者与候选筛选](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7517-L7542) · [预展开](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7559-L7655)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];

      bool is_supported_int8_tensor = input_tensor.type == kTfLiteInt8;
      if (is_supported_int8_tensor) {
        const auto* quant_params = static_cast<const TfLiteAffineQuantization*>(
            input_tensor.quantization.params);
        if (quant_params == nullptr) {
          is_supported_int8_tensor = false;
        }
      }
      if (input_tensor.sparsity == nullptr &&
          (input_tensor.allocation_type == kTfLiteMmapRo ||
           quasi_static_tensors.count(node->inputs->data[0]) != 0) &&
          (input_tensor.type == kTfLiteFloat16 || is_supported_int8_tensor) &&
          output_tensor.type == kTfLiteFloat32) {
        const int input_id = node->inputs->data[0];
        const int output_id = node->outputs->data[0];

        if (input_tensor.type == kTfLiteFloat16) {
          bool skip_this_node = true;
          for (int output_node_index : node_ids_for_input_tensor[output_id]) {
            TfLiteNode* output_node = nullptr;
            TfLiteRegistration* output_registration = nullptr;
            if (context->GetNodeAndRegistration(
                    context, output_node_index, &output_node,
                    &output_registration) != kTfLiteOk) {
              continue;
            }
            switch (output_registration->builtin_code) {
              case kTfLiteBuiltinFullyConnected:
              case kTfLiteBuiltinConv2d:
              case kTfLiteBuiltinDepthwiseConv2d:
                if (output_node->inputs->data[0] == output_id) {
                  skip_this_node = false;
                }
                break;
              default:
                skip_this_node = false;
            }
          }
          if (skip_this_node) {
            f16_input_tensor_for_dequant_f32_tensor_[output_id] = input_id;
          }
        }

        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[output_id] = node_index;
        quasi_static_tensors.insert(output_id);

        if (input_tensor.allocation_type != kTfLiteMmapRo) {
          quasi_static_tensors_to_unpack.insert(input_id);
        }

        if (static_sparse_weights_.count(input_id) != 0) {
          static_sparse_weights_.insert(output_id);
        }

        continue;
      }
    }
```

</details>

以上只把满足条件的反量化输出记成准静态 Tensor；没有在判定时立即转换。等确定有委托节点需要这个结果，再分配空间并展开。

1. 判断 DEQUANTIZE 是否可以提前处理

要求满足：

- 算子是 kTfLiteBuiltinDequantize，且只有 1 个输入和 1 个输出 Tensor。
- 输入必须是 FP16 或具有量化参数的 INT8。
- 输出必须是 FP32。
- 输入不能是稀疏 Tensor（sparsity == nullptr）。
- 输入必须是模型静态常量（kTfLiteMmapRo），或者已经识别出的 Quasi-static Tensor。

满足条件意味着反量化结果可以提前计算。

2. 判断 FP16 → FP32 能否跳过

如果输入是 FP16，检查反量化输出的所有消费者：

- 如果消费者是 FULLY_CONNECTED、CONV2D 或 DEPTHWISE_CONV2D，且 FP32 Tensor 用作其权重等非第 0 个输入，则可能跳过反量化。
- 如果作为这些算子的第 0 个输入（通常是 Activation），则不能跳过。
- 如果存在其他类型的消费者，也不能跳过

```cpp
    if (registration->builtin_code == kTfLiteBuiltinDensify &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];

      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.sparsity != nullptr &&
          (input_tensor.type == kTfLiteFloat16 ||
           input_tensor.type == kTfLiteInt8 ||
           input_tensor.type == kTfLiteFloat32) &&
          output_tensor.type == input_tensor.type) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);
        static_sparse_weights_.insert(node->outputs->data[0]);

        continue;
      }
    }
```

识别是否可以提前处理稀疏张量并记录为quasi tensor，需要builtincode为kTfLiteBuiltinDensify，input&output size均为1，另外还需要tensor不可变，数据类型为input_tensor.type == kTfLiteFloat16 || input_tensor.type == kTfLiteInt8 || input_tensor.type == kTfLiteFloat32，输入输出类型一致。

## 4. VisitNode 与实际预展开

- **选择可提前做的转换。** `DEQUANTIZE` 在该路径要求一进一出、有效类型和已知的输入字节；部分 FP16 权重如果只被支持的权重槽消费，可直接使用，不是先造一份 FP32 再跳过。稀疏常量的 `DENSIFY` 还要看输入是否带 sparsity、输出是否与输入同类型。
- **展开并记账。** 用生产者索引安排展开，找到节点并核对一进一出，从已展开缓存或静态常量读取输入，按输出大小建立缓冲区；反量化或转稠密之后，由 `static_unpacked_data` 按 Tensor 编号保存。再处理权重缓存映射和可省略的原转换节点，最后排序候选提交分区。这里的“提前”是委托准备期，不是模型转换工具预先改写了文件。
- **执行资源的所有权另算。** XNNPACK 可借用 CPU Backend 的线程池，也可拥有自己的线程池；析构只释放自己拥有的。权重 cache 缓解的是 packing/重排成本，缓存 provider 负责查找和管理这些条目；它既不是 Thread Pool，也不是保存 Tensor 索引的表。外部 provider 与由 Delegate 创建的 provider 不能按同一套所有权处理。

[准静态识别与检查](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7413-L7542) · [展开与候选提交前的整理](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7559-L7800)

### VisitNode：能力检查与真正构图

```cpp
    if (Subgraph::VisitNode(
            /*subgraph=*/nullptr, /*delegate=*/*this, context, registration,
            node, node_index, quasi_static_tensors,
            /*input_output_tensors=*/{},
            /*fp32_weight_tensors=*/{},
            /*enable_fp16=*/force_fp16()) != kTfLiteOk) {
      for (int j = 0; j < node->inputs->size; j++) {
        const auto it =
            quasi_static_tensors_producers.find(node->inputs->data[j]);
        if (it != quasi_static_tensors_producers.end()) {
          static_unpack_nodes_.erase(it->second);
        }
      }

      continue;
    }

    for (int j = 0; j < node->inputs->size; j++) {
      if (quasi_static_tensors.count(node->inputs->data[j]) != 0) {
        quasi_static_tensors_to_unpack.insert(node->inputs->data[j]);
      }
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
```

visitnode是方法的核心，用于检查一个算子是否能交给XNNPACK执行，可以的话在构建XNNPACK的subgraph（与tflitesubgraph不同），该算子才会被转换为XNNPACK对应节点。visitnode内部分为两种模式，当前由于还没构建subgraph，所以还是检查模式，会验证算子类型，tensor，参数是否满足，等到构建subgraph是会再次检查，每个算子会有自己对应的检查方法和条件如VisitFullyConnectedNode等等，判断通过会创建reshape或者量化转化节点。对于检查后不支持XNNPACK的节点，对涉及的quasi会撤销其producer的标记，跳过这个节点。

### 预展开顺序、缓冲区和缓存

对照：[准静态权重预展开](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7557-L7802)。

```cpp
  for (const auto& i : local_id_to_resources_) {
    if (i.second.GetProxyValue() >= 0) {
      nodes_to_delegate->data[nodes_to_delegate->size++] =
          i.second.GetVarHandleNodeIndex();
    }
  }

  std::vector<int> sorted_quasi_static_tensors_to_unpack(
      quasi_static_tensors_to_unpack.cbegin(),
      quasi_static_tensors_to_unpack.cend());
  std::sort(sorted_quasi_static_tensors_to_unpack.begin(),
            sorted_quasi_static_tensors_to_unpack.end(),
            [&quasi_static_tensors_producers](int t1, int t2) {
              return quasi_static_tensors_producers[t1] <
                     quasi_static_tensors_producers[t2];
            });
```

对quasi做个排序，因为tensor本身可能会有前后依赖

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    if (f16_input_tensor_for_dequant_f32_tensor_.count(t) != 0) {
      continue;
    }

    const int producer_index = quasi_static_tensors_producers[t];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      return cleanup_and_return_null();  // Hard error.
    }

    if (Subgraph::CheckNumInputs(
            context, node, /*expected_num_inputs=*/1,
            static_cast<BuiltinOperator>(registration->builtin_code),
            producer_index) != kTfLiteOk) {
      return cleanup_and_return_null();  // Hard error.
    }

    if (Subgraph::CheckNumOutputs(
            context, node, /*expected_num_outputs=*/1,
            static_cast<BuiltinOperator>(registration->builtin_code),
            producer_index) != kTfLiteOk) {
      return cleanup_and_return_null();  // Hard error.
    }

    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];

    const auto static_unpacked_input_it =
        static_unpacked_data.find(node->inputs->data[0]);
    if (static_unpacked_input_it == static_unpacked_data.end()) {
      if (input_tensor.allocation_type != kTfLiteMmapRo) {
        TF_LITE_KERNEL_LOG(
            context,
            "unexpected allocation type (%d) in tensor %d in node %d (%d)",
            input_tensor.allocation_type, node->inputs->data[0], producer_index,
            registration->builtin_code);
        return cleanup_and_return_null();  // Hard error.
      }
    }
    const char* packed_data =
        static_unpacked_input_it != static_unpacked_data.end()
            ? static_unpacked_input_it->second.data()
            : static_cast<const char*>(input_tensor.data.data);

    const TfLiteTensor& output_tensor = context->tensors[t];
    size_t tensor_elements = output_tensor.bytes;
    switch (output_tensor.type) {
      case kTfLiteFloat32:
        tensor_elements /= sizeof(float);
        break;
      case kTfLiteFloat16:
        tensor_elements /= sizeof(uint16_t);
        break;
      case kTfLiteInt8:
        tensor_elements /= sizeof(int8_t);
        break;
      default: {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected datatype (%s) in tensor %d in node %d",
                           TfLiteTypeGetName(output_tensor.type),
                           node->outputs->data[0], producer_index);
        return cleanup_and_return_null();  // Hard error.
      }
    }

    Buffer unpacked_data_buffer(context->tensors[t].bytes + XNN_EXTRA_BYTES);
    char* unpacked_data = unpacked_data_buffer.data();
    static_unpacked_data[t] = std::move(unpacked_data_buffer);

    switch (registration->builtin_code) {
      case kTfLiteBuiltinDequantize: {
        TFLITE_DCHECK(input_tensor.sparsity == nullptr);
        switch (input_tensor.type) {
          case kTfLiteFloat16:
            DequantizeFloat16(reinterpret_cast<const uint16_t*>(packed_data),
                              reinterpret_cast<float*>(unpacked_data),
                              tensor_elements);
            break;
          case kTfLiteInt8: {
            TfLiteAffineQuantization* quant_params =
                static_cast<TfLiteAffineQuantization*>(
                    input_tensor.quantization.params);
            TFLITE_DCHECK(quant_params != nullptr);

            if (quant_params->scale->size == 1) {
              DequantizeInt8(reinterpret_cast<const int8_t*>(packed_data),
                             reinterpret_cast<float*>(unpacked_data),
                             GetTensorShape(&input_tensor),
                             input_tensor.params.zero_point,
                             input_tensor.params.scale);
            } else {
              const int* zero_point_data = quant_params->zero_point->data;
              std::vector<int> broadcast_zero_points;
              if (quant_params->zero_point->size != quant_params->scale->size) {
                broadcast_zero_points.resize(quant_params->scale->size,
                                             quant_params->zero_point->data[0]);
                zero_point_data = broadcast_zero_points.data();
              }
              PerChannelDequantizeInt8(
                  reinterpret_cast<const int8_t*>(packed_data),
                  reinterpret_cast<float*>(unpacked_data),
                  GetTensorShape(&input_tensor), zero_point_data,
                  quant_params->scale->data, quant_params->quantized_dimension);
            }
            break;
          }
          default:
            TFLITE_DCHECK(false);
        }
        break;
      }
      case kTfLiteBuiltinDensify: {
        TFLITE_DCHECK(input_tensor.sparsity != nullptr);
        const int dims_count = NumDimensions(&output_tensor);
        std::vector<int> vector_shape(dims_count);
        for (int i = 0; i < dims_count; i++) {
          vector_shape[i] = SizeOfDimension(&output_tensor, i);
        }

        switch (input_tensor.type) {
          case kTfLiteFloat32: {
            const size_t dense_size = context->tensors[t].bytes / sizeof(float);
            float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<float> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const float*>(input_tensor.data.data), dense_size,
                unpacked_fp32_data, context);
            break;
          }
          case kTfLiteFloat16: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(Eigen::half);
            Eigen::half* unpacked_fp16_data =
                reinterpret_cast<Eigen::half*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<Eigen::half> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const Eigen::half*>(input_tensor.data.data),
                dense_size, unpacked_fp16_data, context);
            break;
          }
          case kTfLiteInt8: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(int8_t);
            int8_t* unpacked_int8_data =
                reinterpret_cast<int8_t*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<int8_t> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const int8_t*>(input_tensor.data.data), dense_size,
                unpacked_int8_data, context);
            break;
          }
          default: {
            TFLITE_DCHECK(false);
          }
        }
        break;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "unexpected op registration %d at node %d",
                           registration->builtin_code, producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
    }
  }
```

</details>

上面的循环才真正展开权重，可以按以下五步对照：

1. 遍历需要展开的tensor
2. 获取producer node并检查

- 获取tflitenode和registration
- 检查是否只有一个输入一个输出
- 检查输入是否来自静态数据或之前已经展开的quasi

3. 获取输入数据并分配tensor

- 优先用之前展开的数据，否则读取模型静态数据

4. 根据算子类型执行实际展开，包含反量化和稀疏转稠化

5. 保存结果到static_unpack_data

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      return cleanup_and_return_null();  // Hard error.
    }
    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];
    char* unpacked_data = static_unpacked_data[t].data();
    const auto static_unpacked_input_it =
        static_unpacked_data.find(node->inputs->data[0]);
    const char* packed_data =
        static_unpacked_input_it != static_unpacked_data.end()
            ? static_unpacked_input_it->second.data()
            : static_cast<const char*>(input_tensor.data.data);
    weight_cache_provider_->RemapDataBuffer(packed_data, unpacked_data);
  }

  for (int node_index : static_unpack_nodes_) {
    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);
  if (moe_nodes_to_delegate != nullptr) {
    std::sort(&moe_nodes_to_delegate->data[0],
              &moe_nodes_to_delegate->data[moe_nodes_to_delegate->size]);
  }

#ifdef XNNPACK_DELEGATE_TEST_MODE
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif
  if (moe_ops_to_delegate != nullptr) {
    *moe_ops_to_delegate = moe_nodes_to_delegate;
  }
  return nodes_to_delegate;
}
```

</details>

最后更新权重缓存映射，补充可省掉的静态转换节点，再按节点编号整理委托列表。

1. 更新weight cache的buffer映射

2. 若转换的输出只服务于委托侧，可将原转换节点纳入委托，由后端使用已经展开的数据，不必再次由 Interpreter 执行；仍有普通消费者则不能这样做。

3. 对委托节点数组排序

`XNNPACK_DELEGATE_TEST_MODE` 是测试特例：强制把执行计划里的节点都交给 Delegate，以便暴露未实现路径。实际运行的支持范围不能从这段测试分支推断。

交接点是 `context->ReplaceNodeSubsetsWithDelegateKernels`：入口在切到 Delegate Context 时开放，实现仍在 TFLite 的 Subgraph。下一篇 [图分区与代理节点](04e-delegate-partition.md)接着看为什么这些节点可能组成多个分区，以及哪部分数据必须留在分区边界。
