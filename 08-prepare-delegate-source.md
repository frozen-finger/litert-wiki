# LiteRT 源码精读（八）：Prepare、XNNPACK 选点与图改写

[上一篇：Resolver 与 CPU 内核](07-resolver-kernel-source.md) · [Delegate 概览](04-resolver-delegate.md) · [ADD 实验](05-reading-lab.md)

按原稿的入口、逐节点 Prepare、候选筛选、准静态展开和分区改写顺序读。准备游标指向执行计划的位置，不是节点编号；委托输入/输出是分区边界，不是全图的输入/输出。

本文是逐段整理的原稿。长代码可展开，短代码直接显示；不同代码块可能位于同一函数的不同位置，不应拼成独立示例编译。上游链接用于对照固定快照，少数原稿中的版本差异以实际 checkout 为准。

opresolver实际构建完成后，会由interpreter或者用户调用它，在之前interpreterbuild的环节中已经有一些相关的内容，现在详细说明这部分的内容

## 1. 委托入口：选择子图、调用与恢复

对照：[Interpreter 改图入口](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter.cc#L405-L483)。

显式委托和后来由默认工厂创建的 Delegate 可走相同的改图入口，区别在何时创建、应用及由谁持有。

```cpp
TfLiteStatus Interpreter::ModifyGraphWithDelegateImpl(
    TfLiteDelegate* delegate) {
  std::vector<int> all_subgraph_indices(subgraphs_.size());
  for (int i = 0; i < subgraphs_.size(); ++i) {
    all_subgraph_indices[i] = i;
  }
  return ModifyGraphWithDelegateImpl(delegate, all_subgraph_indices);
}
```

先创建了一个subgraph index到subgraph的映射

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus Interpreter::ModifyGraphWithDelegateImpl(
    TfLiteDelegate* delegate, const std::vector<int>& active_subgraph_indices) {
  if (active_subgraph_indices.empty()) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Active subgraph indices must not be empty.");
    return kTfLiteApplicationError;
  }

  std::vector<bool> active_subgraphs(subgraphs_.size(), false);
  for (int subgraph_index : active_subgraph_indices) {
    if (subgraph_index < 0 || subgraph_index >= subgraphs_.size()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Active subgraph index %d is out of range.",
                           subgraph_index);
      return kTfLiteApplicationError;
    }
    active_subgraphs[subgraph_index] = true;

  TfLiteStatus status = kTfLiteOk;
  for (int subgraph_index = 0; subgraph_index < subgraphs_.size();
       ++subgraph_index) {
    if (!active_subgraphs[subgraph_index]) {
      continue;
    }
    auto& subgraph = subgraphs_[subgraph_index];
    if (IsValidationSubgraph(subgraph->GetName().c_str()) ||
        subgraph->IsDelegationSkippable()) {
      TFLITE_LOG(TFLITE_LOG_INFO,
                 "Skipping calling ModifyGraphWithDelegate on Subgraph %i: %s",
                 subgraph->GetSubgraphIndex(), subgraph->GetName().c_str());
      continue;
    }
```

</details>

subgraph的数据合法性验证

```cpp
    status = subgraph->ModifyGraphWithDelegate(delegate);
    if (status != kTfLiteOk) {
      break;
    }
  }
  if (status == kTfLiteDelegateError) {
    TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
  }
  return status;
}
```

由每个subgraph调用ModifyGraphWithDelegate进行实际替换，最终如果发生问题，会调用remove方法移除interpreter内的delegate

### 1.1 Subgraph 的改图准备

对照：[委托前后处理](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2583-L2715)。

```cpp
TfLiteStatus Subgraph::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  auto status = ModifyGraphWithDelegateImpl(delegate);
  telemetry::TelemetryReportEvent(&context_, "ModifyGraphWithDelegate", status);
  return status;
}

TfLiteStatus Subgraph::ModifyGraphWithDelegateImpl(TfLiteDelegate* delegate) {
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(),
                                       "ModifyGraphWithDelegate");

  if (delegate == nullptr) {
    ReportError("Null delegate.");
    return kTfLiteDelegateError;
  }

  auto reset_delegation_if_not_ok = [this](TfLiteStatus status) {
    if (status != kTfLiteOk) {
      TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
      ReportError(
          "Restored original execution plan after delegate application "
          "failure.");
      return kTfLiteDelegateError;
    }
    return kTfLiteOk;
  };
```

做准备工作，检查delegate是否非空并准备错误发生时的lambda函数

```cpp
  TF_LITE_ENSURE_STATUS(RedoAllDelegates());

  int64_t delegate_flags = TfLiteDelegateGetFlagsInternal(delegate);
  const bool delegate_supports_dynamic_shapes =
      delegate_flags & kTfLiteDelegateFlagsAllowDynamicTensors;
  const bool hint_fully_delegated_to_single_delegate =
      delegate_flags & kTfLiteDelegateFlagsHintFullyDelegatedToSingleDelegate;
  const auto pre_delegation_state = state_;

  if (state_ == kStateInvokableAndImmutable) {
    state_ = kStateUninvokable;
```

首先判断了delegate是否支持dynamic tensor，是否支持single delegate即一个delegate可以支持graph内的所有算子（此处只代表delegate的能力不代表实际替代）。
kStateInvokableAndImmutable代表了当前的计算图已经主备好并且不支持dynamic tensor，为了应用新的delegate，将state暂时转换为uninvokable即不可调用，以便后续prepare和内存规划。

```cpp
  } else if (!delegate_supports_dynamic_shapes &&
             !hint_fully_delegated_to_single_delegate) {
    int last_execution_plan_index_prepared;
    TF_LITE_ENSURE_STATUS(PrepareOpsStartingAt(
        0, execution_plan_, &last_execution_plan_index_prepared));
    if (has_dynamic_tensors_) {
      TF_LITE_ENSURE_STATUS(EnsureMemoryAllocations());
      TFLITE_LOG_PROD_ONCE(
          tflite::TFLITE_LOG_WARNING,
          "Attempting to use a delegate that only supports static-sized "
          "tensors with a graph that has dynamic-sized tensors (tensor#%d is a "
          "dynamic-sized tensor).",
          dynamic_tensor_index_);
      return kTfLiteApplicationError;
    }
  }

  if (delegates_applied_.empty()) {
    pre_delegation_execution_plan_ = execution_plan_;
  }
```

此处的条件判断是为了确认当前的delegate不支持动态tensor并且不能保证整个计算图都能交给它执行的情况，在这种情况下tflite就需要准备原始CPU计算图从而节省初始化的开销。对于single delegate的情况Delegate 可以自行分析输入输出 Shape，Delegate 可以建立自己的内部计算图和内存规划，TFLite 不再需要为了执行原始 CPU 算子而准备这些节点，原始算子的 Prepare() 还可能产生不必要的 Tensor 分配开销，prepare的操作在这些情况下可以被交给之前描述的delegate的prepare阶段

接下来看原始prepare的流程：

## 2. PrepareOpsStartingAt：保留原循环逐段阅读

对照：[PrepareOpsStartingAt](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1584-L1625)。

```cpp
TfLiteStatus Subgraph::PrepareOpsStartingAt(
    int first_execution_plan_index, const std::vector<int>& execution_plan,
    int* last_execution_plan_index_prepared) {
  if (first_execution_plan_index == 0) {
    has_dynamic_tensors_ =
        HasDynamicTensorImpl(context_, outputs(), &dynamic_tensor_index_);
  }
```

`execution_plan_` 保存节点编号，不是 Tensor 编号。起点是计划位置；只有从位置 0 开始才重新扫子图输出（包括没有生产节点的直通输出），不是把 `dynamic_tensor_index_` 固定设成 0。续跑保留前一段的状态。

```cpp
  for (int execution_plan_index = first_execution_plan_index;
       execution_plan_index < execution_plan.size(); execution_plan_index++) {
    int node_index = execution_plan[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    EnsureTensorsVectorCapacity();
```

Prepare 之前给描述数组保留空间。若需要 `reserve`，取当前大小加余量与按比例扩容的较大值，并刷新 `context_.tensors`。预留槽容量不等于分配 Tensor 数据。

```cpp
    size_t reserved_capacity = std::max(
        required_capacity, (size_t)(tensors_.capacity() * kTensorCapacityInc));
```

```cpp
#ifdef TF_LITE_TENSORFLOW_PROFILER
    tflite::OnTfLiteOpPrepare(GetTFLiteOpName(registration), subgraph_index_,
                              node_index);
#endif  // TF_LITE_TENSORFLOW_PROFILER
```

只有启用这个编译选项才记录节点准备的 tracing 事件；它不改变 Prepare 的分支含义。

```cpp
    const TfLiteStatus op_prepare_status = OpPrepare(registration, &node);
    if (op_prepare_status != kTfLiteOk &&
        op_prepare_status != kTfLiteOutputShapeNotKnown) {
      ReportOpError(&context_, node, registration, node_index,
                    "failed to prepare");
      return op_prepare_status;
    }

    *last_execution_plan_index_prepared = execution_plan_index;

    if (HasDynamicTensor(context_, node.outputs, &dynamic_tensor_index_) ||
        op_prepare_status == kTfLiteOutputShapeNotKnown) {
      has_dynamic_tensors_ = true;
      return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}
```

**为什么会停在动态节点，而不是等它的下游先 Prepare？** 下游可能要用它的实际输出尺寸来决定缓冲区大小。假设计划是 `[7, 2, 9]`，节点 7 产出动态 Tensor：本轮可记下位置 0 并暂时结束；这不表示节点 7 已经 Invoke，也不表示 Tensor 编号是 0。普通的动态临时 Tensor 不触发这个停止规则。

### 2.1 OpPrepare 的回调分派

对照：[OpPrepare](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1419-L1491)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
TfLiteStatus Subgraph::OpPrepare(const TfLiteRegistration& op_reg,
                                 TfLiteNode* node) {
  if (op_reg.registration_external) {
    if (op_reg.registration_external->node_index != -1) {
      TfLiteRegistration* referenced_registration =
          &nodes_and_registration_[op_reg.registration_external->node_index]
               .second;
      if (referenced_registration->prepare == nullptr) {
        if (IsUnresolvedCustomOp(op_reg)) {
          ReportError(
              "Encountered unresolved custom op: %s.\nSee instructions: "
              "https://www.tensorflow.org/lite/guide/ops_custom ",
              op_reg.custom_name ? op_reg.custom_name : "UnknownOp");
          return kTfLiteUnresolvedOps;
        } else {
          return kTfLiteOk;
        }
      }
      return referenced_registration->prepare(&context_, node);
    }
    if (op_reg.registration_external->prepare_with_data) {
      void* user_data = op_reg.registration_external->user_data;
      return op_reg.registration_external->prepare_with_data(
          user_data, reinterpret_cast<TfLiteOpaqueContext*>(&context_),
          reinterpret_cast<TfLiteOpaqueNode*>(node));
    }
    if (op_reg.registration_external->prepare) {
      return op_reg.registration_external->prepare(
          reinterpret_cast<TfLiteOpaqueContext*>(&context_),
          reinterpret_cast<TfLiteOpaqueNode*>(node));
    }
  }
```

</details>

`registration_external` 连接不透明算子 API。先看是否有回指原节点的 wrapper，再判断外部对象是否实现带 data 或不带 data 的回调；此处的外部 user data 不等于 `node.user_data`。

- 第一种情况下，external是内部registration算子的包装，例如使用operator对已有的算子abs做一层封装后可以作为ABI稳定的接口暴露
- 第二种情况代表外部算子使用TfLiteOperatorSetPrepareWithData注册的operator，通过这个方法注册的operator可以通过不同的userdata决定不同的自定义配置和状态
- 第三种情况代表使用TfLiteOperatorSetPrepare注册回调

```cpp
  if (op_reg.prepare == nullptr) {
    if (IsUnresolvedCustomOp(op_reg)) {
      if (IsFlexOp(op_reg.custom_name)) {
        ReportError(
            "Select TensorFlow op(s), included in the given model, is(are) not "
            "supported by this interpreter. Make sure you apply/link the Flex "
            "delegate before inference. For the Android, it can be resolved by "
            "adding \"org.tensorflow:tensorflow-lite-select-tf-ops\" "
            "dependency. See instructions: "
            "https://www.tensorflow.org/lite/guide/ops_select");
      } else {
        ReportError(
            "Encountered unresolved custom op: %s.\nSee instructions: "
            "https://www.tensorflow.org/lite/guide/ops_custom ",
            op_reg.custom_name ? op_reg.custom_name : "UnknownOp");
      }
      return kTfLiteUnresolvedOps;
    }
    return kTfLiteOk;
  }
  return op_reg.prepare(&context_, node);
}
```

已解析算子可以没有 Prepare，实现返回成功；没有解析到实现的 Custom/Flex 才在这里报告 unresolved。回到循环时，`kTfLiteOutputShapeNotKnown` 与真正失败一样需要区分处理：它使本轮准备暂停，但可成功返回给调用方。

### 2.2 AllocateTensors 与 Invoke 如何接棒

原稿的 Prepare 循环只推进“已准备到哪里”。调用者 `PrepareOpsAndTensors()` 还要更新准备和分配的两个位置，执行本轮 arena 分配。若某个委托要求形状先传播，还会先准备委托前的计划。进入 Invoke 后，执行走到准备游标才会继续 Prepare；如果本次内核实际改变了动态输出尺寸，还可能回退下游分配游标。不要把这几个计数器合称为 Tensor 编号。

对照：[准备和分配](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1627-L1700) · [执行中续跑](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1743-L1757) · [动态输出改变时回退](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1828-L1856)。

## 3. SwitchToDelegateContext：从 Runtime 进入委托

对照：[Delegate Prepare 调用与 Context 还原](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2647-L2662)。

```cpp
  SwitchToDelegateContext();
  TfLiteStatus status = TfLiteDelegatePrepareInternal(&context_, delegate);
  SwitchToKernelContext();
  TF_LITE_ENSURE_STATUS(reset_delegation_if_not_ok(status));

  if (hint_fully_delegated_to_single_delegate && !IsFullyDelegated()) {
    ReportError(
        "Hint fully delegated to single delegate is set, but the graph is not "
        "fully delegated.");
    return kTfLiteApplicationError;
  }
```

TfLiteDelegatePrepareInternal会调用delegate的prepare方法

```cpp
TfLiteStatus TfLiteDelegatePrepareInternal(TfLiteContext* context,
                                           TfLiteDelegate* delegate) {
  TfLiteStatus status = kTfLiteOk;
  if (TfLiteDelegateIsOpaque(delegate) &&
      delegate->opaque_delegate_builder->Prepare) {
    status = delegate->opaque_delegate_builder->Prepare(
        reinterpret_cast<TfLiteOpaqueContext*>(context),
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate),
        delegate->opaque_delegate_builder->data);
  } else {
    status = delegate->Prepare(context, delegate);
  }
  return status;
}
```

此处对delegate做了类似operator的方式，对于透明和不透明的接口有不同的处理方式, 以XNNPACK为例看一下delegate是怎么做prepare的. 在XNNPACK内部有一个TfLiteDelegate对象，他在xnnpackdelegate被创建时被注册给了subgraph，其中prepare方法会被指向XNNPACK内的DelegatePrepare方法

### 3.1 XNNPACK 的 Prepare

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

prepareOpsToDelegate会决定哪些算子可以交给XNNPACK，而replacenodesubsetswithdelegatekernels负责把这些节点替换成委托执行的节点，这里的data即是指向delegate对象自己的指针。其中算子分为两种，moe算子和普通算子，会有两次替换处理分别针对这两种不同的算子，moe算子是针对混合专家模型的特殊实现，它是一种动态选择计算分支的结构，由多个基础算子组合而成，moe中还包含了路由，决定最后会选择哪个专家模型。

## 4. XNNPACK 选点与准静态 Tensor

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

quasitensor是指准静态张量，虽然它由计算图的某个算子产生，但输入来自静态常量，所以输出可以提前确定，不需要每次提前计算，例如INT8到FP32的反量化转换结果。quasi_static_tensors_producers记录了quasitensor是由哪个node产生的，quasi_static_tensors记录了有哪些quasitensor，quasi_static_tensors_to_unpack记录被委托给XNNPACK节点使用，需要提前展开的quasi_static_tensor，这里的展开指分配buffer，执行反量化或稀疏展开。

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

### 4.1 MoE 候选与普通候选

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

### 4.2 DEQUANTIZE、DENSIFY 与权重直用

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

此处识别了可以提前执行反量化的tesnor，并将tensor记录quasi tensor

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

### 4.3 VisitNode：能力检查与真正构图

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

### 4.4 预展开顺序、缓冲区和缓存

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

此处对需要提前处理的quasitensor执行实际的数据展开，将结果保存到buffer中，包含5个步骤

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

其中有一个测试模式的宏会将所有节点标记为XNNPACK支持，为了禁止不受支持的节点静默回退到tlite kernel，从而无法暴露问题

接下来是做实际替换的环节, 这个函数是在switchtodelegatecontext时注册到context里的，实际使用的是subgraph（tflite）中的实现

## 5. 从候选节点到新执行计划

对照：[ReplaceNodeSubsetsWithDelegateKernels](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L576-L690)。

```cpp
TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  registration.builtin_code = BuiltinOperator_DELEGATE;
  if (registration.registration_external) {
    registration.registration_external->builtin_code = BuiltinOperator_DELEGATE;
  }

  if (TfLiteDelegateIsOpaque(delegate)) {
    if (!registration.registration_external) {
      TFLITE_LOG(
          tflite::TFLITE_LOG_WARNING,
          "For a delegate with the 'opaque_delegate_builder' field set, the "
          "delegate kernel's TfLiteRegistration object must have the "
          "'registration_external' field set.");
      return kTfLiteDelegateError;
    }

    OpResolver::OpId op_id{registration.registration_external->builtin_code,
                           registration.registration_external->custom_name,
                           registration.registration_external->version};
    auto [it, inserted] = registration_externals_->emplace(
        op_id,
        std::unique_ptr<TfLiteOperator>(registration.registration_external));
    if (!inserted) {
      auto registration_external_from_cache = it->second.get();
      registration.registration_external = registration_external_from_cache;
    }
  }
```

将delegate的registration注册为BuiltinOperator_DELEGATE类型，如果使用的不透明的delegate，管理外部注册对象tfliteoperator的所有权和生命周期

```cpp
  if (!nodes_to_replace->size) {
    return kTfLiteOk;
  }

  std::vector<NodeSubset> node_subsets;
  if (PartitionGraph(nodes_to_replace, &node_subsets) == kTfLiteError) {
    return kTfLiteError;
  }
```

`PartitionGraph` 先根据数据与控制依赖把候选与非候选划分为分区，不修改执行计划。保留中间 Tensor、禁止融合或禁用聚类等配置会限制可合并范围，而不是只看节点是否相邻。

### 5.1 分区的结构与依赖条件

对照：[PartitionGraphIntoIndependentNodeSubsets](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/graph_info.cc#L90-L259)。

```cpp
TfLiteStatus Subgraph::PartitionGraph(const TfLiteIntArray* nodes_to_replace,
                                      std::vector<NodeSubset>* node_subsets) {
  const InterpreterInfo info(this);
  const bool disable_node_fusion =
      ShouldPreserveAllTensors() || DisableDelegateNodeFusion();
  return tflite::PartitionGraphIntoIndependentNodeSubsets(
      &info, nodes_to_replace, node_subsets,
      /*greedily=*/!DisableDelegateClustering(), control_edges_,
      disable_node_fusion);
}
```

graphpatrition即使将计算图按照tensor之间的输入输出建图，之后根据依赖关系将节点划分为多个nodesubset，尽可能把可以由同一个delegate执行的节点合并，同时保证元计算图的语义不变，分割的原则如下：

1. 同一分区node必须有相同的分区类型，Partition和NonPartition两种
2. 节点的所有非可选输入必须已可用，才可以加入当前分区。
3. 显式控制前驱也必须已经安排，不能只检查数据边。
4. 尽可能合并
5. fusion限制时不合并

### 5.2 创建代理节点并登记边界 Tensor

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "Replacing %d out of %d node(s) with delegate (%s) node, "
                  "yielding %zu partitions "
                  "for subgraph %d (%s).",
                  nodes_to_replace->size, execution_plan_.size(),
                  GetDelegateKernalName(registration), node_subsets.size(),
                  subgraph_index_, name_.c_str());

  execution_plan_.clear();

  for (auto& node_subset : node_subsets) {
    switch (node_subset.type) {
      case NodeSubset::kTfNonPartition:
        for (auto it = node_subset.nodes.begin(); it != node_subset.nodes.end();
             ++it) {
          execution_plan_.push_back(*it);
        }
        break;
      case NodeSubset::kTfPartition: {
        int node_index;

        void* delegate_params = nullptr;
        if (TfLiteDelegateIsOpaque(delegate)) {
          TfLiteOpaqueDelegateParams* opaque_params =
              CreateOpaqueDelegateParams(delegate, node_subset);
          delegate_params = opaque_params;
        } else {
          TfLiteDelegateParams* params =
              CreateDelegateParams(delegate, node_subset);
          delegate_params = params;
        }
        TF_LITE_ENSURE_STATUS(AddNodeWithParameters(
            node_subset.input_tensors, node_subset.output_tensors, {}, nullptr,
            0, delegate_params, &registration, &node_index));

        for (int tensor_index : node_subset.output_tensors) {
          TfLiteTensor* tensor = &tensors_[tensor_index];
          TF_LITE_ENSURE(&context_, tensor->delegate == nullptr ||
                                        tensor->delegate == delegate);
          tensor->delegate = delegate;
        }

        TfLiteNode* node = &nodes_and_registration_[node_index].first;
        node->delegate = delegate;
      } break;
      case NodeSubset::kTfUnexplored:
        return kTfLiteError;
        break;
    }
  }
  return kTfLiteOk;
}
```

</details>

根据之前的区分结果，非delegate分区直接加入executionplan，delegate分区则会先创建params其中包含node，delegate，input，output等信息，接着创建新的delegatenode，这个node也是tflitenode会关联新的输入输出，params会作为初始化数据传递给registration的init，node和delegate kernel的registration保存到nodes_and_registration_。

```cpp
bool Subgraph::IsFullyDelegated() const {
  for (const int nid : execution_plan_) {
    const TfLiteNode& node = nodes_and_registration_[nid].first;
    if (node.delegate == nullptr) return false;
  }
  return true;
}
```

`IsFullyDelegated()` 检查当前执行计划中的节点是否都由委托负责，不是检查 `nodes_and_registration_` 是否只剩代理节点。
最后是prepare的最后流程

### 5.3 完成改图，更新内存与状态

对照：[应用委托后的状态处理](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L2663-L2716)。

<details>
<summary>展开这段 C++ 跟读代码</summary>

```cpp
  if (!delegate_supports_dynamic_shapes) {
    if (!hint_fully_delegated_to_single_delegate) {
      state_ = kStateUninvokable;
      TF_LITE_ENSURE_STATUS(
          reset_delegation_if_not_ok(EnsureMemoryAllocations()));
    }

    state_ = kStateInvokableAndImmutable;
  } else if (pre_delegation_state == kStateInvokableAndImmutable) {
    int last_execution_plan_index_prepared;
    TF_LITE_ENSURE_STATUS(reset_delegation_if_not_ok(PrepareOpsStartingAt(
        0, execution_plan_, &last_execution_plan_index_prepared)));
    if (has_dynamic_tensors_) {
      TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
      ReportError(
          "Cannot allow dynamic tensors due to previous delegation, resetting "
          "to original execution plan.");
      return kTfLiteApplicationError;
    }
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
    state_ = kStateInvokableAndImmutable;
  } else if (pre_delegation_state == kStateInvokable) {
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
  }
  delegates_applied_.push_back(delegate);

  return status;
}
```

</details>

这里主要负责根据delegate对动态tensor的支持能力，重新准备tensor内存，并更新subgraph的状态，确保修改后的计算图正常，几种状态定义如下：

kStateUninvokable	当前图尚未准备好执行，需要重新 Prepare 或分配内存
kStateInvokable	图已具备执行条件，仍允许后续进行相关调整
kStateInvokableAndImmutable	图已准备好执行，并受到不可变状态约束，通常是应用不支持动态 Tensor 的 Delegate 后的状态

对应了三种情况：

1. 当前 Delegate 不支持动态 Tensor	重新准备内存，然后将图标记为 Immutable
2. 当前 Delegate 支持动态 Tensor，但之前的 Delegate 不支持	检查新 Delegate 是否引入动态 Tensor；若有则移除 Delegates 并报错
3. 当前 Delegate 支持动态 Tensor，之前的图已经是 Invokable	调用 EnsureMemoryAllocations()，保持图的内存分配状态一致

至此modify的流程基本结束了

## 6. Lazy Delegate 的触发

对照：[默认工厂实际应用](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/interpreter.cc#L333-L395)。

lazydelegate会在interpreterbuild创建时判断，条件如下：

```cpp
inline bool ShouldCreateLazyDelegateProviders(int num_fp32_tensors) {
#if defined(XNNPACK_DELEGATE_ENABLE_QS8) || defined(XNNPACK_DELEGATE_ENABLE_QU8)
  return true;
#else
  return num_fp32_tensors > 0;
#endif
}
```

这个快照下，启用 QS8/QU8 或存在 FP32 Tensor 时可以保存 lazy creator；实际是否创建、能接几个节点，还要看工厂与能力筛选。

```cpp
 OpResolver::TfLiteDelegateCreators GetDelegateCreators() const final {
    return delegate_creators_;
  }
```

默认工厂的创建与应用可能在 `AllocateTensors()` 等入口触发，不必等到第一次 `Invoke()`，也不能把登记工厂当成已改图。
