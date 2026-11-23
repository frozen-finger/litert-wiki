# Interpreter 执行（二）：Prepare、内存分配与动态 Invoke

上一篇：[执行（一）](04-resolver-delegate.md) · [目录](README.md) · 下一篇：[Delegate（一）](04c-delegate-setup.md)

普通内核的 Prepare 不等于委托的 Prepare，也不等于已执行节点。先把运行时准备和分配的流程读通，后面就能理解为什么静态 Delegate 应用前还要预跑普通算子的 Prepare。

## 1. PrepareOpsStartingAt：逐个执行计划位置准备

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

## 2. OpPrepare 如何找到实际回调

[固定版本源码：普通与不透明回调](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1419-L1491)

| 按顺序判断的情形 | 实际路径 |
| --- | --- |
| 有 `registration_external`，且 wrapper 记录了原节点编号 | 通过节点编号取原 `TfLiteRegistration`，调用原实现的 Prepare。 |
| 外部对象提供 `prepare_with_data` | 使用外部对象自己的 `user_data`；传入不透明 Context 和 Node。它不是 `TfLiteNode::user_data`。 |
| 外部对象只提供 `prepare` | 调用无 data 参数的不透明回调。 |
| 以上都未接管 | 回到普通 `registration.prepare`。 |
| 最终没有回调 | 已解析的算子可以成功跳过；未解析的 Custom/Flex 占位项返回 `kTfLiteUnresolvedOps`。 |

原笔记里「外部包装可以复用原生算子」的关键是第一行：有不透明对象不代表重写了一遍 ABS，也不代表每次都通过 ext 对象自身的回调。排查先看是不是 wrapper，再查两个回调和 user data 来源。

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

## 3. AllocateTensors 暂停后，Invoke 怎样继续

[准备与内存规划](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1627-L1700) · [Invoke 续跑](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1743-L1757) · [动态输出重置](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1828-L1856)

- **准备原计划与当前计划。** 若有已应用的委托要求先传播形状，并且保存了委托前的计划，例程先用原计划准备，再准备实际执行的计划。不要只根据字段名猜传参：该快照原计划分支写回 `next_original_execution_plan_index_to_prepare_`，实际传给调用的起点却是 `next_execution_plan_index_to_prepare_`。
- **已经准备到哪里，只分配到哪里。** 当前计划的 Prepare 游标更新为 `last + 1`。需要时建立内存规划器并规划，再对本轮已准备区间执行分配；核对自定义分配后推进分配游标。Prepare 与分配是两步，不是 `OpPrepare` 调完就等于所有数据已分配。
- **执行时继续。** 例如节点 7 的输出大小由输入内容决定，后续节点 2 依赖它。初始准备可先停在节点 7；Invoke 执行节点 7 后，走到计划位置 1 才继续准备节点 2。如果本轮执行让动态输出改变大小，准备起点重设为当前项之后；已超前规划的下游内存也会被重置。不会从计划位置 0 无条件重做整图。

原稿的 Prepare 循环只推进“已准备到哪里”。调用者 `PrepareOpsAndTensors()` 还要更新准备和分配的两个位置，执行本轮 arena 分配。若某个委托要求形状先传播，还会先准备委托前的计划。进入 Invoke 后，执行走到准备游标才会继续 Prepare；如果本次内核实际改变了动态输出尺寸，还可能回退下游分配游标。不要把这几个计数器合称为 Tensor 编号。

对照：[准备和分配](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1627-L1700) · [执行中续跑](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1743-L1757) · [动态输出改变时回退](https://github.com/google-ai-edge/LiteRT/blob/e3c11f0d0e7ec02ecb6fd4ee872d5ee569402f62/tflite/core/subgraph.cc#L1828-L1856)。
