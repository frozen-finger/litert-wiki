# LiteRT Wiki｜从模型文件到推理执行

面向想读端侧推理框架源码的开发者：一个 `.tflite` 文件进入进程之后，张量、节点和内存分别在哪里建立，最后又由谁执行？

## 阅读目录

| 主题 | 篇目 |
| --- | --- |
| 模型文件与加载 | [从 FlatBuffers 读懂模型](01-model-format.md) · [加载模型、权重与内部 IR](02-load-model-ir.md) |
| InterpreterBuilder | [（一）创建 Interpreter 和运行时子图](03-interpreter-builder.md) · [（二）Tensor、权重与量化](03b-interpreter-tensors.md) · [（三）算子节点、Signature 与构建收尾](03c-interpreter-nodes.md) |
| Interpreter 执行 | [（一）Resolver、Registration 与 ABS](04-resolver-delegate.md) · [（二）Prepare、内存分配与动态 Invoke](04b-interpreter-prepare.md) |
| Delegate | [（一）创建、应用与执行资源](04c-delegate-setup.md) · [（二）XNNPACK 如何选点与展开权重](04d-delegate-xnnpack.md) · [（三）图分区、代理节点与状态收尾](04e-delegate-partition.md) |
| 实验 | [用一个 ADD 串起模型与执行](05-reading-lab.md) |

**两条读取路径。** 模型格式知识通用，但 `LiteRtModelT` 属于 LiteRT 模型 IR；InterpreterBuilder 读取的是 TFLite FlatBuffer 视图，不需要先构建 `LiteRtModelT`。每篇把代码、分步说明及上游行号放在对应主题内；较长的片段可以展开，不用来回翻另一份文档。

**按问题跳读**

- [模型权重与 Buffer ID](02-load-model-ir.md#用两张-tensor-共用一份权重把-id-对上)
- [Interpreter 的 I/O](03-interpreter-builder.md#inputs--outputs-是怎样填进去的)
- [PrepareOpsStartingAt](04b-interpreter-prepare.md#1-prepareopsstartingat逐个执行计划位置准备)
- [图分区的输入、输出和依赖](04e-delegate-partition.md#2-分区边界怎样从依赖关系算出来)

## 图例与本地阅读

- **归属、持有**：蓝绿色实线。
- **索引、引用或借用**：棕色虚线。
- **调用或控制**：较粗的紫色箭头。
- **数据流或步骤先后**：灰色实线。以图上动词为准。

箭头旁的 1:N、N:1 或多:多只描述这一条边；跨进程或跨阶段不要沿不同类型的边推导所有权。图为普通 PNG，GitHub 页面可直接查看；离线阅读时可通过 Code → Download ZIP 下载，解压后保留 `figures/` 与 Markdown 的相对位置。

## 上游资料

- [Google LiteRT 文档](https://developers.google.com/edge/litert)
- [LiteRT 源码](https://github.com/google-ai-edge/LiteRT)
- [FlatBuffers 二进制布局](https://flatbuffers.dev/internals/)

这是独立的源码学习笔记，不是 LiteRT 官方文档。源码节选及版本对照见各篇的源码链接。
