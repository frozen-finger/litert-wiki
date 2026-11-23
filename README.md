# LiteRT Wiki｜从模型文件到推理执行

面向想读端侧推理框架源码的开发者。这组中文笔记沿着同一个问题展开：一个 `.tflite` 文件被读进进程之后，张量、节点和内存分别在哪里建立，最后又由谁执行？

## 阅读目录

| 顺序 | 篇目 | 这一篇主要回答 |
| --- | --- | --- |
| 一 | [模型文件与 FlatBuffers](01-model-format.md) | 文件里的 offset 和索引怎么看？图、签名、权重和量化参数存在哪？ |
| 二 | [加载模型与内部 IR](02-load-model-ir.md) | 字节如何映射、验证和解包？常量数据由谁持有？ |
| 三 | [InterpreterBuilder 与运行时建图](03-interpreter-builder.md) | Builder 如何从 FlatBuffer 建立 Subgraph、Tensor 和节点？ |
| 四 | [OpResolver、Registration 与 Delegate](04-resolver-delegate.md) | 一个算子如何找到实现？图分区、默认委托和失败回退如何工作？ |
| 五 | [用一个 ADD 串起模型与执行](05-reading-lab.md) | 怎样把表索引、运行时 Buffer 与实际执行逐步对应起来？ |

第一篇的格式知识两条路径通用；第二篇跟的是 LiteRT 模型加载到 IR 的路径，第三、四篇跟的是 TFLite 兼容 C++ Interpreter 路径。它们共用模型格式和部分底层实现，但 `LiteRtModelT → InterpreterBuilder` 不是这两份源码的直接调用链。第一次读可以先看第一篇，再对照第五篇的小图回到各自的实现。

流程块按源码分支和调用顺序整理，行号链接可以直接打开对应版本的实现。遇到代码与其他版本不同，先确认你走的是 LiteRT Loader 还是 Interpreter，再查同名入口的参数和分支。

**按问题跳读**

- [模型权重与 Buffer ID](02-load-model-ir.md#用两张-tensor-共用一份权重把-id-对上)：文件索引、内部句柄和外部权重分别从哪里来。
- [Interpreter 的 I/O](03-interpreter-builder.md#inputs--outputs-是怎样填进去的)：从文件中的两个列表到运行时 Tensor 访问。
- [PrepareOpsStartingAt](04-resolver-delegate.md#511-prepareopsstartingat循环一项一项看)：执行计划索引、动态输出暂停，以及后续怎样继续。
- [分区边界](04-resolver-delegate.md#分区边界怎样从依赖关系算出来)：候选节点可以接管不等于可以合并。

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
