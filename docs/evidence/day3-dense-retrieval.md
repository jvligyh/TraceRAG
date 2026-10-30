# Day 3 稠密检索证据

日期：2026-09-15

## 环境与模型

- 运行时：FastEmbed 0.8.0 + ONNX Runtime CPU
- 模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 维度：384
- 输入：2 篇英文 Markdown，切分为 7 个 Chunk
- 查询：中文「为什么要分别评估检索和生成？」

## 真实运行结果

- top-1 来源：`data/demo/rag_evaluation.md`
- top-1 score：0.555391
- top-1 文本以完整单词 `retrieval` 开始，并说明分别评估 retrieval 和 generation
- top-2、top-3 同样来自正确文档
- 首次运行包含模型下载，索引约 54 秒
- 模型缓存后索引约 1.89 秒
- 缓存后查询约 0.0084 秒

## 确定性测试

- 单元测试不下载模型，使用明确写出的二维向量验证排序和异常路径。
- 全部 29 项仓库测试通过。
- 同一模型、语料、切块配置和查询复跑时，检索排序与分数一致。

## 已知边界

- 当前为内存穷举检索，查询复杂度约为 `O(N × D)`。
- 当前演示语料很小，运行成功不等于已经证明大规模检索效果。
- FastEmbed 对该模型提示当前版本使用 mean pooling；项目固定 FastEmbed 0.8.0，并记录版本以保证结果可复现。
- Windows 终端可能因代码页显示中文乱码，不影响传入模型的 Unicode 字符串和检索结果。
