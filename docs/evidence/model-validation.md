# 本地模型与工具调用验收

## 可运行组件

- `TransformerEmbeddingModel`：`Qwen/Qwen3-Embedding-0.6B`，固定 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。查询加英文 instruction，文档不加；左 padding 后取最后有效 token，L2 归一化，输出1024维。
- `CrossEncoderReranker`：`BAAI/bge-reranker-v2-m3`，固定 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。把问题和候选文段成对编码，按相关性 logit 降序排列，不把 logit 当概率。
- 两个组件在首次调用时才加载，固定本地 snapshot 可离线重用。GPU 使用 FP16，CPU 使用 FP32。
- `ToolAgent`：只允许 `search` 和 `read_chunk`；读取前必须通过搜索得到 Chunk ID；最终引用只能来自实际工具返回的证据。工具数量、参数、模型输出和最大步数均有检查。
- 实际 DeepSeek 以英文问题完成了 `search → final`，生成中文回答并引用真实 Chunk，见 `agent-live.json`。这是一次冒烟样例，不能替代 Agent 成功率评测。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_models.py --device cuda --repeats 5 --output docs\evidence\model-benchmark.json
.\.venv\Scripts\python.exe -m unittest tests.test_reranking tests.test_agent -v
```

基准脚本使用一条问题与两个文段，报告冷启动和预热延迟、模型 revision、PyTorch/CUDA 版本与峰值显存。这只是推理链路验收；完整检索质量必须使用相同标注集比较 Recall/MRR。

## 官方实现依据

- [Qwen3 Embedding 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)：instruction 格式、last-token pooling、归一化。
- [BGE Reranker 官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)：Transformers SequenceClassification 与 query/passage 成对输入。

14项单元测试验证工具循环、最大步数、恶意/未知工具、无效参数、未见 Chunk、伪造引用、工具失败脱敏，以及重排顺序、去重、重新编号、无效模型分数和不完整权重文件。实际模型结果见同目录的 `model-benchmark.json`。

## 2026-10-29 实测

`qwen-embedding-validation.json` 记录了 RTX 4060 Laptop GPU、PyTorch `2.14.0+cu130` 的一次验收：1024维向量，冷加载与首推理6.11秒，5次预热查询 P50 46.6ms、P95 49.3ms，PyTorch 峰值 allocated 显存约1148 MiB。RRF相关文段与问题的余弦分数约0.754，熊猫干扰文段约0.146。单条样例只说明模型封装和排序能正常工作。JSON中同时记录完整instruction、query template、pooling、max_length和官方实现来源。

Qwen权重1191586416字节，SHA256 为 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`，已与Hugging Face固定revision的官方LFS元数据核对。

最终两模型同进程、先后释放显存的 `model-benchmark.json` 样例：

| 组件 | 冷加载与首推理 | 预热P50 | 预热P95 | 峰值allocated显存 |
|---|---:|---:|---:|---:|
| Qwen3 Embedding，单条查询 | 6.74s | 47.3ms | 47.5ms | 1148 MiB |
| BGE Reranker，两个短文段 | 4.08s | 18.5ms | 20.9ms | 1092 MiB |

BGE实际把RRF相关文段排第1（logit约0.722），把熊猫干扰段排第2（约-11.023）。权重2271071852字节，SHA256 为 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`。Hugging Face CDN下载反复停顿后，通过ModelScope同名模型断点续传；完整文件已与Hugging Face固定revision官方LFS SHA256逐字节一致性核对。

质量结论请读取完整40题检索消融报告；这里的两个短文段样例只用于运行验证和延迟示例。
