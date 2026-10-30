# Day 2 工作看板

## 工程目标

- [x] 增加 Markdown 解析器，提取一级标题和正文。
- [x] 增加 PDF 解析器，按非空页生成 `Document` 并保留页码。
- [x] 统一 Unicode、换行符和过多空行，同时保留代码缩进。
- [x] 增加带字符区间和来源信息的 `Chunk` 数据结构。
- [x] 增加段落优先、有重叠的确定性切块器。
- [x] 支持中文句末标点、分号和逗号边界，并为中文切块补充测试。
- [x] 使用 SHA-256 生成稳定 document id 和 chunk id。
- [x] 增加 `day2_ingest.py` 命令行入口。
- [x] 使用仓库 Markdown 和真实PDF完成冒烟测试。
- [x] 全部16项单元测试通过。

## 设计规则

1. Markdown 文件生成一个 `Document`。
2. PDF 的每个可提取文本页生成一个 `Document`，便于页码引用。
3. document id 基于原始文件字节的哈希，不依赖本机绝对路径。
4. chunk id 基于 document id、字符区间和块内容。
5. 相同文件和切块配置必须生成相同ID；文件内容变化时ID应改变。

## 你需要亲手完成

- [x] 解释 `Document` 和 `Chunk` 的区别。
- [x] 解释为什么 PDF 按页生成 `Document`。
- [x] 逐行阅读 `ingestion.py` 和 `chunking.py`。
- [x] 亲手对同一文件运行两次脚本，确认ID一致。
- [x] 将 `--max-chars` 从800改为400，先预测再观察 chunk 数量变化。
- [x] 亲手增加一项边界测试。
- [ ] 用自己的话说清 Transformer 中 token、embedding、attention 和输出向量的数据流。

## 亲手运行

```powershell
python scripts\day2_ingest.py data\demo\rag_evaluation.md --max-chars 120 --overlap-chars 20
python -m unittest discover -s tests -v
```

## 当日复盘

- 我能独立解释的一段 Day 2 代码：
- 我还不理解的 Python 语法：
- 我亲手增加的测试：
