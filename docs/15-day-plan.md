# 15 天交付计划

2026-10-29：工程代码已扩展到本地0.2.0候选版，详见completion.md。Day6为40题AI草稿待人工相关性复核；Day7实测结果见evidence/final-evaluation；Day8～13的工具循环/API/重试/集成测试/部署文件/缓存优化已实现。Day12本机无Docker时容器运行验收仍待环境；Day14已有浏览器演示记录但个人讲解录屏/简历面试和Day15投递需用户本人完成。

| 天数 | 工程产出 | 面试产出 |
|---:|---|---|
| 1 | 仓库、Provider 边界、Mock Pipeline、测试 | 解释架构，复习5道 Hot 100 |
| 2 | Markdown/PDF 接入和稳定 chunk id | Python 基础和 Transformer 数据流 |
| 3 | 稠密检索 Baseline | Embedding、余弦相似度和 top-k |
| 4 | 词级、字符 n-gram 与查询翻译增强 BM25 | BM25 与词法/语义/跨语言检索取舍 |
| 5 | RRF 融合和去重 | RRF、Recall 和 Precision |
| 6 | 30–50 条人工核验评测问题 | Recall@k、MRR 和答案评测 |
| 7 | Embedding 模型、BM25 与 RRF 消融实验表 | 解释模型选型、质量/延迟/资源取舍，准备 5 分钟项目介绍 |
| 8 | 最小工具调用循环 | Agent 状态、最大步数和失败控制 |
| 9 | FastAPI 查询与文档端点 | HTTP、参数校验和异步基础 |
| 10 | 日志、重试和配置 | 开始外部投递 |
| 11 | 单元与集成测试 | RAG 失败分类 |
| 12 | Docker 和一键启动 | Linux、进程和网络基础 |
| 13 | 延迟剖析和一项优化 | 项目深挖模拟面试 |
| 14 | README、架构图和演示录屏 | 简历改写和第二次模拟面试 |
| 15 | 稳定候选版本 | 第一志愿投递华为 AI 应用工程师 |

## Day 7 消融实验预案

Day 7 不凭模型大小做选型，而是在 Day 6 的同一份人工标注集上比较：

1. 当前 MiniLM 稠密检索 Baseline；
2. Qwen3-Embedding-0.6B 稠密检索；
3. 词级、字符 n-gram 与翻译增强 BM25；
4. MiniLM + BM25 家族榜 + RRF；
5. Qwen3-Embedding-0.6B + BM25 家族榜 + RRF。

所有实验固定语料、Chunk、查询、top-k 和评测代码，只改变待研究组件。记录 Recall@1/3/5、MRR、索引时间、查询延迟、模型大小和内存占用。详细设计见 `docs/day7-ablation-plan.md`。
