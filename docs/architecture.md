# 架构决策记录

## 项目定位

TraceRAG 面向公开技术文档回答问题，返回可核验的引用，并测量 Pipeline 在哪个环节失败。它是一个工程与评测项目，不是聊天界面演示。

## 计划组件

1. 文档接入：PDF、Markdown 和 HTML 解析、归一化与元数据保留。
2. 文本切块：可配置块大小、重叠范围和稳定 chunk id。
3. 检索：BM25 和稠密检索共用统一接口。
4. 融合：Reciprocal Rank Fusion（RRF）与去重。
5. 重排：可选 Cross-Encoder 或供应商 Reranker。
6. 生成：供应商无关客户端、Grounded Prompt 和引用。
7. 评测：Recall@k、MRR、引用命中率、答案正确性和延迟。
8. 服务：FastAPI 端点、校验、日志和错误映射。

## 第一版架构图

```mermaid
flowchart LR
    subgraph Offline[离线接入与建索引]
        S[公开资料清单] --> P[PDF / Markdown / HTML 解析]
        P --> C[归一化与切块]
        C --> M[元数据与稳定 chunk id]
        M --> B[BM25 索引]
        M --> V[稠密向量索引]
    end

    subgraph Online[在线查询链路]
        Q[用户问题] --> BR[BM25 检索]
        Q --> DR[稠密检索]
        B --> BR
        V --> DR
        BR --> F[RRF 融合]
        DR --> F
        F --> RR[可选重排]
        RR --> G[Grounded Prompt + LLM]
        G --> A[答案 + chunk 级引用]
    end

    subgraph Evaluation[离线评测]
        E[人工核验的评测集] --> Q
        A --> R[Recall@k / MRR / 引用命中率 / 延迟]
    end
```

Day 1 只有「构造 Grounded Prompt → Mock」可执行。架构图表示目标形态，不代表每个模块已实现。

Day 2 已将「Markdown/PDF 解析 → 归一化 → 稳定 chunk id」变为可执行链路；BM25 和稠密索引仍属于后续任务。

## Day 1 决策

- LLM 接口与检索、API 框架分离。
- 真实供应商未配置前使用确定性 Mock。
- 从第一天开始将 source id 作为一等数据。
- 先建立评测样例，再优化检索。
- C++ 用于笔试与算法，Python 用于 AI 应用项目。

## Day 15 验收标准

- 一条命令本地启动；
- 文档接入和稳定 chunk id；
- Dense、BM25 和融合检索；
- 重排与有依据的生成；
- API 返回块级引用；
- 30–50 条人工核验评测问题；
- Baseline 和消融实验结果；
- 单元测试与集成测试；
- 可复现 README 和架构图。
