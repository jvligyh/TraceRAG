# GraphRAG 中文阅读笔记

Source: https://arxiv.org/abs/2404.16130
Scope: AI-authored Chinese reading note derived from the public abstract; not the official paper translation; pending human review.

GraphRAG 面向整个语料库的全局性问题，例如询问数据集的主要主题。它先让大语言模型从文档抽取实体知识图谱，再为关联紧密的实体社区预先生成摘要。回答问题时先从各社区摘要生成部分回答，最后把部分回答汇总成最终答案。
