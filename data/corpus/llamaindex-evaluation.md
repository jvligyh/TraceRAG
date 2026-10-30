# LlamaIndex: Evaluating

Source: https://docs.llamaindex.ai/en/stable/module_guides/evaluating/
Scope: official-documentation-excerpt; not a full-paper archive.

Framework Component Guides Evaluating Evaluating Copy Markdown Open in Claude Open in ChatGPT Open in Cursor Copy Markdown View as Markdown
Evaluating
Concept Section titled “Concept”
Evaluation and benchmarking are crucial concepts in LLM development. To improve the performance of an LLM app (RAG, agents), you must have a way to measure it.
LlamaIndex offers key modules to measure the quality of generated results. We also offer key modules to measure retrieval quality.
Response Evaluation : Does the response match the retrieved context? Does it also match the query? Does it match the reference answer or guidelines?
Retrieval Evaluation : Are the retrieved sources relevant to the query?
This section describes how the evaluation components within LlamaIndex work.
Response Evaluation Section titled “Response Evaluation”
Evaluation of generated results can be difficult, since unlike traditional machine learning the predicted result isn’t a single number, and it can be hard to define quantitative metrics for this problem.
LlamaIndex offers LLM-based evaluation modules to measure the quality of results. This uses a “gold” LLM (e.g. GPT-4) to decide whether the predicted answer is correct in a variety of ways.
Note that many of these current evaluation modules
do not require ground-truth labels. Evaluation can be done with some combination of the query, context, response,
and combine these with LLM calls.
These evaluation modules are in the following forms:
Correctness : Whether the generated answer matches that of the reference answer given the query (requires labels).
Semantic Similarity Whether the predicted answer is semantically similar to the reference answer (requires labels).
Faithfulness : Evaluates if the answer is faithful to the retrieved contexts (in other words, whether if there’s hallucination).
Context Relevancy : Whether retrieved context is relevant to the query.
Answer Relevancy : Whether the generated answer is relevant to the query.
Guideline Adherence : Whether the predicted answer adheres to specific guidelines.
Question Generation Section titled “Question Generation”
In addition to evaluating queries, LlamaIndex can also use your data to generate questions to evaluate on. This means that you can automatically generate questions, and then run an evaluation flow to test if the LLM can actually answer questions accurately using your data.
Retrieval Evaluation Section titled “Retrieval Evaluation”
We also provide modules to help evaluate retrieval independently.
The concept of retrieval evaluation is not new; given a dataset of questions and ground-truth rankings, we can evaluate retrievers using ranking metrics like mean-reciprocal rank (MRR), hit-rate, precision, and more.
The core retrieval evaluation steps revolve around the following:
Dataset generation : Given an unstructured text corpus, synthetically generate (question, context) pairs.
Retrieval Evaluation : Given a retriever and a set of questions, evaluate retrieved results using ranking metrics.
Integrations Section titled “Integrations”
We also integrate with community evaluation tools.
UpTrain
Tonic Validate (Includes Web UI for visualizing results)
DeepEval
Ragas
RAGChecker
Cleanlab
Usage Pattern Section titled “Usage Pattern”
For full usage details, see the usage pattern below.
Query Eval Usage Pattern
Retrieval Eval Usage Pattern
Modules Section titled “Modules”
Notebooks with usage of these components can be found in the module guides . Note for AI agents: this documentation is built for programmatic access.
- Overview of all docs: https://developers.llamaindex.ai/llms.txt
- Any page is available as raw Markdown by appending index.md to its URL — e.g. https://developers.llamaindex.ai/llamaparse/parse/getting_started/index.md
- Agent-friendly REST search APIs live under https://developers.llamaindex.ai/api/ — search (BM25 full-text), grep (regex), read (fetch a page), and list (browse the doc tree). See https://developers.llamaindex.ai/llms.txt for parameters.
- A hosted documentation MCP server is available at https://developers.llamaindex.ai/mcp. If you support MCP, you can ask the user to install it for browsing these docs directly (an alternative to the REST API). Setup: https://developers.llamaindex.ai/for-agents/mcp/
- Other LlamaIndex tooling for agents — the LlamaParse Platform MCP server, agent skills and plugins, and the n8n node — is mapped at https://developers.llamaindex.ai/for-agents/ Previous
Usage Pattern Next
Contributing A `LabelledRagDataset`
Powered by
