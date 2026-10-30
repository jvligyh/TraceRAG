import json
import unittest

from tracerag.agent import ToolAgent


class ScriptedClient:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return next(self.decisions)


def decision(value):
    return json.dumps(value)


EVIDENCE = {"chunk_id": "c1", "document_id": "d1", "text": "RRF combines ranks.", "source": "paper.md"}


class AgentTests(unittest.TestCase):
    def build(self, decisions, **kwargs):
        client = ScriptedClient(decisions)
        return ToolAgent(client, lambda query, top_k: [EVIDENCE], lambda cid: EVIDENCE, **kwargs), client

    def test_search_read_and_final_preserve_evidence(self):
        agent, client = self.build([
            decision({"tool": "search", "arguments": {"query": "RRF"}}),
            decision({"tool": "read_chunk", "arguments": {"chunk_id": "c1"}}),
            decision({"answer": "RRF融合名次。", "citations": ["c1", "c1"]}),
        ])
        result = agent.run("What is RRF?")
        self.assertEqual("answered", result["status"])
        self.assertEqual([EVIDENCE["chunk_id"]], [c["chunk_id"] for c in result["citations"]])
        self.assertEqual(3, client.calls)

    def test_step_limit_bounds_repeated_search(self):
        agent, client = self.build([decision({"tool": "search", "arguments": {"query": "x"}})] * 3, max_steps=2)
        self.assertEqual("step_limit", agent.run("x")["status"])
        self.assertEqual(2, client.calls)

    def test_unknown_tool_is_never_executed(self):
        agent, _ = self.build([decision({"tool": "shell", "arguments": {"cmd": "anything"}})])
        self.assertEqual("unknown_tool", agent.run("x")["status"])

    def test_read_arbitrary_id_is_rejected(self):
        agent, _ = self.build([decision({"tool": "read_chunk", "arguments": {"chunk_id": "../../secret"}})])
        self.assertEqual("invalid_arguments", agent.run("x")["status"])

    def test_hallucinated_citation_is_rejected(self):
        agent, _ = self.build([decision({"answer": "a", "citations": ["invented"]})])
        self.assertEqual("invalid_citations", agent.run("x")["status"])

    def test_malformed_json_and_invalid_top_k_stop(self):
        agent, _ = self.build(["not JSON"])
        self.assertEqual("invalid_response", agent.run("x")["status"])
        agent, _ = self.build([decision({"tool": "search", "arguments": {"query": "x", "top_k": True}})])
        self.assertEqual("invalid_arguments", agent.run("x")["status"])

    def test_tool_failure_is_sanitized(self):
        client = ScriptedClient([decision({"tool": "search", "arguments": {"query": "x"}})])
        def fail(query, top_k):
            raise RuntimeError("private credential")
        result = ToolAgent(client, fail, lambda cid: None).run("x")
        self.assertEqual("tool_error", result["status"])
        self.assertNotIn("private credential", str(result))

    def test_english_refusal_and_empty_citations(self):
        agent, _ = self.build([decision({"answer": "Insufficient evidence", "citations": []})], answer_language="English")
        output = agent.run("x")
        self.assertEqual("insufficient_evidence", output["status"])
        self.assertEqual("Insufficient evidence to answer reliably.", output["answer"])


if __name__ == "__main__":
    unittest.main()
