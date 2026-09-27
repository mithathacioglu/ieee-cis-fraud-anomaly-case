"""Koordinatör ve dört uzman. Sabit akış, istek başına tek trace."""

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from fraud_case.service import ScoringService


@dataclass(frozen=True)
class Task:
    request_id: str
    sender: str
    recipient: str
    kind: str
    payload: dict


class ScoringAgent:
    name = "scoring_agent"

    def __init__(self, service):
        self.service = service

    def handle(self, task):
        if task.kind != "score":
            raise ValueError("Scoring agent received an unsupported task")
        return self.service.score(**task.payload)


class RuleAgent:
    name = "rule_agent"

    def __init__(self, engine):
        self.engine = engine

    def handle(self, task):
        if task.kind != "evaluate_rules":
            raise ValueError("Rule agent received an unsupported task")
        return self.engine.explain(ScoringService.rule_inputs(task.payload["scoring_result"]))


class KnowledgeAgent:
    name = "knowledge_agent"

    def __init__(self, rag):
        self.rag = rag

    def handle(self, task):
        if task.kind != "retrieve":
            raise ValueError("Knowledge agent received an unsupported task")
        return self.rag.retrieve(task.payload["question"])


class ReasoningAgent:
    name = "reasoning_agent"

    def __init__(self, rag):
        self.rag = rag

    def handle(self, task):
        if task.kind != "explain_evidence":
            raise ValueError("Reasoning agent received an unsupported task")
        return self.rag.reason(**task.payload)


class Coordinator:
    """Explicit task graph: scoring -> rules -> retrieval -> local-model explanation.

    Scoring/rules are deterministic specialists. The reasoning specialist is LLM-backed.
    No unbounded autonomous loop, fabricated consensus or LLM risk override is used.
    """

    def __init__(self, scoring_agent, rule_agent, knowledge_agent=None, reasoning_agent=None):
        self.scoring_agent, self.rule_agent = scoring_agent, rule_agent
        self.knowledge_agent, self.reasoning_agent = knowledge_agent, reasoning_agent

    @staticmethod
    def _dispatch(trace, request_id, sender, agent, kind, payload):
        if agent is None:
            raise RuntimeError("RAG specialists are not configured")
        task = Task(request_id, sender, agent.name, kind, payload)
        event = {"sequence": len(trace) + 1, "request_id": request_id, "sender": sender,
                 "recipient": agent.name, "task": kind, "input_keys": sorted(payload)}
        trace.append(event)
        started = perf_counter()
        try:
            result = agent.handle(task)
        except Exception as exc:
            event.update(status="failed", error_type=type(exc).__name__, duration_ms=round(1000 * (perf_counter() - started), 2))
            raise
        event.update(status="completed", duration_ms=round(1000 * (perf_counter() - started), 2))
        return result

    def run(self, score_request, include_rag=False, question=None):
        trace, request_id = [], str(uuid4())
        score = self._dispatch(trace, request_id, "coordinator", self.scoring_agent, "score", score_request)
        rules = self._dispatch(trace, request_id, "scoring_agent", self.rule_agent, "evaluate_rules", {"scoring_result": score})
        public = score.public()
        public.update(request_id=request_id, rule_decision=rules["action"], winning_rule=rules["winning_rule"],
                      explanation={"anomaly": score.anomaly_explanation, "context": score.context_explanation, "rules": rules})
        if include_rag:
            question = question or (
                f"Why does rule {rules['winning_rule']} result in {rules['action']} for this transaction? "
                "Explain the observed values, rule priority and limitations."
                if rules["winning_rule"] else
                "Why has no business rule matched this transaction, and what can we conclude from that?"
            )
            matched = [r["id"] for r in rules["rules"] if r["matched"]]
            search_question = question + " " + " ".join(matched)
            sources = self._dispatch(trace, request_id, "rule_agent", self.knowledge_agent, "retrieve", {"question": search_question[:2000]})
            # Kural kanıtı aynen kalsın ama kullanılmayan koşulları LLM'e yollama
            def observations(node):
                if "field" in node:
                    return {node["field"]: node["observed"]}
                return {key: value for group in ("all", "any") for child in node.get(group, [])
                        for key, value in observations(child).items()}
            evidence = {"transaction_id": public["transaction_id"],
                        "raw_anomaly_score": public["scores"]["raw_anomaly_score"],
                        "adjusted_anomaly_score": public["scores"]["adjusted_anomaly_score"],
                        "decision": rules["action"], "winning_rule": rules["winning_rule"],
                        "matched_rules": [{"id": r["id"], "priority": r["priority"], "action": r["action"],
                                           "status": r["status"], "observations": observations(r["condition"]),
                                           "condition": r["condition"]}
                                          for r in rules["rules"] if r["matched"]],
                        "limitations": score.limitations}
            evidence["context_profile"] = public["context_profile"]
            if public["context_profile"] == "product_risk":
                evidence["product_risk_context"] = score.context_explanation["product_risk_context"]
            public["rag"] = self._dispatch(trace, request_id, "knowledge_agent", self.reasoning_agent, "explain_evidence",
                                          {"question": question, "sources": sources, "evidence": evidence})
        public["agent_trace"] = trace
        return public
