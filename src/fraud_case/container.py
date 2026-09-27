"""Composition root. Testler provider'ları buradan override ediyor."""

import json
from pathlib import Path

from dependency_injector import containers, providers

from fraud_case.agents import Coordinator, KnowledgeAgent, ReasoningAgent, RuleAgent, ScoringAgent
from fraud_case.rag import OllamaClient, RagConfig, RagService, VectorIndex, load_documents
from fraud_case.rules import RuleEngine
from fraud_case.service import FeatureRepository, ScoringService


def rag_config(root):
    return RagConfig(**json.loads((Path(root) / "config/rag.json").read_text()))


def rag_service(root, client):
    root = Path(root)
    return RagService(client, VectorIndex.load(root / "artifacts/rag"), load_documents(root))


class Container(containers.DeclarativeContainer):
    root = providers.Dependency(instance_of=Path)
    context_profile = providers.Object("raw")
    repository = providers.ThreadSafeSingleton(FeatureRepository, root=root)
    scoring = providers.ThreadSafeSingleton(ScoringService, root=root, repository=repository, context_profile=context_profile)
    rules = providers.ThreadSafeSingleton(lambda root: RuleEngine.from_json(root / "config/rules.json"), root=root)
    config = providers.ThreadSafeSingleton(rag_config, root=root)
    client = providers.ThreadSafeSingleton(OllamaClient, config=config)
    rag = providers.ThreadSafeSingleton(rag_service, root=root, client=client)
    scoring_agent = providers.Factory(ScoringAgent, service=scoring)
    rule_agent = providers.Factory(RuleAgent, engine=rules)
    knowledge_agent = providers.Factory(KnowledgeAgent, rag=rag)
    reasoning_agent = providers.Factory(ReasoningAgent, rag=rag)
    coordinator = providers.Factory(Coordinator, scoring_agent=scoring_agent, rule_agent=rule_agent)
    rag_coordinator = providers.Factory(Coordinator, scoring_agent=scoring_agent, rule_agent=rule_agent,
                                        knowledge_agent=knowledge_agent, reasoning_agent=reasoning_agent)
