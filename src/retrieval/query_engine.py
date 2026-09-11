"""Wraps the LlamaIndex query engine with input/output guardrails so both the
FastAPI service and the evaluation script share one code path — the metric
you eval offline is the metric guarding the live API, not a different one."""
from dataclasses import dataclass, field
from typing import List

from llama_index.core.postprocessor import SimilarityPostprocessor

from src import config
from src.guardrails import guardrails
from src.indexing.build_index import load_existing_index


@dataclass
class RAGResponse:
    answer: str
    abstained: bool
    top_score: float
    sources: List[dict] = field(default_factory=list)
    warning: str = None


class ClinicalRAGEngine:
    def __init__(self):
        self.index = load_existing_index()
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=config.SIMILARITY_TOP_K,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.0)  # scoring only; gating happens below
            ],
        )

    def query(self, question: str) -> RAGResponse:
        input_check = guardrails.check_input(question)
        if not input_check.allowed:
            return RAGResponse(
                answer=f"Request rejected: {input_check.reason}",
                abstained=True,
                top_score=0.0,
            )

        warning = None
        if input_check.reason == "off_topic_warning":
            warning = "This question doesn't look clinical — answer may be unreliable."

        response = self.query_engine.query(input_check.sanitized_text)
        source_nodes = response.source_nodes or []
        top_score = max((n.score or 0.0) for n in source_nodes) if source_nodes else 0.0

        output_check = guardrails.check_output(top_score, source_nodes)
        if not output_check.allowed:
            return RAGResponse(
                answer=config.ABSTAIN_MESSAGE,
                abstained=True,
                top_score=top_score,
                sources=[],
                warning=warning,
            )

        sources = [
            {
                "source_name": n.metadata.get("source_name", "unknown"),
                "score": round(n.score or 0.0, 4),
                "excerpt": n.text[:280],
            }
            for n in source_nodes
        ]

        return RAGResponse(
            answer=guardrails.redact_pii(str(response)),
            abstained=False,
            top_score=top_score,
            sources=sources,
            warning=warning,
        )
