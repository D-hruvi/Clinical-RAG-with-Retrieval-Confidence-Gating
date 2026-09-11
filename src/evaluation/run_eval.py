"""Runs the eval dataset through the full guardrailed pipeline and reports:

1. Faithfulness  — LLM-as-judge: is the answer supported by the retrieved
   context? (llama_index.core.evaluation.FaithfulnessEvaluator)
2. Relevancy     — LLM-as-judge: does the answer actually address the
   question? (RelevancyEvaluator)
3. Grounding overlap — a cheap, non-LLM lexical check: what fraction of the
   answer's content words appear in the retrieved source text. This exists
   because LLM-judges can themselves hallucinate a "yes" — having a second,
   deterministic signal that mostly agrees with the LLM judge is the actual
   evidence that the faithfulness number is trustworthy, not just a number.
4. Abstention behavior — for the deliberately out-of-scope question(s) in the
   dataset, did the guardrail correctly refuse instead of guessing?

Output: src/evaluation/eval_report.json + a printed markdown table.
Requires OPENAI_API_KEY (the judge calls the same LLM configured in config.py).
"""
import json
import re
from collections import Counter

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

from src import config
from src.retrieval.query_engine import ClinicalRAGEngine

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "for", "and", "or", "in",
    "on", "with", "should", "be", "this", "that", "it", "as", "by", "at",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def grounding_overlap(answer: str, sources: list[dict]) -> float:
    """Fraction of the answer's content words that also appear somewhere in
    the retrieved source excerpts. 0.0 if there's nothing to compare."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    source_text = " ".join(s["excerpt"] for s in sources)
    source_words = _content_words(source_text)
    if not source_words:
        return 0.0
    overlap = answer_words & source_words
    return round(len(overlap) / len(answer_words), 4)


def run_evaluation():
    with open(config.EVAL_DATASET_PATH) as f:
        dataset = json.load(f)

    engine = ClinicalRAGEngine()
    faithfulness_judge = FaithfulnessEvaluator()
    relevancy_judge = RelevancyEvaluator()

    results = []
    for item in dataset:
        question = item["question"]
        rag_response = engine.query(question)

        record = {
            "id": item["id"],
            "question": question,
            "answer": rag_response.answer,
            "abstained": rag_response.abstained,
            "top_retrieval_score": rag_response.top_score,
        }

        if rag_response.abstained:
            # Nothing to judge for faithfulness/relevancy — abstention is
            # itself the correct behavior for out-of-scope questions.
            record.update(
                {"faithfulness": None, "relevancy": None, "grounding_overlap": None}
            )
        else:
            source_texts = [s["excerpt"] for s in rag_response.sources]
            faithfulness_result = faithfulness_judge.evaluate(
                query=question, response=rag_response.answer, contexts=source_texts
            )
            relevancy_result = relevancy_judge.evaluate(
                query=question, response=rag_response.answer, contexts=source_texts
            )
            record.update(
                {
                    "faithfulness": bool(faithfulness_result.passing),
                    "relevancy": bool(relevancy_result.passing),
                    "grounding_overlap": grounding_overlap(
                        rag_response.answer, rag_response.sources
                    ),
                }
            )

        results.append(record)

    summary = _summarize(results)
    report = {"results": results, "summary": summary}

    with open(config.EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    _print_table(results, summary)
    return report


def _summarize(results: list[dict]) -> dict:
    answered = [r for r in results if not r["abstained"]]
    n = len(answered)
    return {
        "total_questions": len(results),
        "abstention_rate": round(
            sum(r["abstained"] for r in results) / len(results), 4
        ),
        "faithfulness_pass_rate": (
            round(sum(r["faithfulness"] for r in answered) / n, 4) if n else None
        ),
        "relevancy_pass_rate": (
            round(sum(r["relevancy"] for r in answered) / n, 4) if n else None
        ),
        "avg_grounding_overlap": (
            round(sum(r["grounding_overlap"] for r in answered) / n, 4) if n else None
        ),
    }


def _print_table(results: list[dict], summary: dict):
    print("\n| id | abstained | faithful | relevant | grounding | top_score |")
    print("|----|-----------|----------|----------|-----------|-----------|")
    for r in results:
        print(
            f"| {r['id']} | {r['abstained']} | {r['faithfulness']} | "
            f"{r['relevancy']} | {r['grounding_overlap']} | "
            f"{r['top_retrieval_score']:.3f} |"
        )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_evaluation()
