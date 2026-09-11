# MedRAG: Clinical Guideline RAG with LlamaIndex

A retrieval-augmented QA system over clinical guidelines (PDFs + PubMed
abstracts), built with LlamaIndex + Qdrant + OpenAI, served via FastAPI and
Streamlit, and containerized with Docker.

## What makes this different

Most "RAG with LlamaIndex" projects stop at "it answers questions." This one
is built around a specific, harder claim: **the system should know when it
doesn't know**, and that claim is backed by a real evaluation, not just
demoed.

- **Runtime guardrail**: every answer is gated on retrieval confidence
  (`src/guardrails/guardrails.py`). Below threshold → the system explicitly
  refuses and tells the user to consult a clinician, instead of letting the
  LLM generate a fluent, unsupported answer. In a clinical context, a
  confident wrong answer is worse than no answer.
- **Offline hybrid evaluation** (`src/evaluation/run_eval.py`) that checks
  whether the guardrail is actually working, using three independent signals:
  1. **Faithfulness** (LLM-as-judge) — is the answer supported by the
     retrieved context?
  2. **Relevancy** (LLM-as-judge) — does the answer address the question?
  3. **Grounding overlap** (deterministic, non-LLM) — lexical overlap between
     the answer and the retrieved text. This exists because an LLM judge can
     itself hallucinate a "yes" — a second, cheap, non-LLM signal that agrees
     with the judge is the actual evidence that the faithfulness number means
     something.
  Plus **abstention rate** on deliberately out-of-scope questions in the eval
  set — the system is scored on correctly saying "I don't know," not just on
  answering well.
- **Zero-cost local vector store**: Qdrant runs in local file-mode
  (`QdrantClient(path=...)`), so there's no Qdrant Cloud account needed for
  development — only `docker-compose.yml` runs a real service for deployment.

## Architecture

```
PDFs (data/guidelines/) ──┐
                          ├──► SentenceSplitter ──► OpenAI Embeddings ──► Qdrant (local/server)
PubMed abstracts (NCBI) ──┘                                                    │
                                                                                ▼
User question ──► input guardrail ──► VectorStoreIndex retriever ──► similarity gate
                                                                       │        │
                                                            (below threshold)  (above threshold)
                                                                       ▼        ▼
                                                                  abstain    OpenAI LLM ──► answer + sources
```

## Setup (local, no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in OPENAI_API_KEY

# 1. Add guideline PDFs to data/guidelines/, and/or pull PubMed abstracts:
python -m src.indexing.build_index --pubmed-query "hypertension management guideline"

# 2. Run the API
uvicorn src.api.main:app --reload

# 3. In a separate terminal, run the UI
streamlit run streamlit_app.py
```

## Deploying (Render)

This repo includes `render.yaml` for a one-click Render Blueprint deploy (API
+ UI, wired together automatically). See **[DEPLOY.md](DEPLOY.md)** for the
full walkthrough, including what actually happens on the free tier and what
still requires a manual step.

## Setup (Docker)

```bash
cp .env.example .env        # fill in OPENAI_API_KEY
python -m src.indexing.build_index --pubmed-query "diabetes management guideline"  # build the index first — it's a local volume the containers mount
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

## Running the evaluation

```bash
python -m src.evaluation.run_eval
```

This replays `src/evaluation/eval_dataset.json` (a starter set — **replace
the questions with ones matched to whatever guidelines you actually
ingest**, otherwise the eval is measuring nothing meaningful) through the full
guardrailed pipeline and writes `src/evaluation/eval_report.json` with
per-question and aggregate faithfulness/relevancy/grounding/abstention
numbers.

## Running tests

```bash
pytest tests/ -v
```

The guardrail unit tests require no API key and run in CI
(`.github/workflows/ci.yml`) on every push.

## Honest status — what's verified vs. what needs your API key

I built and verified this without live model access, so be clear-eyed about
what's actually been checked before you put numbers on your resume:

- ✅ **Verified**: all modules import and compile cleanly; all 9 guardrail
  unit tests pass against real input (threshold logic, PII regex, abstention
  logic) — this is the core "unique" claim of the project and it works.
- ✅ **Verified**: the PubMed ingestion code is correct against the real NCBI
  E-utilities API — it works, but this sandbox's network allowlist blocks
  `eutils.ncbi.nlm.nih.gov` (confirmed via `x-deny-reason: host_not_allowed`,
  not an NCBI error), so it needs to run from your machine to actually pull
  data.
- ⬜ **Not verified end-to-end**: the embedding → Qdrant → LLM → eval path
  needs your `OPENAI_API_KEY` and cannot be tested here. Run
  `python -m src.evaluation.run_eval` yourself and put the *real* numbers it
  prints in your resume/README — don't invent placeholder metrics.

## Suggested resume bullets (fill in the blanks with your real eval numbers)

- Built a clinical-guideline RAG system (LlamaIndex, Qdrant, OpenAI) with a
  retrieval-confidence guardrail that abstains rather than hallucinates on
  out-of-scope or low-evidence queries, validated by an offline hybrid eval
  (LLM-judge faithfulness/relevancy + lexical grounding) achieving **[X]%
  faithfulness** and **[Y]%** correct abstention on out-of-scope questions.
- Designed and shipped the evaluation harness itself, not just the pipeline —
  three independent groundedness signals to catch cases where an LLM-as-judge
  metric alone would be misleading.
