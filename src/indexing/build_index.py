"""Chunks ingested documents, embeds them, and stores them in a Qdrant
VectorStoreIndex. Uses Qdrant's local file-mode client (QdrantClient(path=...))
so the project runs with zero external services beyond the OpenAI API —
no Qdrant Cloud account or Docker container required for local development.
Swap to QdrantClient(url=...) for a real deployment; see docker-compose.yml.
"""
import argparse

import qdrant_client
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src import config
from src.ingestion.pdf_loader import load_guideline_pdfs
from src.ingestion.pubmed_loader import load_pubmed_documents


def configure_llama_index():
    Settings.llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )


def get_vector_store() -> QdrantVectorStore:
    client = qdrant_client.QdrantClient(path=config.QDRANT_PATH)
    return QdrantVectorStore(client=client, collection_name=config.QDRANT_COLLECTION)


def build_index(pubmed_queries: list[str] = None) -> VectorStoreIndex:
    configure_llama_index()

    documents = load_guideline_pdfs()
    for query in pubmed_queries or []:
        documents.extend(load_pubmed_documents(query))

    if not documents:
        raise RuntimeError(
            "No documents to index. Add PDFs to data/guidelines/ or pass "
            "--pubmed-query, then rerun."
        )

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    print(f"[build_index] Indexed {len(documents)} source document(s) into Qdrant.")
    return index


def load_existing_index() -> VectorStoreIndex:
    configure_llama_index()
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the clinical guideline index.")
    parser.add_argument(
        "--pubmed-query",
        action="append",
        default=[],
        help="PubMed search query to ingest as additional source material. "
        "Can be passed multiple times.",
    )
    args = parser.parse_args()
    build_index(pubmed_queries=args.pubmed_query)
