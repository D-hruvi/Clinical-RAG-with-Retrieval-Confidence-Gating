"""Fetches abstracts from PubMed via NCBI E-utilities and wraps them as
LlamaIndex Documents. Implemented directly against the REST API (esearch +
efetch) instead of a black-box reader package, so ingestion behavior and
rate-limit handling are visible and testable.

NCBI usage policy: identify yourself with an email, and keep to <=3 req/s
without an API key (<=10 req/s with one). See config.NCBI_EMAIL / NCBI_API_KEY.
"""
import time
import xml.etree.ElementTree as ET
from typing import List

import requests
from llama_index.core import Document

from src import config

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _rate_limit_delay() -> float:
    return 0.11 if config.NCBI_API_KEY else 0.35


def search_pubmed_ids(query: str, max_results: int = 20) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": config.NCBI_EMAIL,
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: List[str]) -> List[Document]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": config.NCBI_EMAIL,
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY

    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    documents = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//AbstractText")

        pmid = pmid_el.text if pmid_el is not None else "unknown"
        title = title_el.text if title_el is not None else "Untitled"
        abstract = " ".join(
            (part.text or "") for part in abstract_parts
        ).strip()

        if not abstract:
            continue  # skip records with no usable text (e.g. retracted, no abstract)

        text = f"Title: {title}\n\nAbstract: {abstract}"
        documents.append(
            Document(
                text=text,
                metadata={
                    "source_type": "pubmed_abstract",
                    "source_name": f"PMID:{pmid}",
                    "pmid": pmid,
                    "title": title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
            )
        )

    return documents


def load_pubmed_documents(query: str, max_results: int = 20) -> List[Document]:
    """Search PubMed for `query` and return abstracts as Documents.
    Batches efetch calls in groups of 20 to stay well within NCBI limits."""
    pmids = search_pubmed_ids(query, max_results=max_results)
    if not pmids:
        print(f"[pubmed_loader] No PubMed results for query: {query!r}")
        return []

    documents = []
    for i in range(0, len(pmids), 20):
        batch = pmids[i : i + 20]
        documents.extend(fetch_abstracts(batch))
        time.sleep(_rate_limit_delay())

    print(f"[pubmed_loader] Loaded {len(documents)} abstracts for query: {query!r}")
    return documents
