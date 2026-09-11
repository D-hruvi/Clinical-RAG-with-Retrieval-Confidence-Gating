"""Loads clinical guideline PDFs (e.g. WHO, CDC, NICE, ICMR PDFs you download
into data/guidelines/) into LlamaIndex Document objects."""
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

from src import config


def load_guideline_pdfs(directory: str = None) -> List[Document]:
    directory = directory or config.GUIDELINES_DIR
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    pdfs = list(path.glob("*.pdf"))
    if not pdfs:
        print(
            f"[pdf_loader] No PDFs found in {path}. Drop guideline PDFs there "
            "(e.g. WHO hypertension guideline, ADA diabetes standards) and rerun."
        )
        return []

    reader = SimpleDirectoryReader(input_files=[str(p) for p in pdfs])
    documents = reader.load_data()

    for doc in documents:
        doc.metadata["source_type"] = "guideline_pdf"
        # keep only the filename in metadata shown to the LLM, not the full path
        file_name = doc.metadata.get("file_name", "unknown.pdf")
        doc.metadata["source_name"] = file_name

    print(f"[pdf_loader] Loaded {len(documents)} document chunks from {len(pdfs)} PDF(s).")
    return documents
