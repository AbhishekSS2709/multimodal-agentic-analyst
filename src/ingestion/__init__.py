"""Ingestion sub-package — document loaders, metadata tagging, and pipeline.

Quick start::

    from src.ingestion import IngestPipeline

    pipeline = IngestPipeline()
    docs = pipeline.ingest("data/raw/report.pdf")
    docs = pipeline.ingest_directory("data/raw/")

Individual loaders can also be used directly::

    from src.ingestion import load_pdf, load_csv, load_txt
"""

from .csv_loader import load_csv
from .metadata_tagger import tag_document, tag_documents
from .pdf_loader import Document, load_pdf
from .pipeline import IngestPipeline
from .txt_loader import load_txt

__all__ = [
    "Document",
    "IngestPipeline",
    "load_csv",
    "load_pdf",
    "load_txt",
    "tag_document",
    "tag_documents",
]
