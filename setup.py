from setuptools import setup, find_packages

setup(
    name="enterprise-rag",
    version="1.0.0",
    description="Enterprise Retrieval-Augmented Generation System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "faiss-cpu",
        "sentence-transformers",
        "rank-bm25",
        "networkx",
        "sqlalchemy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "plotly",
    ],
    extras_require={
        "llm": ["openai"],
        "pdf": ["pypdf"],
        "viz": ["pyvis"],
        "eval": ["tiktoken"],
        "dev": ["pytest", "httpx"],
    },
)
