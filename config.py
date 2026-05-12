from functools import lru_cache
import os
import re
from langchain_openai.embeddings import OpenAIEmbeddings
from vector_store import VectorStoreManager
from langchain_openai import ChatOpenAI

_ROOT = os.path.dirname(os.path.abspath(__file__))
_raw_db = os.getenv("DATABASE_DIRECTORY", "./db_folder")
DATABASE_DIRECTORY_NAME = (
    _raw_db if os.path.isabs(_raw_db) else os.path.normpath(os.path.join(_ROOT, _raw_db.lstrip("./")))
)

_raw_vs_root = os.getenv("VECTOR_STORES_ROOT", "vector_stores")
VECTOR_STORES_ROOT = (
    _raw_vs_root
    if os.path.isabs(_raw_vs_root)
    else os.path.normpath(os.path.join(_ROOT, _raw_vs_root.lstrip("./")))
)


def sanitize_vector_db_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        raise ValueError("El nombre no puede estar vacío.")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", s):
        raise ValueError("Usa solo letras, números, guion (-) y guion bajo (_).")
    return s


def vector_store_path_for_name(name: str) -> str:
    return os.path.join(VECTOR_STORES_ROOT, sanitize_vector_db_name(name))


def list_vector_store_entries() -> list[tuple[str, str]]:
    """Pares (etiqueta para la UI, ruta absoluta de persistencia Chroma)."""
    entries: list[tuple[str, str]] = []
    if os.path.isdir(VECTOR_STORES_ROOT):
        for n in sorted(os.listdir(VECTOR_STORES_ROOT)):
            p = os.path.join(VECTOR_STORES_ROOT, n)
            if os.path.isdir(p):
                entries.append((n, os.path.normpath(p)))
    legacy = os.path.normpath(DATABASE_DIRECTORY_NAME)
    seen = {path for _, path in entries}
    if os.path.isdir(legacy) and legacy not in seen:
        label = os.path.basename(legacy) or "legacy"
        entries.insert(0, (f"{label} (legacy)", legacy))
    return entries

EMBEDDING_MODEL_NAME=os.getenv('EMBEDDING_MODEL','text-embedding-3-small')
LLM_MODEL_NAME=os.getenv('LLM_MODEL','gpt-4o-mini')
LLM_TEMPERATURE=float(os.getenv('LLM_TEMPERATURE','0'))


@lru_cache(maxsize=1)
def get_embedding()-> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

@lru_cache(maxsize=1)
def get_vectorstore():
     return VectorStoreManager(embedding_model=get_embedding(),persist_directory=DATABASE_DIRECTORY_NAME)


@lru_cache(maxsize=32)
def get_vectorstore_at(persist_directory: str) -> VectorStoreManager:
    return VectorStoreManager(
        embedding_model=get_embedding(),
        persist_directory=os.path.normpath(persist_directory),
    )

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
