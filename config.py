from functools import lru_cache
import os
from langchain_openai.embeddings import OpenAIEmbeddings
from vector_store import VectorStoreManager
from langchain_openai import ChatOpenAI

_ROOT = os.path.dirname(os.path.abspath(__file__))
_raw_db = os.getenv("DATABASE_DIRECTORY", "./db_folder")
DATABASE_DIRECTORY_NAME = (
    _raw_db if os.path.isabs(_raw_db) else os.path.normpath(os.path.join(_ROOT, _raw_db.lstrip("./")))
)

EMBEDDING_MODEL_NAME=os.getenv('EMBEDDING_MODEL','text-embedding-3-small')
LLM_MODEL_NAME=os.getenv('LLM_MODEL','gpt-4o-mini')
LLM_TEMPERATURE=float(os.getenv('LLM_TEMPERATURE','0'))


@lru_cache(maxsize=1)
def get_embedding()-> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

@lru_cache(maxsize=1)
def get_vectorstore():
     return VectorStoreManager(embedding_model=get_embedding(),persist_directory=DATABASE_DIRECTORY_NAME)

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
