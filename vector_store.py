from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self, embedding_model, persist_directory='./default_db'):
        self.embeddings = embedding_model
        self.persist_directory = str(persist_directory)
        self.vector_store = None

    def create_or_update(self, documents: list):
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # ← agregar
        )
        return self.vector_store

    def load(self):
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"}  # ← agregar
            )
        return self.vector_store

    def retrieve(self, query: str, k: int = 3):
        db = self.load()
        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def retrieve_with_score(self, query: str, k: int = 3):
        db = self.load()
        return db.similarity_search_with_score(query=query, k=k)

    def delete_by_source(self, source_name: str):
        db = self.load()
        docs = db.get(where={"source": source_name})
        if docs["ids"]:
            db.delete(ids=docs["ids"])
            print(f"Eliminados {len(docs['ids'])} fragmentos de {source_name}")

    def switch_provider(self, provider: str):
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        elif provider == "huggingface":
            # Nota: Si usas HuggingFace en el futuro, también es recomendable 
            # instalar langchain-huggingface para evitar alertas similares.
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")