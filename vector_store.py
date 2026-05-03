from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
<<<<<<< HEAD
    def __init__(self, embedding_model, persist_directory='./default_db'):
        self.embeddings = embedding_model
=======
    def __init__(self, model_name='text-embedding-ada-002', persist_directory='./default_db'):
        self.embeddings = OpenAIEmbeddings(model=model_name)
>>>>>>> 4b8892ae4b811e4845bd94d79656662a80b7b175
        self.persist_directory = persist_directory
        self.vector_store = None

    def create_or_update(self, documents: list):
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        return self.vector_store

    def load(self):
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vector_store

    def retrieve(self, query: str, k: int = 3):
        # Aseguramos que la DB esté cargada
        db = self.load()
        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def delete_by_source(self, source_name: str):
        db = self.load()
        # Buscamos los IDs de los documentos que coincidan con la fuente
        docs = db.get(where={"source": source_name})
        if docs["ids"]:
            db.delete(ids=docs["ids"])
            print(f"Eliminados {len(docs['ids'])} fragmentos de {source_name}")

    def switch_provider(self, provider: str):
        if provider == "openai":
<<<<<<< HEAD
            self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
=======
            self.embeddings = OpenAIEmbeddings()
>>>>>>> 4b8892ae4b811e4845bd94d79656662a80b7b175
        elif provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Nota: Al cambiar esto, deberías apuntar a una carpeta de persistencia diferente