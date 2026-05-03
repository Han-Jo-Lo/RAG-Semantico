
# Implementacion de RAG con Chunking Semantico

Este proyecto implementa un RAG con Chunking Semantico, en el cual se le carga un documento y en base a este documento se hacen preguntas para que el llm responda, se utiliza **SemanticChunker** para hacer los chunks de forma semantica y **Chromadb** como base de datos vectorial

---

## 🚀 Flujo de Trabajo

Las fases que intervienen en el proceso de son los siguientes:
- Cargar el documento PDF que se establezaca para hacer el chunk y guardarlo en la base de datos vectoria si la DB no esta creada
- Chunking semantico del documento
- Cargar el documento PDF que se establezca
- Chunking del documento
- Crear una base de datos vectorial si no esta creada
- Guardar los chunk del documento en la base de datos vectorial
- Crear la definicion del grafo para la ejecución de la logica del llm
- Hacer una busqueda semantica (retrieve) de la pregunta del usuario
- Pasar al llm el contexto obtenido por medio de la busqueda semantica y la pregunta del usuario
- Mostrar respuestas

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.10.19**
- **pypdf:** Carga de documentos
- **SemanticChunker:** Chunk de documentos
- **RecursiveCharacterTextSplitter:** Chunk de documentos
- **Chroma:** Bases de datos vectorial
- **LangGraph & LangChain:** Orquestación de la lógica del LLM y manejo de estados

---

## ⚙️ Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/Han-Jo-Lo/RAG-simple.git
cd RAG-simple
```

### 2. Se sugiere crear un entorno virtual

### 3. Ejecutar el entorno virtual

### 4. instalar las librerias necesarias desde el terminal

```bash
pip install -r requirements.txt
```

### 5. crear el archivo .env con la API de OpenAI OPENAI_API_KEY="API KEY"

### 6. Ejecutar el script desde el terminal

```bash
python graph.py
```

---

## Codigo

### load_doc.py

- Se cargan el documento para trabajar con el 
- Se hace chunk de ellos

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings# Clase de los embeddingd

def load_file(file:str,embedding_model:Embeddings):# Se indica que el embedding_model es de tipo embedding, esto con el fin de pasarle el embedding completo al metodo
    loader_pdf=PyPDFLoader(file)# Se instancia la clase con el nombre del archivo a cargar
    pages_pdf=loader_pdf.load()# Se carga el archivo

    for i, page in enumerate(pages_pdf):# Se recorren las diferentes paginas del documento cargado. Al usar enumerate, se obtiene: la página en sí (page) y su índice de posición (i), empezando desde 0.
        page.page_content = ' '.join(page.page_content.split())# Se dividen todos los caracteres y se vuelven a juntar por medio de un espacio(' '), esto con el fin de eliminar los saltos de linea (\n) 
        page.metadata["page_number"] = i# Agrega una nueva "etiqueta" o metadato al objeto de la página.
```

```python

    splitter = SemanticChunker(
        embedding=embedding_model,# Para el chunking semantico se debe pasar el modelo del embedding
        breakpoint_threshold_type='percentile'

    pages_char_split=splitter.split_documents(pages_pdf)# Se hace el chunk sobre el documento

    return pages_char_split # Se retorna el documento con los chunks
```

### vector_store.py
Se define la clase que va a manejar todo lo relacionado con la base de datos vectoria (Chroma)
* Constructor de la clase
* crear or actualizar la DB
* Cargar la DB
* Busqueda (retieve)
* Eliminacion de documentos
* cambiar de proveedor del embedding


```python
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self, embedding_model, persist_directory='./default_db'): # Se define el constructor de la clase
        """Inicializa el modelo de embeddings y la configuración de la base de datos."""
        self.embeddings = embedding_model
        self.persist_directory = persist_directory
        self.vector_store = None
```
```python
    def create_or_update(self, documents: list):
        """Crea una nueva base de datos o añade documentos a la existente."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()# En las versiones mas recientes de Chroma esto no es necesario
        return self.vector_store
```
```python
    def load(self):
        """Carga la base de datos desde el disco si no está ya en memoria."""
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vector_store
```
```python
    def retrieve(self, query: str, k: int = 3):
        """Realiza una búsqueda de similitud."""
        # Aseguramos que la DB esté cargada
        db = self.load()
        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
```
```python
    def delete_by_source(self, source_name: str):
        """Elimina todos los documentos que provengan de un archivo específico."""
        db = self.load()
        # Buscamos los IDs de los documentos que coincidan con la fuente
        docs = db.get(where={"source": source_name})
        if docs["ids"]:
            db.delete(ids=docs["ids"])
            print(f"Eliminados {len(docs['ids'])} fragmentos de {source_name}")
```
```python
    def switch_provider(self, provider: str):
        """Permite cambiar el motor de embeddings sobre la marcha."""
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        elif provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Nota: Al cambiar esto, deberías apuntar a una carpeta de persistencia diferente
```

### graph.py
Aca se gestiona todo
* Se llama al metodo que carga el documento
* Se instancia la clase VectorStoreManager
* Si es necesario se crea la DB de lo contrario se carga la DB
* Se crea el grafo y se ejecuta

```python
from load_doc import load_file
from vector_store import VectorStoreManager
import os

from langchain_openai.embeddings import OpenAIEmbeddings

embedding=OpenAIEmbeddings(model='text-embedding-ada-002') #Se define el modelo del embeddig

appi_db=VectorStoreManager(persist_directory='./rappi',embedding_model=embedding)

# 2. Solo cargamos el PDF si la base de datos NO existe todavía
if not os.path.exists('./rappi'):
    print("Creando base de datos vectorial...")
    chunked_doc = load_file('Términos y Condiciones de Uso de la Plataforma Rappi.pdf',
    embedding_model=embedding)# Se indica el nombre del documento, junto con el modelo del embedding
    rappi_db.create_or_update(chunked_doc)# se crea la DB
else:
    print("Base de datos detectada. Cargando...")
    rappi_db.load()# Se carga la DB si ya existe
```
```python
from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()



llm=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
)

class State(TypedDict):
    messages:Annotated[list,add_messages]

def ask_question(state:State):# En este estado se hace la pregunta al usuario para que lo muestre en el terminal
    print('Cual es tu pregunta respecto a las politicas de Rappi?')
    return {'messages':[HumanMessage(input())]}

def chatbot_node(state:State):
    pregunta=state['messages'][-1].content# Se toma el contenido del mensaje
    contexto=rappi_db.retrieve(pregunta)# se realiza la busqueda en la DB
    prompt=f'''basado en la informacion del contexto a continuacion
    {contexto}

    responde la siguiente pregunta
    {pregunta}
    '''
    response=llm.invoke(prompt)

    response.pretty_print()
    
    return {'message':response}


builder=StateGraph(State)
builder.add_node('chatbot',chatbot_node)
builder.add_node('question',ask_question)

builder.add_edge(START,'question')
builder.add_edge('question','chatbot')
builder.add_edge('chatbot',END)

Graph=builder.compile()

respuesta=Graph.invoke({'messages':[]})
```

    





    

    

    

    

