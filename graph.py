from load_doc import load_file
from vector_store import VectorStoreManager
import os

from langchain_openai.embeddings import OpenAIEmbeddings

embedding=OpenAIEmbeddings(model='text-embedding-ada-002')

rappi_db=VectorStoreManager(persist_directory='./rappi',embedding_model=embedding)

if not os.path.exists('./rappi'):
    print("Creando base de datos vectorial...")
    chunked_doc = load_file('Términos y Condiciones de Uso de la Plataforma Rappi.pdf',
    embedding_model=embedding)

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

def ask_question(state:State):
    print('Cual es tu pregunta respecto a las politicas de Rappi?')
    return {'messages':[HumanMessage(input())]}

def chatbot_node(state:State):
    pregunta=state['messages'][-1].content
    contexto=rappi_db.retrieve(pregunta)
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
