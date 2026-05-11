from config import get_embedding,get_vectorstore, get_llm
from load_doc import load_file
import os


embedding=get_embedding()

vectorial_db=get_vectorstore()

if not os.path.exists('./db_folder'):
    print("Creando base de datos vectorial...")
    chunked_doc = load_file('rappi',
    embedding_model=embedding)
    vectorial_db.create_or_update(chunked_doc)

from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm=get_llm()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente de soporte técnico experto. Tu objetivo es ayudar "
    "a los usuarios basándote exclusivamente en el contexto proporcionado.\n"
    "REGLAS CRÍTICAS:\n"
    "1. Usa el CONTEXTO_RECUPERADO para responder.\n"
    "2. Si la respuesta no está en el contexto, di: 'Lo siento, no tengo esa información'.\n"
    "3. Responde siempre en español de forma profesional."
)

class State(TypedDict):
    messages:Annotated[list,add_messages]
    retrieved_context:str

def ask_question(state:State):
    print('Cual es tu pregunta respecto a las politicas de Rappi?')
    return {'messages':[HumanMessage(input())]}

def context_node(state:State):
        pregunta=state['messages'][-1].content
        context=vectorial_db.retrieve(pregunta)

        context_str="\n\n---\n\n".join(doc.page_content for doc in context)
        return {'retrieved_context':context_str}

def chatbot_node(state:State):
    context=state['retrieved_context']
    
    messages=state['messages']

    prompt_msgs=[
        SystemMessage(content=(
    f"{SYSTEM_INSTRUCTIONS}\n\n"
    f"<CONTEXTO_RECUPERADO>\n{context}\n</CONTEXTO_RECUPERADO>"
    ))
    ]+messages  

    response=llm.invoke(prompt_msgs)

    response.pretty_print()
    
    return {'messages':[response]}


builder=StateGraph(State)
builder.add_node('chatbot',chatbot_node)
builder.add_node('question',ask_question)
builder.add_node('context',context_node)

builder.add_edge(START,'question')
builder.add_edge('question','context')
builder.add_edge('context','chatbot')
builder.add_edge('chatbot',END)

Graph=builder.compile()

respuesta=Graph.invoke({'messages':[]})
