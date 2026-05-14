import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import  SystemMessage,RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

from config import DATABASE_DIRECTORY_NAME, get_embedding, get_llm, get_vectorstore
from vector_store import VectorStoreManager

from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DOC = os.path.join(_PACKAGE_DIR, "rappi")

embedding = get_embedding()
_default_store = get_vectorstore()


llm = get_llm()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente de soporte técnico experto. Tu objetivo es ayudar "
    "a los usuarios basándote exclusivamente en el contexto proporcionado.\n"
    "REGLAS CRÍTICAS:\n"
    "1. Usa el CONTEXTO_RECUPERADO para responder.\n"
    "2. Si la respuesta no está en el contexto, di: 'Lo siento, no tengo esa información'.\n"
    "3. Responde siempre en español de forma profesional."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_context: str
    summary:str


def _make_context_node(store: VectorStoreManager):
    def context_node(state: State):
        pregunta = state["messages"][-1].content
        context = store.retrieve(pregunta)
        context_str = "\n\n---\n\n".join(doc.page_content for doc in context)
        return {"retrieved_context": context_str}

    return context_node


def chatbot_node(state: State):
    context = state["retrieved_context"]
    messages = state["messages"]
    summary=state.get('summary','')
    summary_context=f'Resumen de la conversacion previa: {summary}\n' if summary else ''

    prompt_msgs = [
        SystemMessage(
            content=(
                f"{SYSTEM_INSTRUCTIONS}\n\n"
                f'{summary_context}\n'
                f"<CONTEXTO_RECUPERADO>\n{context}\n</CONTEXTO_RECUPERADO>"
            )
        )
    ] + messages
    response = llm.invoke(prompt_msgs)
    return {"messages": [response]}

def summarize_memory(state: State):

    messages = state["messages"]
    summary = state.get("summary", "")

    if len(messages) < 10:
        return {}

    prompt_msgs = [SystemMessage(content=(
    f"resumen actual: {summary} \n\n"
    f'Eres un sistema que hace resumen de una conversacion utilizando\n'
    f'el resumen dado anteriormente y las conversaciones dadas a continuacion'
    ))]+messages
    
    new_summary = llm.invoke(prompt_msgs)
    

    messages_to_remove = [RemoveMessage(id=m.id) for m in messages[:-9]]

    return {
        "summary": new_summary.content,
        "messages": messages_to_remove
    }



def build_app_graph(vector_store: VectorStoreManager):
    context_node = _make_context_node(vector_store)
    builder = StateGraph(State)
    builder.add_node("context", context_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("summarize", summarize_memory)
    builder.add_edge(START, "context")
    builder.add_edge("context", "chatbot")
    builder.add_edge("chatbot", "summarize")
    builder.add_edge("summarize", END)

    client = Redis(host="localhost", port=6379, db=0)
    saver = RedisSaver(redis_client=client)
    saver.setup()

    return builder.compile(checkpointer=saver)


app_graph = build_app_graph(get_vectorstore())

