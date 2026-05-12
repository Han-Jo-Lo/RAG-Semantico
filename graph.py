import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

from config import DATABASE_DIRECTORY_NAME, get_embedding, get_llm, get_vectorstore
from load_doc import load_file
from vector_store import VectorStoreManager

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


def ask_question(state: State):
    print("Cual es tu pregunta respecto a las politicas de Rappi?")
    return {"messages": [HumanMessage(input())]}


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
    prompt_msgs = [
        SystemMessage(
            content=(
                f"{SYSTEM_INSTRUCTIONS}\n\n"
                f"<CONTEXTO_RECUPERADO>\n{context}\n</CONTEXTO_RECUPERADO>"
            )
        )
    ] + messages
    response = llm.invoke(prompt_msgs)
    return {"messages": [response]}



def build_app_graph(vector_store: VectorStoreManager):
    context_node = _make_context_node(vector_store)
    builder = StateGraph(State)
    builder.add_node("context", context_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "context")
    builder.add_edge("context", "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


app_graph = build_app_graph(get_vectorstore())

