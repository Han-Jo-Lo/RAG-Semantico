import os
import sys

_pkg = os.path.dirname(os.path.abspath(__file__))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

import streamlit as st
from langchain_core.messages import HumanMessage

from graph import app_graph

st.set_page_config(
    page_title="RAG Rappi",
    page_icon="💬",
    layout="centered",
)

st.title("Asistente de políticas Rappi")
st.caption("Respuestas basadas en el documento indexado. Una sola app Python (Streamlit + RAG).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if prompt := st.chat_input("Escribe tu pregunta…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando contexto y generando respuesta…"):
            result = app_graph.invoke(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "retrieved_context": "",
                }
            )
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
