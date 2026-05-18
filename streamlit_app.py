"""
Interfaz Streamlit: única capa (sin API). Gestión de bases Chroma y chat RAG.
Ejecutar: streamlit run streamlit_app.py
sudo docker start redis-stack
"""
import os
import shutil
import sys
import tempfile

_pkg = os.path.dirname(os.path.abspath(__file__))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

import streamlit as st
from langchain_core.messages import HumanMessage

from config import (
    VECTOR_STORES_ROOT,
    get_embedding,
    get_vectorstore_at,
    list_vector_store_entries,
    sanitize_vector_db_name,
    vector_store_path_for_name,
    get_sql
)
from graph import build_app_graph
from load_doc import load_file
from vector_store import VectorStoreManager


@st.cache_resource
def app_graph_for_path(persist_path: str):
    return build_app_graph(get_vectorstore_at(persist_path))


def _labels_paths(entries: list[tuple[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    labels_to_path = {label: path for label, path in entries}
    paths_to_labels = {path: label for label, path in entries}
    return labels_to_path, paths_to_labels


def _bootstrap_store_from_uploaded_pdf(persist_path: str, uploaded_pdf) -> None:
    """Crea Chroma en `persist_path` indexando el PDF subido (objeto Streamlit UploadedFile)."""
    os.makedirs(VECTOR_STORES_ROOT, exist_ok=True)
    emb = get_embedding()
    mgr = VectorStoreManager(embedding_model=emb, persist_directory=persist_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="rag_upload_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_pdf.getvalue())
        chunked = load_file(tmp_path, embedding_model=emb)
        mgr.create_or_update(chunked)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _invalidate_caches() -> None:
    get_vectorstore_at.cache_clear()
    app_graph_for_path.clear()


st.set_page_config(
    page_title="RAG Rappi",
    page_icon="💬",
    layout="wide",
)

# --- IDENTIFICACIÓN DE USUARIO (NUEVA SECCIÓN) ---
if "user_id" not in st.session_state:
    try:
        system_user = os.getlogin()
    except:
        import getpass
        system_user = getpass.getuser()
    st.session_state.user_id = f"sys_{system_user}"
# ------------------------------------------------

os.makedirs(VECTOR_STORES_ROOT, exist_ok=True)

entries = list_vector_store_entries()
labels_to_path, paths_to_labels = _labels_paths(entries)
valid_paths = set(labels_to_path.values())

if "active_db_path" not in st.session_state:
    st.session_state.active_db_path = entries[0][1] if entries else None
elif st.session_state.active_db_path not in valid_paths:
    st.session_state.active_db_path = entries[0][1] if entries else None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Bases de datos vectoriales")
    st.caption("Cada base es una colección vectorial independiente (Chroma).")

    if entries:
        st.subheader("Disponibles")
        for lab, _p in entries:
            st.markdown(f"- **{lab}**")
    else:
        st.info("No hay bases. Crea una nueva abajo.")

    st.divider()
    st.subheader("Base activa (chat)")
    if entries:
        options = [e[0] for e in entries]
        default_idx = 0
        if st.session_state.active_db_path in paths_to_labels:
            default_idx = options.index(paths_to_labels[st.session_state.active_db_path])
        choice = st.selectbox("Usar en conversación", options=options, index=default_idx)
        st.session_state.active_db_path = labels_to_path[choice]

        st.write("---")
            # En la barra lateral
        sql_db=get_sql(choice)
        datos_excel=sql_db.preparar_excel_descarga()

        if datos_excel:
            st.download_button(
                label="📥 Descargar reporte de fallos",
                data=datos_excel,
                file_name=f"Gaps_{choice}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"btn_descarga_{choice}_{len(datos_excel)}" # Truco de la KEY dinámica
        )
        else:
            st.caption("✅ No hay preguntas fallidas.")

    else:
        st.session_state.active_db_path = None

        

    st.divider()
    st.subheader("Crear base")
    new_name = st.text_input(
        "Nombre",
        placeholder="ej. politicas_2025",
        help="Solo letras, números, guion y guion bajo.",
    )
    uploaded_pdf = st.file_uploader(
        "Documento PDF a indexar",
        type=["pdf"],
        help="Obligatorio: la nueva base se crea a partir de este archivo.",
        key="create_db_pdf",
    )
    if uploaded_pdf is not None:
        st.caption(f"Archivo seleccionado: **{uploaded_pdf.name}**")
    if st.button("Crear base de datos"):
        if uploaded_pdf is None:
            st.error("Debes seleccionar un PDF para crear la base vectorial.")
        else:
            try:
                name = sanitize_vector_db_name(new_name)
                path = vector_store_path_for_name(name)
                if os.path.exists(path):
                    st.error("Ya existe una base con ese nombre.")
                else:
                    try:
                        _bootstrap_store_from_uploaded_pdf(path, uploaded_pdf)
                    except Exception as exc:
                        if os.path.isdir(path):
                            shutil.rmtree(path, ignore_errors=True)
                        st.error(f"No se pudo crear o indexar la base: {exc}")
                    else:
                        _invalidate_caches()
                        st.session_state.active_db_path = path
                        st.session_state.messages = []
                        st.success(f"Base «{name}» creada e indexada con «{uploaded_pdf.name}».")
                        st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Eliminar base")
    if not entries:
        st.caption("Nada que eliminar.")
    else:
        del_label = st.selectbox("Seleccionar", options=[e[0] for e in entries], key="delete_pick")
        confirm = st.checkbox("Confirmo borrado permanente del directorio en disco.", key="delete_confirm")
        if st.button("Eliminar", disabled=not confirm):
            target = labels_to_path[del_label]
            shutil.rmtree(target, ignore_errors=False)
            _invalidate_caches()
            st.session_state.messages = []
            if st.session_state.active_db_path == target:
                st.session_state.active_db_path = None
            st.success("Base eliminada.")
            st.rerun()



st.title("Asistente de consulta de documentos")
st.caption(
    "Respuestas con RAG sobre la base activa."
)

if not st.session_state.active_db_path:
    st.warning("Crea o restaura una base vectorial desde la barra lateral para usar el chat.")
else:
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    graph = app_graph_for_path(entries[0][1])

    if prompt := st.chat_input("Escribe tu pregunta…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        config = {'configurable': {'thread_id': st.session_state.user_id}}

        with st.chat_message("user"):
            st.markdown(prompt)


        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            token_counter = 0  # Definimos el contador fuera del for
            
            # Usamos stream_mode="messages" para obtener los tokens (chunks)
            for msg, metadata in graph.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="messages"
            ):
                # Filtro: Solo mensajes del chatbot y que tengan contenido
                if metadata.get("langgraph_node") == "chatbot" and msg.content:
                    full_response += msg.content
                    token_counter += 1
                    
                    # Solo actualizamos el markdown cada 2 tokens
                    if token_counter % 2 == 0:
                        placeholder.markdown(full_response + "▌")
            
            # Forzamos la última actualización para asegurar que se vea el mensaje completo
            placeholder.markdown(full_response)
            answer = full_response
            st.session_state.messages.append({"role": "assistant", "content": answer})

            final_state=graph.get_state(config)
            if final_state.values.get('no_answer'):
                sql_db=get_sql(choice)
                sql_db.registrar_pregunta(prompt,usuario=st.session_state.user_id)
                st.rerun()
                

        
