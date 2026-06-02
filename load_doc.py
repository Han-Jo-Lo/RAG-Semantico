from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_file(file:str,embedding_model:Embeddings):
    loader_pdf=PyPDFLoader(file)
    pages_pdf=loader_pdf.load()

    import re

    for i, page in enumerate(pages_pdf):
        text = page.page_content
        
        # Primero normalizar saltos de línea con espacios intercalados (patrón de Google Docs)
        text = re.sub(r'\n ', ' ', text)       # quitar \n seguido de espacio
        text = re.sub(r' \n', ' ', text)       # quitar espacio seguido de \n
        text = re.sub(r'\n+', '\n', text)      # colapsar múltiples \n en uno solo
        text = re.sub(r'[ \t]+', ' ', text)    # normalizar espacios horizontales
        text = text.strip()
        
        page.page_content = text
        page.metadata["page_number"] = i

    # Pre-corte por tamaño para dar unidades manejables al SemanticChunker
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       # Te sugiero subirlo un poco para que quepan cláusulas enteras
        chunk_overlap=200,     # ¡Crucial! Si una cláusula es muy larga, el traslape mantiene el contexto
        is_separator_regex=True, # <-- OBLIGATORIO para que funcione el \d+
        separators=[
            r"\n\s*\d+\.\s*",  # 1. Títulos numéricos (ej: \n26. o \n 26. )
            r"\n",             # 2. Salto de línea simple
            r"\.\s"            # 3. Punto seguido (oraciones)
        ]
    )
    pre_chunks = pre_splitter.split_documents(pages_pdf)


    # Ahora el SemanticChunker opera sobre chunks más pequeños
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type='percentile'
    )

    return splitter.split_documents(pre_chunks)

