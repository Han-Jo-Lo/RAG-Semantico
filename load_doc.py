from langchain_community.document_loaders import PyPDFLoader
<<<<<<< HEAD
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

def load_file(file:str,embedding_model:Embeddings):
=======
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

def load_file(file:str):
>>>>>>> 4b8892ae4b811e4845bd94d79656662a80b7b175
    loader_pdf=PyPDFLoader(file)
    pages_pdf=loader_pdf.load()


    for i, page in enumerate(pages_pdf):
        page.page_content = ' '.join(page.page_content.split())
        page.metadata["page_number"] = i

<<<<<<< HEAD
    splitter = SemanticChunker(
        embedding=embedding_model,
        breakpoint_threshold_type='percentile'
=======
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
>>>>>>> 4b8892ae4b811e4845bd94d79656662a80b7b175
    )
    pages_char_split=splitter.split_documents(pages_pdf)

    return pages_char_split

