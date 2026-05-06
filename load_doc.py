from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

def load_file(file:str,embedding_model:Embeddings):
    loader_pdf=PyPDFLoader(file)
    pages_pdf=loader_pdf.load()


    for i, page in enumerate(pages_pdf):
        page.page_content = ' '.join(page.page_content.split())
        page.metadata["page_number"] = i


    splitter = SemanticChunker(
        embedding=embedding_model,
        breakpoint_threshold_type='percentile'
    )

    pages_char_split=splitter.split_documents(pages_pdf)

    return pages_char_split

