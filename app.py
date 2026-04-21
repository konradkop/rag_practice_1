import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/books"
CHROMA_PATH = "choma"


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Remove previous database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
    chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saves {len(chunks)} chunks to {CHROMA_PATH}")

    return chunks




load_documents()