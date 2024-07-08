from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import shutil

from langchain.schema import Document


CHROMA_PATH = "chroma"
DIRECTORY = "./documents/small"

def main():
    documents = load_docs()
    chunks = split_documents(documents)
    save_to_chroma(chunks)


def load_docs():
    loader = PyPDFDirectoryLoader(DIRECTORY)
    documents = loader.load()
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 750, chunk_overlap = 200, length_function = len, add_start_index = True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def save_to_chroma(chunks):
    print("Attempting to write embeddings to Chroma")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding = OllamaEmbeddings(model="llama3")

    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)


if __name__ == "__main__":
    main()