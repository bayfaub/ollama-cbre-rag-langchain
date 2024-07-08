from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

documents_directory = './documents/small'

loader = PyPDFDirectoryLoader(documents_directory, glob='**/*.pdf')
documents = loader.load()

embeddings = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=300,add_start_index=True)
split_documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents=split_documents, embedding=embeddings, persist_directory='chroma')



