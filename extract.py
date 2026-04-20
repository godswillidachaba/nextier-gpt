from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# load documents

folder_path = 'pdfs'


loader = PyPDFDirectoryLoader(folder_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    length_function = len 
)


chunks = text_splitter.split_documents(docs)


# Embedding

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-V2"
)


# Vector Store


vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=model
)


vectorstore.save_local("faiss_index")

