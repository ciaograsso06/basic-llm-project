import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

DOCUMENT_PATH = "../data/index/documentation/zabbix_documentation.pdf"
VECTORSTORE_PATH = "../data/index"

def index_documents():
    
    print(f"Carregando documentação de {DOCUMENT_PATH}...")
    loader = PyPDFLoader(document_path=DOCUMENT_PATH)
    documents = loader.load()
    
    print("Dividindo o texto em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    print("Carregando embeddings...")
    
    embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    
    print("Indexando documentos...")
    texts = [doc.page_content for doc in docs]
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Indice salve em {VECTORSTORE_PATH}")


if __name__ == "__main__":
    index_documents()
