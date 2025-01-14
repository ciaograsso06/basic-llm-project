import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENTS_DIR = "../data/documents"
INDEX_DIR = "../data/index"

def index_documents():
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    documents = SimpleDirectoryReader(input_dir=DOCUMENTS_DIR).load_data()

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print(f"Índice criado e salvo em: {INDEX_DIR}")

if __name__ == "__main__":
    index_documents()
