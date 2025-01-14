from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_DIR = "../data/index"

def query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = VectorStoreIndex.load_from_storage_context(storage_context, embed_model=embed_model)

    query_engine = index.as_query_engine()

    print("Digite sua pergunta (ou 'sair' para encerrar):")
    while True:
        user_query = input("Pergunta: ")
        if user_query.lower() == "sair":
            print("Encerrando...")
            break
        response = query_engine.query(user_query)
        print(f"Resposta: {response}")

if __name__ == "__main__":
    query_engine()
