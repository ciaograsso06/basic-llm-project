from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document  

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(text="Este é o primeiro documento."),
    Document(text="Este é o segundo documento, que é diferente do primeiro.")
]

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

query_engine = index.as_query_engine()
response = query_engine.query("Qual documento é sobre diferenças?")
print(response)
