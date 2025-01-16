from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate


VECTORSTORE_PATH = "../data/index"

def query_engine():
    print("Carregando o vectorstore...")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH)
    
    print("Carregando o modelo de chat...")
    llm = ChatOllama(model_name="gpt-3.5-turbo", temperature=0)

    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_docuemnts=True
    )
    
    print(f"Realizando consulta: {query}")
    result = chain({"query": query})
    print("\nResposta:", result["result"])

    print("\nFontes:")
    for doc in result["source_documents"]:
        print(f"- Página: {doc.metadata.get('page', 'desconhecida')}")
        print(f"Conteúdo: {doc.page_content[:200]}...\n")
        
if __name__ == "__main__":
    query= "Como monitorar o status de um host no Zabbix?"
    query_engine(query)
