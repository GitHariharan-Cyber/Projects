from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = OpenAIEmbeddings()

VECTOR_DB = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


def add_documents(documents):
    VECTOR_DB.add_documents(documents)
    VECTOR_DB.persist()


def similarity_search(query, k=4):
    return VECTOR_DB.similarity_search(query, k=k