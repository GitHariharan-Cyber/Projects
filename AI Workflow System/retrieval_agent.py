from app.db.chroma_manager import similarity_search


def retrieval_agent(state):
    query = state["question"]

    docs = similarity_search(query)

    retrieved_text = [doc.page_content for doc in docs]

    state["retrieved_docs"] = retrieved_text

    return state