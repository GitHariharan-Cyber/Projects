def supervisor_agent(state):
    question = state["question"].lower()

    if "latest" in question or "recent" in question:
        return "web"

    return "retrieval"