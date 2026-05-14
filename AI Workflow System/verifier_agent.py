from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")


def verifier_agent(state):
    summary = state["summary"]

    prompt = f"""
    Verify factual consistency.

    Summary:
    {summary}

    Improve clarity and reduce hallucinations.
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    state["final_answer"] = response.content

    return state