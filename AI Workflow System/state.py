from typing import TypedDict, List

class AgentState(TypedDict):
    question: str
    retrieved_docs: List[str]
    web_results: str
    summary: str
    final_answer: str