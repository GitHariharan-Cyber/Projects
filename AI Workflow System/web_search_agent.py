import requests
from bs4 import BeautifulSoup


def web_search_agent(state):
    query = state["question"]

    url = f"https://www.google.com/search?q={query}"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")

    snippets = []

    for g in soup.find_all("div")[:5]:
        snippets.append(g.get_text())

    state["web_results"] = "\n".join(snippets)

    return state