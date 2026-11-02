from dotenv import load_dotenv
from rich import print
from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup as bs
import requests as req
import os

load_dotenv()



tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# tool
@tool 
def web_search(query: str)->str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""

    res = tavily.search(query, max_results=5)
    # print(res)

    output = []
    for r in res['results']:
        output.append(
            f"title: {r['title']}\nurl: {r['url']}\nsnippet: {r['content'][:300]}\n"
        )

    # print(output)
    return "\n-------\n\n".join(output)



# print(web_search.invoke("what is langchain"))



@tool
def scrape_tool(url: str)->str:
    """Fetch a webpage URL and return its readable text content."""
    
    try:
        resp = req.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # return resp.text

        soup = bs(resp.text, "html.parser")

        # Remove unnecessary elements
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(separator="\n", strip=True)[:3000]
    
    except req.exceptions.RequestException as e:
        return f"Error scraping webpage: {e}"


# print(scrape_tool.invoke({"url": "https://www.codewithlokesh.com"}))