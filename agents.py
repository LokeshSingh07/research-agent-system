
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   # remove *, **, # 
from langchain_core.runnables import RunnableLambda
from tools import web_search, scrape_tool
import requests as req
import os
from dotenv import load_dotenv

load_dotenv()



# Model setup
# llm = init_chat_model("groq:openai/gpt-oss-120b")
llm = init_chat_model("gpt-4.1")


# 1st aegent -> web search that searches the web for recent and reliable information on a topic, returns Titles , URLs and content snippets
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search],
    )


# 2nd agent > reader that scrapes the content of a webpage, returns the HTML content
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_tool],
    )





# writer chain -> writer that writes a clear , structured and insightfull reports based on the research gathered by the reader
writer_prompt = ChatPromptTemplate.from_messages([
    ('system', "You are an expert research writer. Write a clear , structured and insightfull reports"),
    ('human', """Write a detailed research report on the following topic: 
Topic: {topic}

Research Gathered: 
{research_gathered}

structure the output as:
- introduction
- key finding (minimum 3 well-explained points)
- solution
- conclusion
- source (list of references)

Be deatiled, factual and professional in your writing.""")
])

writer_chain = writer_prompt | llm  | StrOutputParser()




# critic_chain -> critic that provides feedback on a research report
critic_prompt = ChatPromptTemplate.from_messages([
    ('system', "You are an expert research critic. You provide feedback on a research report."),
    ('human', """Review the research report below and evaluate it strictly.
Report: {report}

Respond in the following format:
- Overall score: ...
- Strengths:...
- Areas for improvement: ...

- One line verdict
""")
])

critic_chain = critic_prompt | llm  | StrOutputParser()



