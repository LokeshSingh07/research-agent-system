from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain
from rich import print


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # =============================================================
    # step1: search agent working
    # ===========================================================
    print("\n" +  "="*50)
    print("step 1: search agent working ... ")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and relevant information on {topic}")]
    })
    # print(search_result)
    state["search_result"] = search_result['messages'][-1].content
    # print("\nsearch result => \n", state["search_result"])




    # =============================================================
    # step2: reader agent working
    # ============================================================
    print("\n" +  "="*50)
    print("step 2: reader agent working ... ")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user", 
            f"Based on the following search resluts topic {topic}"
            f"Pick the most relevant url and scrape it for deeper content.\n\n"
            f"search results: {state['search_result'][:800]}"              
        )]
    })

    # print(reader_result)
    state["reader_result"] = reader_result['messages'][-1].content
    # print("\n\n Scraped content => \n", state["reader_result"][:800])




    # =============================================================
    # step 3: Writer chain working
    # =============================================================
    print("\n" +  "="*50)
    print("step 3: Writer chain writing the report ... ")
    print("="*50)

    research_combined = (
        f"Topic: {topic}\n\n"
        f"Deatiled scraped content: {state['reader_result']}\n\n"
    )

    writer_result = writer_chain.invoke({
        "topic": topic,
        "research_gathered": research_combined
    })
    # print(writer_result)

    state["report"] = writer_result
    # print("\n\n Research report => \n", state["report"])




    # =============================================================
    # state 4: Critic chain working
    # =============================================================
    print("\n" +  "="*50)
    print("step 4: Critic is reviewing the report ... ")
    print("="*50)

    critic_result = critic_chain.invoke({
        "report": state["report"]
    })

    state["feedback"] = critic_result
    # print("\n\n Critic \n", state["feedback"])


    return state




if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    result = run_research_pipeline(topic)
    print("result => \n", result)

