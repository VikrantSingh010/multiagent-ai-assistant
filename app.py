import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("Neural Nexus: Multi-Agent AI Interface")
st.markdown("""
    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #61DAFB;'>🤖 Integrated AI Agents</h3>
        <ul style='color: #FFFFFF;'>
            <li>🔍 Advanced Web Search Engine</li>
            <li>📚 Research Paper Analysis</li>
            <li>🧠 Knowledge Base Integration</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("AI Configuration")
api_key = st.sidebar.text_input("🔑 Groq API Key:", type="password")

model_option = st.sidebar.selectbox(
    "🤖 Select Foundation Model",
    ["Llama", "Gemini", "Mixtral"],
    help="Choose the AI model that best suits your needs"
)

if model_option == "Llama":
    model_name = "Llama3-8b-8192"
elif model_option == "Gemini":
    model_name = "llama3-groq-70b-8192-tool-use-preview"
elif model_option == "Mixtral":
    model_name = "mixtral-8x7b-32768"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Greetings! I'm your multi-agent AI assistant. How may I assist you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="Enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True)

    search_agent = initialize_agent([search], llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)
    arxiv_agent = initialize_agent([arxiv], llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)
    wiki_agent = initialize_agent([wiki], llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Step 1: Perform a search query
        search_response = search_agent.run(prompt, callbacks=[st_cb])
        st.write(f"Search Results: {search_response}")

        # Step 2: Use search result to refine Arxiv query
        if "research" in prompt.lower() or "paper" in prompt.lower():
            arxiv_prompt = f"{prompt} Based on search results: {search_response}"
            arxiv_response = arxiv_agent.run(arxiv_prompt, callbacks=[st_cb])
            final_response = f"Search Results: {search_response}\n\nArxiv Analysis: {arxiv_response}"

        # Step 3: Use search result and optionally Arxiv result for Wikipedia query
        elif "define" in prompt.lower() or "meaning" in prompt.lower():
            wiki_prompt = f"{prompt} Based on search results: {search_response}"
            wiki_response = wiki_agent.run(wiki_prompt, callbacks=[st_cb])
            final_response = f"Search Results: {search_response}\n\nWikipedia Definition: {wiki_response}"

        else:
            wiki_prompt = f"{prompt} Based on search results: {search_response}"
            wiki_response = wiki_agent.run(wiki_prompt, callbacks=[st_cb])
            final_response = f"Search Results: {search_response}\n\nWikipedia Summary: {wiki_response}"

        st.session_state.messages.append({'role': 'assistant', "content": final_response})
        st.write(final_response)