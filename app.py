import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.chains import LLMMathChain
from langchain_classic.callbacks import StreamlitCallbackHandler

# ----------------------------------------------------
# 1. PAGE SETUP & CONFIGURATION
# ----------------------------------------------------
load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Maths & Wikipedia Agent", 
    page_icon=":robot:", 
    layout="wide"
)

st.title("Text-to-Math Problem Solver & Search Assistant")

# Sidebar Configuration
groq_api_key = st.sidebar.text_input(
    label="Enter your Groq API Key:", 
    type="password", 
    key="groq_api_key", 
    help="Provide your Groq API Key to power the models."
)

temperature = st.sidebar.slider(
    label="Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.4, 
    step=0.1
)

if not groq_api_key:
    st.warning("Please provide your Groq API Key in the sidebar to use the application.")
    st.stop()


# ----------------------------------------------------
# 2. AGENT & TOOLS INITIALIZATION (CACHED)
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_math_bot(api_key, temp):
    # Base LLM Model
    model = ChatGroq(
        groq_api_key=api_key, 
        model_name="qwen/qwen3.8-27b", 
        temperature=temp
    )

    # Tool 1: Wikipedia API
    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = Tool(
        name="wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to answer questions about current events or general knowledge. Input should be a search query.",
    )

    # Tool 2: Math Chain / Calculator
    math_chain = LLMMathChain.from_llm(llm=model, verbose=True)
    calculator_tool = Tool(
        name="calculator",
        func=math_chain.run,
        description="A tool for answering math-related questions. Input must be a mathematical expression or clean textual word problem.",
    )

    # Tool 3: Custom Chain & Reasoning Tool
    prompt_template = """You are a helpful AI agent for solving user mathematical questions.
Logically arrive at the solution, explain your reasoning step by step, and display the final answer in a markdown code block.

Question: {question}
Answer:"""
    
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    reasoning_chain = prompt | model

    reasoning_tool = Tool(
        name="reasoning",
        func=lambda q: reasoning_chain.invoke({"question": q}).content,
        description="A tool for answering logic-based, step-by-step math reasoning queries.",
    )

    # Primary Agent Build
    agent = initialize_agent(
        tools=[wikipedia_tool, calculator_tool, reasoning_tool],
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
    
    return agent

# Compile the agent instance
assistant_agent = initialize_math_bot(groq_api_key, temperature)


# ----------------------------------------------------
# 3. CHAT HISTORY MANAGEMENT
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I am your Math & Search Chatbot. Ask me a complex math problem or tell me to look up something on Wikipedia!",
        }
    ]

# Render historic message logs
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


# ----------------------------------------------------
# 4. BOT CORE INTERACTION FLOW
# ----------------------------------------------------
def generate_response(question, callback_handler):
    with st.spinner("Analyzing rules and calculating outputs..."):
        # Legacy Zero Shot template maps incoming questions via the 'input' key
        response = assistant_agent.invoke(
            {"input": question}, 
            config={"callbacks": [callback_handler]}
        )
    return response["output"]


# User Chat Input
question = st.chat_input("Ask me anything about maths or search for information on Wikipedia...")

if st.button("Submit Question", key="submit_button") or question:
    if question:
        # Write user query to screen and history logs
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Build thought containers and parse text outputs inside assistant context block
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            try:
                final_answer = generate_response(question, st_cb)
                st.write(final_answer)
                
                # Append finalized output to log history
                st.session_state["messages"].append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"An error occurred while running the agent: {str(e)}")
    else:
        st.warning("Please type a question in the chat input bar before clicking submit.")
