import streamlit as st
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
LANGSMITH_REPO = "your-handle/mr-omar-assistant" # Update this!

st.set_page_config(page_title="Agent Interface & Prompt Hub", layout="wide")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_prompt_text" not in st.session_state:
    st.session_state.system_prompt_text = "You are a helpful AI assistant."

# --- Helper Functions ---
def clear_history():
    st.session_state.chat_history = []

def pull_from_langsmith():
    try:
        prompt_obj = hub.pull(LANGSMITH_REPO)
        for msg in prompt_obj.messages:
            if isinstance(msg, SystemMessagePromptTemplate):
                st.session_state.system_prompt_text = msg.prompt.template
                break
        st.success("Prompt loaded successfully from LangSmith!")
    except Exception as e:
        st.error(f"Failed to pull from LangSmith. Error: {e}")

def commit_to_langsmith(new_system_prompt):
    try:
        new_prompt = ChatPromptTemplate.from_messages([
            ("system", new_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        hub.push(LANGSMITH_REPO, new_prompt)
        st.session_state.system_prompt_text = new_system_prompt
        st.success(f"Successfully committed to {LANGSMITH_REPO}")
    except Exception as e:
        st.error(f"Failed to push to LangSmith: {e}")

def get_llm(model_choice):
    """Dynamically loads the correct LLM class based on the dropdown selection."""
    try:
        if "gpt" in model_choice:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_choice, temperature=0)
        elif "gemini" in model_choice:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_choice, temperature=0)
        elif "deepseek" in model_choice:
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(model=model_choice, temperature=0)
    except Exception as e:
        st.error(f"Error initializing {model_choice}. Check your API keys. Details: {e}")
        st.stop()

# --- Sidebar: Prompt Engineering & Model Selection ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    
    # New: Model Selection Dropdown
    selected_model = st.selectbox(
        "Select LLM Model",
        options=[
            "gpt-4o", 
            "gpt-4o-mini", 
            "claude-3-5-sonnet-20240620", 
            "claude-3-haiku-20240307",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        index=0
    )
    
    st.divider()
    
    st.subheader("Prompt Hub Integration")
    if st.button("⬇️ Pull Latest from LangSmith", use_container_width=True):
        pull_from_langsmith()
        
    edited_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.system_prompt_text, 
        height=350
    )
    
    if st.button("⬆️ Commit to LangSmith", type="primary", use_container_width=True):
        commit_to_langsmith(edited_prompt)

# --- Main Chat Interface ---
st.title("💬 Agent Chat Interface")

col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🗑️ Clear History"):
        clear_history()

# Render Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- Chat Execution ---
if user_input := st.chat_input("Type your message here..."):
    # 1. UI update
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. Add to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # 3. Compile prompt dynamically
    current_prompt = ChatPromptTemplate.from_messages([
        ("system", edited_prompt), 
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # 4. Initialize the selected LLM dynamically
    llm = get_llm(selected_model)
    
    if llm:
        chain = current_prompt | llm
        
        # 5. Execute and stream response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {selected_model}..."):
                try:
                    response = chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history[:-1] 
                    })
                    st.markdown(response.content)
                    # 6. Save AI response
                    st.session_state.chat_history.append(AIMessage(content=response.content))
                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")
                    # Remove the failed user message from history to keep state clean
                    st.session_state.chat_history.pop()