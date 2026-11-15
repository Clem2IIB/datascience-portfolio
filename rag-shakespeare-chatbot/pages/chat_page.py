import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import os
import openai
import anthropic

BASE_DIR = Path(__file__).resolve().parent.parent
SHAKESPEARE_AVATAR = Image.open(BASE_DIR / "assets" / "shakespeare_avatar.jpg") # add a Shakespeare avatar for the chatbot
USER_AVATAR = Image.open(BASE_DIR / "assets" / "user_avatar.jpg") # add a user avatar for the user's messages
INITIAL_MESSAGE = "What do you want to learn about Shakespeare's plays and poems?" # initial message from the chatbot when the conversation starts

# Add project root to path so chat.py can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from chat import get_response, choose_works, AUTO, ALL_WORKS, WORK_TITLES, LM_STUDIO, CLAUDE_API

# -------------------------
# Sidebar settings
# -------------------------

# Clear chat history and start a new conversation
if st.sidebar.button("New chat", width="stretch"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": INITIAL_MESSAGE
        }
    ]
    st.session_state.last_works = [] # a new conversation does not continue the play of the previous one
    st.rerun()
st.sidebar.caption("Clears the current conversation to start a new one.")

# add a horizontal line to separate the "New chat" button from the settings
st.sidebar.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)


st.sidebar.markdown("### Settings")

# Where the LLM runs: locally with LM Studio, or with the Claude API
provider = st.sidebar.radio("Where should the model run?", [LM_STUDIO, CLAUDE_API])

if provider == LM_STUDIO:
    # Model selector from LM Studio with Ministral and Gemma
    model_options = {
        "Ministral 3 3b": "mistralai/ministral-3-3b",
        "Gemma 4 E2B": "google/gemma-4-e2b"
    }
else:
    # Model from the Claude API: only Haiku, the cheapest Claude model, enough for a test project
    model_options = {
        "Claude Haiku 5.5": "claude-haiku-5-5"
    }
selected_model = st.sidebar.selectbox("Which model would you like to use?", list(model_options.keys()))
llm_model = model_options[selected_model] # identifier of the chosen model

st.sidebar.caption("Only the last 4 messages are kept in chat memory.")

# Work selector: detect the play from the question, search all works, or choose one work
search_choice = st.sidebar.selectbox("Where should I search?", [AUTO, ALL_WORKS] + WORK_TITLES)
st.sidebar.caption("Auto finds the play from the titles and character names in your question.")

# Number of chunks to retrieve from the vector database slider
n_chunks = st.sidebar.slider("Number of chunks", min_value=3, max_value=20, value=8)

# Add the chunks before and after each retrieved chunk, to give the rest of the dialogue
neighbours = st.sidebar.checkbox("Add the surrounding passages", value=True)
st.sidebar.caption("More chunks and surrounding passages give more context but slower answers.")


# -------------------------
# Main chat interface
# -------------------------

st.title("Shakespeare Chatbot")
st.caption("Ask me anything about Shakespeare's plays and poems : themes, characters, quotes, and more...")

# Initialize conversation history and the works used for the last question
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": INITIAL_MESSAGE
        }
    ]
if "last_works" not in st.session_state:
    st.session_state.last_works = []

# Display full conversation history, with where each answer searched and the passages it used
for message in st.session_state.messages:
    avatar = SHAKESPEARE_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "search_info" in message:
            st.caption(message["search_info"])
            with st.expander("Passages used"):
                st.text(message["context"])


# Chat input for the user to ask questions, which triggers the retrieval and response generation process
if user_query := st.chat_input("What do you want to know about Shakespeare's works?"):

    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    # Choose the works to search in, then show it under the answer
    works, reason = choose_works(user_query, search_choice, st.session_state.last_works)
    search_info = f"Searched in: {', '.join(works) if works else 'all works'} ({reason})"

    # Get and display assistant response
    with st.chat_message("assistant", avatar=SHAKESPEARE_AVATAR):
        # without a key the Claude API cannot answer, so stop before searching
        if provider == CLAUDE_API and not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("Put your Anthropic API key in a .env file in the project folder (ANTHROPIC_API_KEY=...), then restart the app.")
            st.stop()

        with st.spinner("Thinking..."):
            try:
                response, context = get_response(user_query, history=st.session_state.messages, model=llm_model,
                                                 n_chunks=n_chunks, works=works, neighbours=neighbours,
                                                 provider=provider)
            except openai.APIConnectionError:
                # the LLM runs in LM Studio: without its local server there is no answer
                st.error("I cannot reach LM Studio. Start its local server on port 1234 with the selected model loaded.")
                st.stop()
            except anthropic.AuthenticationError:
                st.error("The Anthropic API key is not valid. Check it in the .env file.")
                st.stop()
            except anthropic.APIConnectionError:
                st.error("I cannot reach the Claude API. Check your internet connection.")
                st.stop()
        st.markdown(response)
        st.caption(search_info)
        with st.expander("Passages used"):
            st.text(context)

    # Update conversation history in session state to keep track of the full conversation and provide context for the LLM
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": response, "search_info": search_info, "context": context})
    st.session_state.last_works = works
