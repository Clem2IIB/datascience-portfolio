import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # get the root directory of the project

# -------------------------
# Homepage design
# -------------------------

st.title("Shakespeare Chatbot")
st.caption("Welcome to an AI guide to the complete works of William Shakespeare: 38 plays, the Sonnets and 5 other poems.")

# create two columns, one for the image (1/3 of the width) and one for the text (2/3 of the width)
col1, col2 = st.columns([1, 2]) 

# display the image of Othello and Desdemona with captions
with col1:
    st.image(str(BASE_DIR / "assets" / "othello.jpg"), caption="Othello et Desdemona à Venise, by Théodore Chassériau, 1850")

# display the project description and instructions to use the chatbot
with col2:
    st.markdown("""
                This is a chatbot dedicated to answering any question about Shakespeare's works : themes, characters, quotes, plot and more.
                It started as a chatbot on Othello and now covers the complete works from Project Gutenberg.

                This page is powered by a local LLM and a retrieval system built directly from Shakespeare's text.
                When your question names a play or a character, the chatbot searches in that play, otherwise in all the works.
                You can choose the LLM from the sidebar :
                - Mistral's ministral 3.3b model
                - Google's Gemma 4 E2B 

                This page is built with the Streamlit Python library.
                """)

# display a quote from the play 
st.markdown("> *She loved me for the dangers I had passed, and I loved her that she did pity them.* — Othello, Act I, Scene III")

# display a button to switch to the chat page and start asking questions to the chatbot
_, col, _ = st.columns([2, 1, 2]) # create a centered column for the button
with col:
    if st.button("Try it out!", width="stretch"):
        st.switch_page("pages/chat_page.py")
