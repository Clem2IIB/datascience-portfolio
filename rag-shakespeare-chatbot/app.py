import streamlit as st

# main app file that runs the Streamlit app and handles page navigation
pg = st.navigation([
    st.Page("pages/homepage.py", title="Home", icon="🏠"),
    st.Page("pages/chat_page.py", title="Chat", icon="🎭"),
])
pg.run()

# To launch the chatbot, paste : "streamlit run Chatbot/app.py" in the terminal