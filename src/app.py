from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
import streamlit as st

def init_database(
    user:str, password:str, host:str, port:str, database:str
)-> SQLDatabase:
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)



load_dotenv()

st.set_page_config(page_title="semantic search")

st.title("Semantic Search")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application")
    
    st.text_input("Host", value="aws-0-ap-south-1.pooler.supabase.com", key="Host")
    st.text_input("Port", value="6543", key="Port")
    st.text_input("User", value="postgres.czfqaiamgvhynpguxryw", key="User")
    st.text_input("Password", type="password", value="rzJYOy7SRuR2Szkn", key="Password")
    st.text_input("Database", value="postgres", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello, I am SQL Assistant"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)


user_query = st.chat_input("Type a message...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):# Generate response (replace with actual AI response logic)
        response = "I don't know"  # Replace with your AI response logic
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
    
