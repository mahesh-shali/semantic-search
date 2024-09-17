import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
import streamlit as st

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
db_host = os.getenv("DB_HOSTNAME")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_database = os.getenv("DB_DATABASE")

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=groq_api_key
    )
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, SQL query, and SQL response, write a concise, single-sentence natural language response.
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User Question: {question}
    SQL Response: {response}
    
    Provide a concise answer in one sentence:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0,
        api_key=groq_api_key
    )
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain)
        .assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke({
        "question": user_query,
        "chat_history": chat_history
    })
    
    return result.strip()

# Streamlit UI and integration
st.set_page_config(page_title="Semantic Search")

st.title("Semantic Search")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application")
    
    st.text_input("Host", value=db_host, key="Host")
    st.text_input("Port", value=db_port, key="Port")
    st.text_input("User", value=db_user, key="User")
    st.text_input("Password", type="password", value=db_password, key="Password")
    st.text_input("Database", value=db_database, key="Database")
    
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
        AIMessage(content="Hello, I am SQL Assistant"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    if "db" in st.session_state:
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        with st.chat_message("AI"):
            st.markdown(response)
        st.session_state.chat_history.append(AIMessage(content=response))
    else:
        with st.chat_message("AI"):
            st.markdown("Database is not connected yet.")
