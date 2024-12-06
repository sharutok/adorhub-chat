from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import os
from urllib.parse import quote_plus

try:
    load_dotenv()

    def init_database(
        user: str, password: str, host: str, port: str, database: str
    ) -> SQLDatabase:
        db_uri = f"oracle+cx_oracle://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/?service_name={quote_plus(database)}"
        return SQLDatabase.from_uri(db_uri)

    def get_sql_chain(db):
        template = """
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
      
      <SCHEMA>{schema}</SCHEMA>
      
      Conversation History: {chat_history}
      
      Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks and semicolon at the end.
      
      For example:
      Question: how many users are there?
      SQL Query: select count(*) from user_permission_master upm
      Question: what is total estimated man hours where punch in and puch out is not null
      SQL Query: select count(estimated_man_hrs) from assign_work_to_tradesmen awtt where punch_in is not null and punch_out is not null
      
      Your turn:
      
      Question: {question}
      SQL Query:
      """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

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
      Based on the table schema below, question, sql query, and sql response, write a natural language response.
      <SCHEMA>{schema}</SCHEMA>

      Conversation History: {chat_history}
      SQL Query: <SQL>{query}</SQL>
      User question: {question}
      SQL Response: {response}"""

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

        chain = (
          RunnablePassthrough.assign(query=sql_chain).assign(
              schema=lambda _: db.get_table_info(),
              response=lambda vars: db.run(vars["query"]),
          )
          | prompt
          | llm
          | StrOutputParser()
      )

        return chain.invoke({
      "question": user_query,
      "chat_history": chat_history,
    })

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
      ]

    st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

    st.title("Chat with MySQL")

    with st.sidebar:
        st.subheader("Settings")
        st.write("This is a simple chat application. Connect to the database and start chatting.")

        st.text_input("Host", value="172.17.2.226", key="Host")
        st.text_input("Port", value="1521", key="Port")
        st.text_input("User", value="AWL_INVBOT", key="User")
        st.text_input(
            "Password", type="password", value="ADor#$123", key="Password"
        )
        st.text_input("Database", value="DEV", key="Database")

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

        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))
except Exception as e:
    print("ERROR!!!",e)


# streamlit run app.py
