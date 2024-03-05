from scripts.utils.helper_structured_rag import (
    get_database,
    get_service_context,
    get_query_engine,
    get_response,
    delete_database,
)
import streamlit as st
import pandas as pd
import pickle as pkl
import os
from dotenv import load_dotenv
import datetime
import sqlite3

load_dotenv()


TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("CACHE_DIR")

DB_URL = os.getenv("DB_URL")
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_STRUCTURED = os.getenv("EXPERIMENT_LOGGER_STRUCTURED")
CHAT_HISTORY_STRUCTURED = os.getenv("CHAT_HISTORY_STRUCTURED")


def answer_query(query_engine, query):
    time, resp, sql = get_response(
        query_engine=query_engine, query_str=query, print_=False
    )
    return time, resp, sql


def clean_response(response):
    response = response.replace("$", "\$")
    response = response.replace('"', "'")

    return response


def render(history_file, models, model_names_to_id, portfolios):
    st.sidebar.divider()

    model = st.sidebar.selectbox("Choose Model", models)
    if model_names_to_id[model].lower() == "openai":
        openai_key = st.sidebar.text_input("OpenAI API Key")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        else:
            st.info("Please Enter OpenAI's API Key to continue.")
            st.stop()

    elif model == "Enter HuggingFace ID":
        model_id = st.sidebar.text_input("Enter HuggingFace Model ID")
        if model_id:
            model_names_to_id[model_id] = model_id
            model = model_id
        else:
            st.info("Please Enter HuggingFace Model ID to continue.")
            st.stop()

    portfolio = st.sidebar.selectbox("Choose Portfolio", portfolios)

    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=portfolio)

    with st.expander("Portfolio", expanded=False):
        st.dataframe(df, use_container_width=True)

    st.divider()

    with st.spinner("Initializing Database"):
        sql_database = get_database(
            DB_URL,
            EXCEL_FILE_PATH=EXCEL_FILE_PATH,
            portfolio=portfolio,
            SOURCE_DOCUMENTS_PATH=SOURCE_DOCUMENTS_PATH,
            csv_path=ASSET_MAPPING_PATH,
        )

        service_context = get_service_context(
            model_names_to_id[model], token=TOKEN, cache_dir=CACHE_DIR
        )

        query_engine = get_query_engine(
            sql_database=sql_database, service_context=service_context
        )

    st.sidebar.divider()
    but = st.sidebar.button("Clear History", use_container_width=True)
    if but:
        if os.path.exists(CHAT_HISTORY_STRUCTURED):
            os.remove(CHAT_HISTORY_STRUCTURED)

    try:
        st.session_state.messages = pkl.load(open(history_file, "rb"))
    except:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(clean_response(message["content"]))

        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                with st.expander("SQL", expanded=False):
                    st.code(message["sql"], language="sql")
                    if type(message["df_sql"]) == str:
                        st.write(message["df_sql"])
                    else:
                        st.dataframe(message["df_sql"], hide_index=True)

            with st.chat_message("⚒️"):
                st.write(f"Time Taken: {message['time']:.3f} seconds")
                st.write(f"Model: {message['model']}")
                st.write(f"Portfolio: {message['portfolio']}")

    if input_query := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": input_query})

        with st.chat_message("user"):
            st.markdown(input_query)

        with st.spinner("Getting Response"):
            time, resp, sql = answer_query(query_engine=query_engine, query=input_query)

        try:
            conn = sqlite3.connect(DB_URL)
            df_sql = pd.read_sql(sql, con=conn)
            conn.close()
        except:
            df_sql = "Incorrect Query!"

        with st.chat_message("assistant"):
            st.markdown(clean_response(resp))

            with st.expander("SQL", expanded=True):
                st.code(sql, language="sql")
                if type(df_sql) == str:
                    st.write(df_sql)
                else:
                    st.dataframe(df_sql, hide_index=True)

        with st.chat_message("⚒️"):
            st.write(f"Time Taken: {time:.3f} seconds")
            st.write(f"Model: {model}")
            st.write(f"Portfolio: {portfolio}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "model": model,
                "portfolio": portfolio,
                "content": resp,
                "sql": sql,
                "time": time,
                "df_sql": df_sql,
            }
        )

        df = pd.DataFrame(
            {
                "timestamp": [datetime.datetime.now()],
                "model": [model],
                "portfolio": [portfolio],
                "user_input": [input_query],
                "llm_response": [resp],
                "sql_query": [sql],
                "time_taken": [time],
            }
        )

        try:
            results = pd.read_csv(EXPERIMENT_LOGGER_STRUCTURED)
            results = pd.concat([results, df], axis=0, ignore_index=True)
            results.to_csv(EXPERIMENT_LOGGER_STRUCTURED, index=False)

        except:
            df.to_csv(EXPERIMENT_LOGGER_STRUCTURED, index=False)

    try:
        pkl.dump(st.session_state.messages, open(history_file, "wb"))
    except:
        pass
