import streamlit as st
import pandas as pd
import pickle as pkl
import os
from dotenv import load_dotenv
import datetime
import time
import scripts.utils.st_ingest as st_ingest
from scripts.utils.helper_unstructured_rag import (
    get_service_context,
    get_query_engine,
    load_docs_and_save_index,
)

load_dotenv()


CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")

DB_URL = os.getenv("DB_URL")
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_UNSTRUCTURED = os.getenv("EXPERIMENT_LOGGER_UNSTRUCTURED")
CHAT_HISTORY_UNSTRUCTURED = os.getenv("CHAT_HISTORY_UNSTRUCTURED")

VECTOR_DB_INDEX = os.getenv("VECTOR_DB_INDEX")


def answer_query(query_engine, query):
    start = time.time()
    response = query_engine.query(query)
    end = time.time()

    return end - start, response


def clean_response(response):
    response = response.replace("$", "\$")
    response = response.replace('"', "'")

    return response


def render(history_file, models, model_names_to_id):
    st.sidebar.divider()
    model = st.sidebar.selectbox("Model", models)

    if model_names_to_id[model].lower() == 'openai':
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
            
    with st.spinner("Initializing App"):
        service_context = get_service_context(
            model_name=model_names_to_id[model], token=TOKEN, cache_dir=CACHE_DIR
        )

        try:
            query_engine = get_query_engine(
                model_name=model_names_to_id[model], service_context=service_context
            )
        except:
            load_docs_and_save_index(
                model_names_to_id[model], service_context=service_context
            )
            query_engine = get_query_engine(
                model_name=model_names_to_id[model], service_context=service_context
            )

    st.sidebar.divider()
    refresh_db = st.sidebar.button(
        "Refresh News", use_container_width=True, help="Might take a while to complete"
    )

    if refresh_db:
        try:
            st_ingest.st_ingest_data()
            load_docs_and_save_index(
                model_names_to_id[model], service_context=service_context
            )

        except:
            st.sidebar.write("Failed to Refresh DB")

    but = st.sidebar.button("Clear History", use_container_width=True)
    if but:
        if os.path.exists(CHAT_HISTORY_UNSTRUCTURED):
            os.remove(CHAT_HISTORY_UNSTRUCTURED)

    try:
        st.session_state.messages = pkl.load(open(history_file, "rb"))
    except:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                with st.expander("Sources"):
                    tabs = st.tabs(
                        [f"Source {i+1}" for i in range(len(message["sources"]))]
                    )
                    for i, source in enumerate(message["sources"]):
                        tabs[i].link_button(source["title"], source["url"])
                        tabs[i].markdown(source["date"])
                        tabs[i].markdown(clean_response(source["text"]))

                        if i == 3:
                            break

            with st.chat_message("⚒️"):
                st.write(f"Time Taken: {message['time']:.3f} seconds")
                st.write(f"Model: {message['model']}")

    if input_query := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": input_query})

        with st.chat_message("user"):
            st.markdown(input_query)

        with st.spinner("Getting Response"):
            time, resp = answer_query(query_engine=query_engine, query=input_query)

        source_nodes = resp.source_nodes
        sources = []
        for node in source_nodes:
            title = node.metadata["title"].replace("_", " ")
            date = title.rsplit(" ", 1)[-1].split(".")[0]
            title = title.rsplit(" ", 1)[0]

            temp = {}
            temp["url"] = ":".join(node.text.split("\n")[0].split(":")[1:])
            temp["title"] = title
            temp["date"] = f"*Dated: {date}*"
            temp["text"] = "\n".join(node.text.rsplit("\n", -2)[2:])

            sources.append(temp)

        resp = clean_response(resp.response)

        with st.chat_message("assistant"):
            st.markdown(resp)

            with st.expander("Sources"):
                tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
                for i, source in enumerate(sources):
                    tabs[i].link_button(source["title"], source["url"])
                    tabs[i].markdown(source["date"])
                    tabs[i].markdown(clean_response(source["text"]))

                    if i == 3:
                        break

        with st.chat_message("⚒️"):
            st.write(f"Time Taken: {time:.3f} seconds")
            st.write(f"Model: {model}")

        timestamp = datetime.datetime.now()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": resp,
                "timestamp": timestamp,
                "time": time,
                "model": model,
                "sources": sources,
            }
        )

        df = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "model": [model],
                "user_input": [input_query],
                "llm_response": [resp],
                "sources": [sources],
                "time_taken": [time],
            }
        )

        try:
            results = pd.read_csv(EXPERIMENT_LOGGER_UNSTRUCTURED)
            results = pd.concat([results, df], axis=0, ignore_index=True)
            results.to_csv(EXPERIMENT_LOGGER_UNSTRUCTURED, index=False)

        except:
            df.to_csv(EXPERIMENT_LOGGER_UNSTRUCTURED, index=False)

    try:
        pkl.dump(st.session_state.messages, open(history_file, "wb"))
    except:
        pass
