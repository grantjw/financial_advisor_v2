# import scripts.pages.structured_rag as structured_rag
# import scripts.pages.unstructured_rag as unstructured_rag
# import scripts.pages.auto_rag as auto_rag
# import scripts.pages.auto_rag as auto_rag
import torch
from scripts.pages import structured_rag, unstructured_rag, auto_rag, chatbot
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
import streamlit as st
import datetime
import plotly.express as px
import pickle as pkl

load_dotenv()

st.set_page_config(layout="wide", page_title="FinAdvisor", page_icon="home.png")


DB_URL = os.getenv("DB_URL")

CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")

MISTRAL_7B_INSTRUCT = os.getenv("MISTRAL_7B_INSTRUCT")
FINETUNED_MISTRAL = os.getenv("FINETUNED_MISTRAL")
DPO_MISTRAL = os.getenv("DPO_MISTRAL")
LLAMA2_7B_CHAT_HF = os.getenv("LLAMA2_7B_CHAT_HF")

EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_AUTO = os.getenv("EXPERIMENT_LOGGER_AUTO")
EXPERIMENT_LOGGER_STRUCTURED = os.getenv("EXPERIMENT_LOGGER_STRUCTURED")
EXPERIMENT_LOGGER_UNSTRUCTURED = os.getenv("EXPERIMENT_LOGGER_UNSTRUCTURED")
EXPERIMENT_LOGGER_CHATBOT = os.getenv("EXPERIMENT_LOGGER_CHATBOT")

CHAT_HISTORY_AUTO = os.getenv("CHAT_HISTORY_AUTO")
CHAT_HISTORY_STRUCTURED = os.getenv("CHAT_HISTORY_STRUCTURED")
CHAT_HISTORY_UNSTRUCTURED = os.getenv("CHAT_HISTORY_UNSTRUCTURED")
CHAT_HISTORY_CHATBOT = os.getenv("CHAT_HISTORY_CHATBOT")

VECTOR_DB_INDEX = os.getenv("VECTOR_DB_INDEX")
GRAPH_DB_INDEX = os.getenv("GRAPH_DB_INDEX")

PORTFOLIOS = [
    "low risk",
    "moderate risk",
    "medium risk",
    "high risk",
]

MODELS = [
    "Mistral-7B-Instruct",
    "Mistral-7B-Instruct-FT",
    "Mistral-7B-Instruct-DPO",
    "Llama2-7B-Chat-HF",
    "GPT-3.5-Turbo",
    "Enter HuggingFace ID",
]

MODEL_NAMES_TO_ID = {
    "Mistral-7B-Instruct": MISTRAL_7B_INSTRUCT,
    "Mistral-7B-Instruct-FT": FINETUNED_MISTRAL,
    "Mistral-7B-Instruct-DPO": DPO_MISTRAL,
    "Llama2-7B-Chat-HF": LLAMA2_7B_CHAT_HF,
    "GPT-3.5-Turbo": "OpenAI",
    "Enter HuggingFace ID": "",
}


import gc


def clear_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # print(type(obj))
                del obj
            pass

        except:
            pass

    gc.collect()
    torch.cuda.empty_cache()


def create_path(path_):
    if not os.path.exists(path_.rsplit("/", 1)[0]):
        os.makedirs(path_.rsplit("/", 1)[0])
    return


create_path(EXPERIMENT_LOGGER_AUTO)
create_path(EXPERIMENT_LOGGER_STRUCTURED)
create_path(EXPERIMENT_LOGGER_UNSTRUCTURED)
create_path(EXPERIMENT_LOGGER_CHATBOT)
create_path(DB_URL)
create_path(CACHE_DIR)
create_path(SOURCE_DOCUMENTS_PATH)
create_path(VECTOR_DB_INDEX)
create_path(GRAPH_DB_INDEX)

pages = [
    "Auto RAG",
    "General Chatbot",
    "Unstructured RAG",
    "Structured RAG",
    "History",
]

page = st.sidebar.radio("Choose Page", pages)

st.image("home.png", width=100)
st.header("FinAdvisor")
st.write("*Welcome, ask away!*")

if page == "Auto RAG":
    torch.cuda.empty_cache()
    st.sidebar.divider()
    st.sidebar.write("Intent-based RAG over structured or unstructured Data.")

    try:
        auto_rag.render(
            history_file=CHAT_HISTORY_AUTO,
            models=MODELS,
            model_names_to_id=MODEL_NAMES_TO_ID,
            portfolios=PORTFOLIOS,
        )

    except:
        clear_memory()
        st.error("CUDA Out of Memory: Please restart the app")


elif page == "General Chatbot":
    torch.cuda.empty_cache()
    st.sidebar.divider()
    st.sidebar.write(
        "General-purpose Chatbot with text-generation capabilities. No context is required."
    )

    try:
        chatbot.render(
            history_file=CHAT_HISTORY_CHATBOT,
            models=MODELS,
            model_names_to_id=MODEL_NAMES_TO_ID,
        )

    except:
        clear_memory()
        st.error("CUDA Out of Memory: Please restart the app")

elif page == "Unstructured RAG":
    torch.cuda.empty_cache()
    st.sidebar.divider()
    st.sidebar.write(
        "RAG over news articles for multiple Stock market tickers using Vector Stores."
    )

    try:
        unstructured_rag.render(
            history_file=CHAT_HISTORY_UNSTRUCTURED,
            models=MODELS,
            model_names_to_id=MODEL_NAMES_TO_ID,
        )
    except:
        clear_memory()
        st.error("CUDA Out of Memory: Please restart the app")

elif page == "Structured RAG":
    torch.cuda.empty_cache()
    st.sidebar.divider()
    st.sidebar.write(
        "RAG over historical stock prices for multiple Stock market tickers using Natural Language to SQL Approach."
    )

    try:
        structured_rag.render(
            history_file=CHAT_HISTORY_STRUCTURED,
            models=MODELS,
            model_names_to_id=MODEL_NAMES_TO_ID,
            portfolios=PORTFOLIOS,
        )
    except:
        clear_memory()
        st.error("CUDA Out of Memory: Please restart the app")

elif page == "History":
    torch.cuda.empty_cache()
    rag_type = st.sidebar.selectbox(
        "Page", ["Auto RAG", "Unstructured RAG", "Structured RAG"]
    )
    experiment_loggers = {
        "Auto RAG": EXPERIMENT_LOGGER_AUTO,
        "Unstructured RAG": EXPERIMENT_LOGGER_UNSTRUCTURED,
        "Structured RAG": EXPERIMENT_LOGGER_STRUCTURED,
    }

    try:
        st.markdown("#### " + rag_type)
        df = pd.read_csv(experiment_loggers[rag_type])

        df["response_length"] = df["llm_response"].apply(lambda x: len(x))

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="time_taken")
            fig.update_xaxes(title="Time Taken (s)")
            fig.update_layout(title="Distribution of Time Taken to Generate Response")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x="response_length")
            fig.update_xaxes(title="Response Length")
            fig.update_layout(title="Distribution of Response Length")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df, x="response_length", y="time_taken")
        fig.update_xaxes(title="Response Length")
        fig.update_yaxes(title="Time Taken (s)")
        fig.update_layout(title="Time Taken with Response Length")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)
    except:
        st.write("No History Found")


# st.write(st.session_state)
