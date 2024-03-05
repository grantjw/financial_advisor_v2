import streamlit as st
import pandas as pd
import pickle as pkl
import os
from dotenv import load_dotenv
import datetime
import time

import torch

from llama_index.llms import HuggingFaceLLM
import transformers
import torch
import time
from llama_index.llms import OpenAI

load_dotenv()

CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")

DB_URL = os.getenv("DB_URL")
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_CHATBOT = os.getenv("EXPERIMENT_LOGGER_CHATBOT")
CHAT_HISTORY_CHATBOT = os.getenv("CHAT_HISTORY_CHATBOT")

VECTOR_DB_INDEX = os.getenv("VECTOR_DB_INDEX")


@st.cache_resource
def get_llm(model_name, token, cache_dir, temperature=0.5, max_new_tokens=500):
    if model_name.lower() == "openai":
        llm = OpenAI(
            temperature=temperature,
            model="gpt-3.5-turbo",
            max_tokens=max_new_tokens,
        )
        return llm

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=cache_dir
    )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=token,
        trust_remote_code=True,
        cache_dir=cache_dir,
        pad_token_id=tokenizer.eos_token_id,
    )

    with st.spinner("Loading HuggingFaceLM"):
        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=max_new_tokens,
            generate_kwargs={"temperature": temperature},
            tokenizer=tokenizer,
            model_name=model_name,
            device_map="cuda:0",
            model_kwargs={
                "trust_remote_code": True,
                "config": model_config,
                "quantization_config": bnb_config,
                "use_auth_token": token,
                "cache_dir": cache_dir,
            },
        )

    return llm


def answer_query(model_name, llm, query):
    start = time.time()
    if model_name.lower() == "openai":
        resp = llm.complete(query).text

    else:
        query = "[INST] " + query + " [/INST]"
        resp = llm.complete(query).text

    return time.time() - start, resp


def clean_response(response):
    response = response.replace("$", "\$")
    return response


def render(history_file, models, model_names_to_id):
    st.sidebar.divider()
    model = st.sidebar.selectbox("Model", models)

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
    # temperature = st.sidebar.slider(
    #     "Model *temperature*",
    #     min_value=0.0,
    #     max_value=1.0,
    #     step=0.1,
    #     value=0.7,
    # )

    # max_new_tokens = st.sidebar.slider(
    #     "Model *max_new_tokens*",
    #     min_value=10,
    #     max_value=1000,
    #     step=10,
    #     value=100,
    # )

    try:
        st.session_state.messages = pkl.load(open(history_file, "rb"))
    except:
        st.session_state.messages = []

    with st.spinner("Loading Model"):
        try:
            llm = get_llm(
                model_name=model_names_to_id[model],
                token=TOKEN,
                cache_dir=CACHE_DIR,
                # temperature=temperature,
                # max_new_tokens=int(max_new_tokens),
            )

        except:
            get_llm.clear()

    st.sidebar.divider()
    but = st.sidebar.button("Clear History", use_container_width=True)
    if but:
        if os.path.exists(CHAT_HISTORY_CHATBOT):
            os.remove(CHAT_HISTORY_CHATBOT)

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(clean_response(message["content"]))

        else:
            with st.chat_message(message["role"]):
                st.markdown(clean_response(message["content"]))

            with st.chat_message("⚒️"):
                st.write(f"Time Taken: {message['time']:.3f} seconds")
                st.write(f"Model: {message['model']}")
                # st.write(f"Model Temperature: {message['temperature']}")

    if input_query := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": input_query})

        with st.chat_message("user"):
            st.markdown(input_query)

        with st.spinner("Getting Response"):
            time, resp = answer_query(
                model_names_to_id[model], llm=llm, query=input_query
            )

        with st.chat_message("assistant"):
            st.markdown(clean_response(resp))

        with st.chat_message("⚒️"):
            st.write(f"Time Taken: {time:.3f} seconds")
            st.write(f"Model: {model}")
            # st.write(f"Model Temperature: {temperature}")

        timestamp = datetime.datetime.now()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": resp,
                "timestamp": timestamp,
                "time": time,
                "model": model,
                # "temperature": temperature,
            }
        )

        df = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "model": [model],
                "user_input": [input_query],
                "llm_response": [resp],
                "time_taken": [time],
                # "model_temperature": [temperature],
            }
        )

        try:
            results = pd.read_csv(EXPERIMENT_LOGGER_CHATBOT)
            results = pd.concat([results, df], axis=0, ignore_index=True)
            results.to_csv(EXPERIMENT_LOGGER_CHATBOT, index=False)

        except:
            df.to_csv(EXPERIMENT_LOGGER_CHATBOT, index=False)

    try:
        pkl.dump(st.session_state.messages, open(history_file, "wb"))
    except:
        pass
