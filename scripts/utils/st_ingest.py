import streamlit as st
from google.cloud import storage
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

credential_filename = os.getenv("CREDENTIAL_FILENAME")
bucket_name = os.getenv("BUCKET_NAME")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    storage_client = storage.Client.from_service_account_json(credential_filename)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    blob_list = []
    for blob in blobs:
        blob_list.append(blob.name)

    return blob_list


def download_blob(bucket_name, blob_name, dst_path):
    """Downloads a blob into memory."""

    storage_client = storage.Client.from_service_account_json(
        credential_filename
    )  # storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    dir_path = dst_path.rsplit("/", 1)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    blob.download_to_filename(dst_path)


def st_ingest_data():
    blobs = list_blobs_with_prefix(bucket_name, "yfinance/")
    dst_blobs = [blob.replace("yfinance", SOURCE_DOCUMENTS_PATH) for blob in blobs]

    progress_text = "Ingestion in progress. Please wait."
    bar = st.sidebar.progress(0, text=progress_text)
    for i in range(len(blobs)):
        try:
            download_blob(bucket_name, blobs[i], dst_path=dst_blobs[i])
        except:
            continue
        bar.progress(i / len(blobs), text=progress_text)

    bar.empty()


# ingest_data()
