"""
通用loader封装工具
"""
import os
import glob
import pickle
import re
import textwrap
import faiss
from const import Index
from sys import path
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

path.append("/opt/configs/ramjet")


def pretty_print(text: str) -> str:
    text = text.strip()
    return textwrap.fill(text, width=60, subsequent_indent="    ")


def is_file_scaned(index: Index, fpath):
    return os.path.split(fpath)[1] in index.scaned_files


def load_store(dirpath, name) -> Index:
    """
    Args:
        dirpath: dirpath to store index files
        name: project/file name
    """
    index = faiss.read_index(f"{os.path.join(dirpath, name)}.index")
    with open(f"{os.path.join(dirpath, name)}.store", "rb") as f:
        store = pickle.load(f)
    store.index = index

    with open(f"{os.path.join(dirpath, name)}.scanedfile", "rb") as f:
        scaned_files = pickle.load(f)

    return Index(
        store=store,
        scaned_files=scaned_files,
    )


def new_store() -> Index:
    store = FAISS.from_texts(["world"], OpenAIEmbeddings(), metadatas=[{"source": "hello"}])
    return Index(
        store=store,
        scaned_files=set([]),
    )


def save_store(index: Index, dirpath, name):
    store_index = index.store.index
    fpath_prefix = os.path.join(dirpath, name)
    print(f"save store to {fpath_prefix}")
    faiss.write_index(store_index, f"{fpath_prefix}.index")
    index.store.index = None
    with open(f"{fpath_prefix}.store", "wb") as f:
        pickle.dump(index.store, f)
    index.store.index = store_index

    with open(f"{fpath_prefix}.scanedfile", "wb") as f:
        pickle.dump(index.scaned_files, f)
