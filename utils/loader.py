import os
import pickle
import textwrap
import faiss

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

#from const import Index
from utils.const import Index


def pretty_print(text: str) -> str:
    """
    分段输出
    :param text:
    :return:
    """
    text = text.strip()
    #return textwrap.fill(text, width=60, subsequent_indent="")
    return text


def is_file_scaned(index: Index, fpath):
    return os.path.split(fpath)[1] in index.scaned_files


def load_store(dirpath, name) -> Index:
    """
    load_store
    :param dirpath: to store index files
    :param name: project/file name
    :return:
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
    print(store)
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


if __name__ == '__main__':
    ss = "在去年一季度的财报中，51talk就宣布对旗下大陆业务和境外业务进行拆分并完成运营重组，大陆业务从上市公司剥离，将重点发展境外青少英语业务，并成为一家海外上市的全球互联网教育公司。"
    print(pretty_print(ss))
