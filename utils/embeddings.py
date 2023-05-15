import os
import codecs
from urllib.parse import quote

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter

from const import Index

N_BACTCH_FILES = 5

text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)


def is_file_scaned(index: Index, fpath):
    """
    is_file_scaned
    :param index:
    :param fpath:
    :return:
    """
    return os.path.split(fpath)[1] in index.scaned_files


def embedding_pdfs(index: Index, fpaths, url, replace_by_url):
    i = 0
    docs = []
    metadatas = []
    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        if is_file_scaned(index, fname):
            continue

        loader = PyPDFLoader(fpath)
        for page, data in enumerate(loader.load_and_split()):
            splits = text_splitter.split_text(data.page_content)
            docs.extend(splits)
            for ichunk, _ in enumerate(splits):
                fnameurl = quote(fpath.removeprefix(replace_by_url), safe="")
                furl = url + fnameurl
                metadatas.append({"source": f"{furl}#page={page + 1}"})

        index.scaned_files.add(fname)
        print(f"scaned {fpath}")
        i += 1
        if i > N_BACTCH_FILES:
            break

    if i != 0:
        index.store.add_texts(docs, metadatas=metadatas)

    return i


def embedding_markdowns(index: Index, fpaths, url, replace_by_url):
    i = 0
    docs = []
    metadatas = []
    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        if is_file_scaned(index, fpath):
            continue

        with codecs.open(fpath, "rb", "utf8") as fp:
            docus = markdown_splitter.create_documents([fp.read()])
            for ichunk, docu in enumerate(docus):
                docs.append(docu.page_content)
                title = quote(docu.page_content.strip().split("\n", maxsplit=1)[0])
                if url:
                    fnameurl = quote(fpath.removeprefix(replace_by_url), safe="")
                    furl = url + fnameurl
                    metadatas.append({"source": f"{furl}#{title}"})
                else:
                    metadatas.append({"source": f"{fname}#{title}"})

        index.scaned_files.add(fname)

        print(f"scaned {fpath}")
        i += 1
        if i > N_BACTCH_FILES:
            break

    if i != 0:
        index.store.add_texts(docs, metadatas=metadatas)

    return i


def embedding_51talk(index: Index, fpaths, url, replace_by_url):
    print(fpaths)
    i = 0
    docs = []
    metadatas = []
    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        if is_file_scaned(index, fname):
            continue

        loader = UnstructuredFileLoader(fpath)
        document = loader.load()
        split_docs = text_splitter.split_documents(document)
        docs.append(split_docs[0].page_content)
        title = quote(split_docs[0].page_content.strip().split("\n", maxsplit=1)[0])
        if url:
            fnameurl = quote(fpath.removeprefix(replace_by_url), safe="")
            furl = url + fnameurl
            metadatas.append({"source": f"{furl}#{title}"})
        else:
            metadatas.append({"source": f"{fname}#{title}"})

        index.scaned_files.add(fname)

        print(f"scaned {fpath}")
        i += 1
        if i > N_BACTCH_FILES:
            break

    if i != 0:
        index.store.add_texts(docs, metadatas=metadatas)

    return i
