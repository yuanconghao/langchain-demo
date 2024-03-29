import os
import glob
from utils.embeddings import embedding_pdfs
from utils.embeddings import embedding_markdowns
from utils.embeddings import embedding_51talk

from utils.const import vector_store
from utils.const import data_path
from utils.loader import load_store
from utils.loader import save_store
from utils.loader import new_store


def run_scan_pdfs():
    name = 'pdf'
    index_path = os.path.join(vector_store, name + ".index")
    if os.path.exists(index_path):
        index = load_store(
            dirpath=vector_store,
            name=name,
        )
    else:
        index = new_store()
    total = 0
    while True:
        n = embedding_pdfs(
            index=index,
            fpaths=gen_pdfs(),
            url="data/pdf/",
            replace_by_url=os.path.join(data_path, name),
        )
        total += n
        save_store(
            index=index,
            dirpath=vector_store,
            name=name,
        )

        print(f"scanned {total} files")
        if n == 0:
            return


def run_scan_markdowns():
    name = 'md'
    index_path = os.path.join(vector_store, name + ".index")
    if os.path.exists(index_path):
        index = load_store(
            dirpath=vector_store,
            name=name,
        )
    else:
        index = new_store()
    total = 0
    while True:
        files = gen_markdowns()
        n = embedding_markdowns(
            index=index,
            fpaths=files,
            url="",
            replace_by_url=os.path.join(data_path, name),
        )
        save_store(
            index=index,
            dirpath=vector_store,
            name=name,
        )

        print(f"{n=}")
        if n == 0:
            return


def run_scan_51talk():
    name = '51talk'
    index_path = os.path.join(vector_store, name + ".index")
    if os.path.exists(index_path):
        index = load_store(
            dirpath=vector_store,
            name=name,
        )
    else:
        index = new_store()
    total = 0
    while True:
        files = gen_51talk()
        print(files)
        n = embedding_51talk(
            index=index,
            fpaths=files,
            url="",
            replace_by_url=os.path.join(data_path, name),
        )
        save_store(
            index=index,
            dirpath=vector_store,
            name=name,
        )

        print(f"{n=}")
        if n == 0:
            return


def gen_markdowns():
    pathname = os.path.join(data_path, 'md', '**/*.md')
    yield from glob.glob(pathname, recursive=True)


def gen_pdfs():
    pathname = os.path.join(data_path, 'pdf', '**/*.pdf')
    yield from glob.glob(pathname, recursive=True)


def gen_51talk():
    pathname = os.path.join(data_path, '51talk', '**/*.txt')
    yield from glob.glob(pathname, recursive=True)


if __name__ == '__main__':
    # run_scan_pdfs()
    #run_scan_markdowns()
    run_scan_51talk()
