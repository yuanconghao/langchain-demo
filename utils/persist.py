import os
import glob
from embeddings import embedding_pdfs
from embeddings import embedding_markdowns

from const import vector_store
from const import data_path
from loader import load_store
from loader import save_store
from loader import new_store


def run_scan_pdfs():
    name = 'pdf'
    index_path = os.path.join(vector_store, name+".index")
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
            replace_by_url=os.path.join(data_path, 'pdf'),
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
            url="data/md/",
            replace_by_url=os.path.join(data_path, 'md'),
        )
        save_store(
            index=index,
            dirpath=vector_store,
            name="md",
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


if __name__ == '__main__':
    #run_scan_pdfs()
    run_scan_markdowns()
