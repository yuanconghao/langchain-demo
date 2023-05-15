import os
import glob
from loader import load_store
from loader import save_store
from loader import new_store
from embeddings import embedding_pdfs
from embeddings import embedding_markdowns

data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(data_path, 'data')
vector_store = "/data/vector_store/index"


def run_scan_pdfs():
    name = 'pdf'
    index_path = os.path.join(vector_store, name+".index")
    print(index_path)
    if os.path.exists(index_path):
        index = load_store(
            dirpath=vector_store,
            name=name,
        )
    else:
        index = new_store()
    print(index)
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
    #         index = new_store()
    while True:
        index = load_store(
            dirpath=vector_store,
            name="security",
        )
        files = gen_markdowns()
        n = embedding_markdowns(
            index=index,
            fpaths=files,
            url="https://s3.laisky.com/public/papers/security/",
            replace_by_url=os.path.join(data_path, 'md'),
        )
        save_store(
            index=index,
            dirpath=vector_store,
            name="security",
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
    run_scan_pdfs()
