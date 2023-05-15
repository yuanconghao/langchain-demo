import os
from collections import namedtuple

Index = namedtuple("index", ["store", "scaned_files"])
data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(data_path, 'data')
vector_store = "/data/vector_store/index"
