from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('./data/', glob='**/*.txt')
# 将数据转成Document对象，每个文件作为一个Document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的Document
split_docs = text_splitter.split_documents(documents)

# 初始化openai的embeddings对象
embeddings = OpenAIEmbeddings()
# 将Document通过openai的embeddings对象计算embedding向量信息中临时存入chroma向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory="/data/vector_store")
docsearch.persist()
print(docsearch)
