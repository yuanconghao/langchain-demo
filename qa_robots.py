from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# 加载文件夹的所有txt文件
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

# 创建问答对象
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

# 进行问答
result = qa({"query": "秦逸是谁"})
print(result)
result = qa({"query": "51talk业务有哪些"})
print(result)
