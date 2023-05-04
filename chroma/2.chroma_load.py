from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA

# 初始化openai的embeddings对象
embeddings = OpenAIEmbeddings()
# 加载数据
docsearch = Chroma(persist_directory="/data/vector_store", embedding_function=embeddings)

# 创建问答对象
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

# 进行问答
result = qa({"query": "秦逸是谁"})
print(result)
result = qa({"query": "51talk业务有哪些"})
print(result)
