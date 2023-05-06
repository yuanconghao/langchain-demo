from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
import time


# 初始化openai的embeddings对象
embeddings = OpenAIEmbeddings()
# 加载数据
docsearch = Chroma(persist_directory="/data/vector_store", embedding_function=embeddings)

# 创建问答对象
time1 = time.time()
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,
                                return_source_documents=True)
# text-davinci-003进行问答
result = qa({"query": "为什么飞学英语不可"})
print(result)
time2 = time.time()
print("model text-davinci-003 cost:", time2 - time1)
