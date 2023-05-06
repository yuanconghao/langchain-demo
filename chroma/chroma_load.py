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
qa = VectorDBQA.from_chain_type(llm=OpenAI(model_name='ada'), chain_type="stuff", vectorstore=docsearch,
                                return_source_documents=True)
# ada进行问答
result = qa({"query": "秦逸是谁"})
print(result)
result = qa({"query": "51talk业务有哪些"})
print(result)
time2 = time.time()
print("model ada cost:", time2 - time1)

# 创建问答对象
time3 = time.time()
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,
                                return_source_documents=True)
# text-davinci-003进行问答
result = qa({"query": "秦逸是谁"})
print(result)
result = qa({"query": "51talk业务有哪些"})
print(result)
time4 = time.time()
print("model text-davinci-003 cost:", time4 - time3)

