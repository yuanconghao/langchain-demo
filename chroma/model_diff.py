from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI

from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
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



# 创建问答对象
time3 = time.time()
# 通过向量存储初始化检索器
retriever = docsearch.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{context}
-----------
{chat_history}
"""

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template('{question}')
]
# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)

# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, max_tokens=1024), retriever, prompt)
result = qa({"query": "为什么飞学英语不可"})
print(result)
time4 = time.time()
print("model gpt3.5 cost:", time4 - time3)

