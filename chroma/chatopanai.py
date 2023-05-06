from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI

from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 初始化openai的embeddings对象
embeddings = OpenAIEmbeddings()
# 加载数据
docsearch = Chroma(persist_directory="/data/vector_store", embedding_function=embeddings)

# 创建问答对象
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
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, max_tokens=1024)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, prompt, memory=memory)


def retrieve_answer(query, chat_history):
    memory.chat_memory.add_user_message(query)
    res = qa({"question": query})
    retrieval_result = res["answer"]

    if "The given context does not provide" in retrieval_result:
        base_result = llm.generate([query])
        return base_result.generations[0][0].text
    else:
        return retrieval_result


messages = []
while True:
    user_message = input("You：")
    answer = retrieve_answer(user_message, messages)
    print("Assistant：", answer)
    messages.append((user_message, answer))
