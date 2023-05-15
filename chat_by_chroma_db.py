import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


system_template = """Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

# 加载文档
loader = DirectoryLoader('./data/51talk', glob='**/*.txt')
docs = loader.load()
# 文档切块
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
doc_texts = text_splitter.split_documents(docs)
# 调用openai Embeddings
embeddings = OpenAIEmbeddings()
# 向量化
vectordb = Chroma.from_documents(doc_texts, embeddings, persist_directory="/data/chroma")
vectordb.persist()
# 创建聊天机器人对象chain
chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb,
                                   return_source_documents=True, condense_question_prompt=prompt)


def get_answer(question, chat_history):
    result = chain({"question": question, "chat_history": chat_history})
    return {'answer':result["answer"], 'sources':result['sources']}


chat_history = []
while True:
    user_message = input("You: ")
    if user_message == 'quit':
        break
    if len(chat_history) == '2':
        chat_history = []
    result = get_answer(user_message, chat_history)
    print("Assistant:", result['answer'])
    print("Sources:", result['sources'])
    chat_history.append((user_message, result['answer']))
