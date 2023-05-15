import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader

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
                                   return_source_documents=True)


def get_answer(question, chat_history):
    result = chain({"question": question, "chat_history": chat_history});
    return result["answer"]


chat_history = []

print("Welcome to the chatbot. Enter 'quit' to exit the program.")
while True:
    user_message = input("You: ")
    if user_message == 'quit':
        break
    if len(chat_history) == '2':
        chat_history = []
    answer = get_answer(user_message, chat_history)
    print("Assistant:", answer)
    chat_history.append((user_message, answer))
