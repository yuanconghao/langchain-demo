from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory="/data/chroma", embedding_function=embeddings)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rqa = ConversationalRetrievalChain.from_llm(llm, docsearch.as_retriever(), memory=memory)

def retrieve_answer(query, chat_history):
    memory.chat_memory.add_user_message(query)
    res = rqa({"question": query})
    retrieval_result = res["answer"]

    if "The given context does not provide" in retrieval_result:
        base_result = llm.generate([query])
        return base_result.generations[0][0].text
    else:
        return retrieval_result

messages = []

print("Welcome to the chatbot. Enter 'quit' to exit the program.")
while True:
    user_message = input("You: ")
    if user_message == 'quit':
        break
    if len(messages) == '2':
        messages = []
    answer = retrieve_answer(user_message, messages)
    print("Assistant:", answer)
    messages.append((user_message, answer))