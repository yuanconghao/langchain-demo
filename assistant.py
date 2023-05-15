from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory="/data/vector_store", embedding_function=embeddings)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rqa = ConversationalRetrievalChain.from_llm(llm, docsearch.as_retriever(), memory=memory)

def retrieve_answer(query, chat_history):
    memory.chat_memory.add_user_message(query)
    result = rqa({"question": query})

    sources = []
    if result['source_documents'] is not None:
        for source in result['source_documents']:
            sources.append(source.metadata['source'])
    return {'answer': result["answer"], 'sources': sources}


chat_history = []
while True:
    user_message = input("You: ")
    if user_message == 'quit':
        break
    if len(chat_history) == '2':
        chat_history = []
    result = retrieve_answer(user_message, chat_history)
    print("Assistant:", result['answer'])
    print("Sources:", result['sources'])
    chat_history.append((user_message, result['answer']))