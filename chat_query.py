from utils import loader
from utils import const
from urllib.parse import unquote

from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
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

index = loader.load_store(
    dirpath=const.vector_store,
    name="51talk",
)

chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1000)

chain = VectorDBQAWithSourcesChain.from_chain_type(
    llm=llm,
    vectorstore=index.store,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    reduce_k_below_max_tokens=True,
)

while True:
    question = input("Question: ")
    if question == 'quit':
        break

    result = chain(
        {
            "question": question,
        },
        return_only_outputs=True,
    )
    answer = loader.pretty_print(result['answer'])
    sources = unquote(result['sources'])
    print(f"Result: {answer}\n")
    print(f"Sources: {sources}")



# question = "谷歌旗下智能家居业务部门有哪些"
# result = chain(
#     {
#         "question": question,
#     },
#     return_only_outputs=True,
# )
#
# print(f"Question: {question}\n")
# print(f"Result: {loader.pretty_print(result['answer'])}\n")
# print(f"Sources: {result['sources']}")
