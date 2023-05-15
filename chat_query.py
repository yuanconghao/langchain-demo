import os
import glob
import codecs
import pickle
import re
import textwrap
from collections import namedtuple

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS
from pymongo import MongoClient

from sys import path
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

index = load_store(
    dirpath="/data/vector_store/index",
    name="security",
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

question = "Asher学习心得有哪些"
result = chain(
    {
        "question": question,
    },
    return_only_outputs=True,
)

print(f"Question: {question}\n")
print(f"Result: {pretty_print(result['answer'])}\n")
print(f"Sources: {result['sources']}")
