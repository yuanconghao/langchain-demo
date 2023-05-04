from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

loader = UnstructuredImageLoader("./data/arch.png")

data = loader.load()
print("=============")
print(data)
print("1111111111111")
print(type(data))
data[0]

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(data)
print(f'documents:{len(split_documents)}')



# 加载llm模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链
chain.run(split_documents[:5])

