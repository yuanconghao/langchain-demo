from langchain.llms import OpenAI
import os

#.bashrc  export OPENAI_API_KEY=sk-your openai api key
#os.environ["OPENAI_API_KEY"] = 'sk-your openai api key'

llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
print(llm("怎么评价人工智能"))
