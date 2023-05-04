from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
import os

#os.environ["OPENAI_API_KEY"] = 'sk-your openai api key'
os.environ["SERPAPI_API_KEY"] = 'd45cee179361cfdb47b85b21524b8fce59c66a6e760c24c5e12cd2a9497befe3'

# 加载 OpenAI 模型
llm = OpenAI(temperature=0, max_tokens=2048)

# 加载 serpapi 工具
tools = load_tools(["serpapi"])
# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行 agent
agent.run("百度今日股价是多少，分析一下股价走势")
