# agents/recommender_agent.py

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from tools.recommend_tool import recommend_videos
# recommender_agent.py
import os
from dotenv import load_dotenv

load_dotenv()  # 自动从 .env 加载环境变量


# === 用 LangChain @tool 装饰器注册函数（可选，或者直接用 Tool(...)） ===
@tool
def recommend(user_tags: list[str]) -> list[dict]:
    """推荐视频：根据用户感兴趣的标签列表（如 ['猫', '搞笑']）返回视频推荐列表"""
    return recommend_videos(user_tags)

# === 构造 Tool 列表 ===
tools = [
    Tool.from_function(
        func=recommend,
        name="VideoRecommender",
        description="基于用户兴趣关键词推荐视频，输入为标签列表，例如 ['搞笑', '足球']"
    )
]

# === 构建 Agent ===
llm = ChatOpenAI(
    temperature=0,
    model="mistralai/mistral-7b-instruct",  # 推荐免费模型之一
)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# === 对外暴露 Agent 实例 ===
def get_recommender_agent():
    return agent
