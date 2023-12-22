import json
import logging
import numpy as np
from datetime import datetime

from llm.llm import LLM
from llm.prompt import get_prompt
from agents.memory import AgentMemory, MemoryRetriever

logger = logging.getLogger("MyLogger")

class SocialNetAgent:
    def __init__(self, user_dic: dict, llm: LLM, reflection_threshold: float):
        self.user_prof: dict = user_dic                     # user profile dictionary
        self.memory: AgentMemory = AgentMemory(llm, MemoryRetriever(llm), reflection_threshold)
        self.llm: LLM = llm

    def add_to_memory(self, observation: str, cur_time: datetime):
        self.memory.add_memory(observation, cur_time)

    def generate_post(self, cur_time: datetime):
        logger.info("agent %s generate a post" % self.user_prof["id"])
        # When generating a post, the agent will decide what to post
        prompt_template_path = "llm/prompt_template/post_generation.txt"
        prompt_input = self.user_prof.copy()
        prompt_input["token_limit"] = 150
        retrieved_memories = self.memory.retrive_memories("", cur_time)
        prompt_input["memories"] = '\n'.join(["[%d] %s" % (idx, memory["text"]) for idx, memory in enumerate(retrieved_memories)])
        prompt = get_prompt(prompt_template_path, prompt_input)
        response = self.llm.invoke(prompt)
        post = self.llm.parse_response(response)
        logger.info(prompt + "\n\n" + post)
        # logger.debug(prompt + "\n\n" + post)
        observation = "我在社交平台上发布了一篇文章，文章内容是：" + post # what idea driving the post generation  todo
        self.add_to_memory(observation, cur_time)
        return post

    def react_to_post(self, msg, cur_time: datetime):
        logger.info("agent %s react to a post recieved from agent %s" % (self.user_prof["id"], msg["agent_id"]))
        # When recieving a message, the agent will decide how to react
        prompt_template_path = "llm/prompt_template/react_to_post.txt"
        prompt_input = self.user_prof.copy()
        prompt_input["post_content"] = msg["post_content"]
        retrieved_memories = self.memory.retrive_memories("我看到了一篇文章，文章内容是" + msg["post_content"], \
                                                                cur_time)
        prompt_input["memories"] = '\n'.join(["[%d] %s" % (idx, memory["text"]) for idx, memory in enumerate(retrieved_memories)])
        prompt = get_prompt(prompt_template_path, prompt_input)
        result = self.llm.invoke(prompt)
        logger.info(prompt + "\n\n" + result)
        # logger.debug(prompt + "\n\n" + result)
        # parse the result of llm
        if "不感兴趣" in result:
            observation = "我阅读了一篇来自我的关注者——" + msg["post_author"] + "的文章。" \
                        "文章的内容是：" + msg["post_content"]
            self.add_to_memory(observation, cur_time)
            return {
                "follow": False,
                "repost": False
            }
        else:
            observation = "我阅读了一篇来自我的关注者——" + msg["post_author"] + "的文章。" \
                            "我对这篇文章很感兴趣，文章的内容是：" + msg["post_content"]
            if "转发" in result:
                observation += "\n我转发了这篇文章。" # 转发配文  todo
            self.add_to_memory(observation, cur_time)
            react = {
                "follow": "关注" in result,
                "repost": "转发" in result
            }
            return react
        
    def reset(self):
        self.memory.clear()
        
    def save_to_dict(self):
        return {
            "user_prof": self.user_prof,
            "memory": self.memory.save_to_dict()
        }
    
    def load_from_dict(self, agent_dict):
        self.user_prof = agent_dict["user_prof"]
        self.memory.load_from_dict(agent_dict["memory"])
