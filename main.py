import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
from yacs.config import CfgNode

from data.data import Data
from agents.agent import SocialNetAgent
from llm.llm import *
from utils.interval import *
# from recommender.recommender import Recommender

with open('log.txt', 'w') as f:
    pass
# logging.basicConfig(format='---%(asctime)s %(levelname)s \n%(message)s ---\n\n', level=logging.INFO)
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter("---%(asctime)s %(levelname)s \n%(message)s ---\n\n"))
logger.addHandler(stdout_handler)
logfile_handler = logging.FileHandler("log.txt")
logfile_handler.setLevel(logging.DEBUG)
logfile_handler.setFormatter(logging.Formatter("---%(asctime)s %(levelname)s \n%(message)s ---\n\n"))
logger.addHandler(logfile_handler)


class Simulator():
    def __init__(self, config: dict):
        self.config: dict = config
        self.round_num: int = config["round_num"]
        self.round_cnt: int = 0
        self.log_folder: str = config["log_folder"]

        self.agents: List[SocialNetAgent] = []
        self.id2agent: Dict[str, SocialNetAgent] = dict()
        self.recommender = None
        self.logs: List[dict] = [{
            "log_num": 0,
            "log_content": []
        } for _ in range(config["round_num"])]
        self.cur_time: datetime = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval: Interval = parse_interval(config["interval"])
        self.data: Data = Data(config)

        self.load_data()
        self.load_agents()
        self.create_recommender()

    def load_data(self):
        self.data.load_users()
        self.data.load_network()

    def load_agents(self):
        if self.config["llm"] == "zhipuai":
            llm = ZhipuAi(self.config["zhipuai_api_key"])
        elif self.config["llm"] == "llama2":
            llm = Llama2(self.config["llama2_model_path"])
        elif self.config["llm"] == "chatglm3":
            llm = ChatGlm3(self.config["chatglm3_model_path"])
        for user_prof in self.data.users:
            self.agents.append(
                SocialNetAgent(user_prof, llm, self.config["reflection_thershold"])
            )
            self.id2agent[user_prof["id"]] = self.agents[-1]

    def create_recommender(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def run_round(self, round_cnt):
        logger.info("Round %d: %s", round_cnt, format_time(self.cur_time))

        msgs = self.logs[round_cnt - 1]["log_content"] if round_cnt > 0 else []
        # all agents receive messages and react to posts and reposts
        for msg in msgs:
            if msg["type"] != "post" and msg["type"] != "repost":
                continue
            agent_id = msg["agent_id"]
            for neibor_agent_id in self.data.network[agent_id]:
                logger.info("agent %d recieve msg from agent %d"%(neibor_agent_id, agent_id))
                neibor_agent = self.id2agent[neibor_agent_id]
                react = neibor_agent.react_to_post(msg, self.cur_time)
                if react["repost"]:
                    new_msg = {
                        "type": "repost",
                        "agent_id": neibor_agent.user_prof["id"],
                        "origin_agent_id": msg["origin_agent_id"],
                        "post_author": msg["post_author"],
                        "post_content": msg["content"],
                        "time": self.cur_time
                    }
                    self.logs[round_cnt]["log_num"] += 1
                    self.logs[round_cnt]["log_content"].append(new_msg)
                if react["follow"]:
                    pass # must have followed
                    # new_msg = {
                    #     "type": "follow",
                    #     "agent_id": neibor_agent.user_prof["id"],
                    #     "target_agent_id": agent_id,
                    #     "time": self.cur_time
                    # }
                    # self.data.network[agent_id].append(neibor_agent.user_prof["id"])

        # recommend system recommend posts to agents
        for agent in self.agents:
            pass

        # random sample some agents to generate posts
        followers_count_sum = np.array([agent.user_prof["followers_count"] for agent in self.agents]).sum()
        sample_weights = np.array([agent.user_prof["followers_count"] for agent in self.agents]) / followers_count_sum
        sampled_agent_ids = np.random.choice([agent.user_prof["id"] for agent in self.agents], size=random.randint(int(len(self.agents) * self.config["sample_rate_low"]), int(len(self.agents) * self.config["sample_rate_high"])), p=sample_weights, replace=False)
        logger.info("%d sampled agents: " % len(sampled_agent_ids) + ",".join([str(sampled_agent_id) for sampled_agent_id in sampled_agent_ids]))
        for agent_id in sampled_agent_ids:
            agent = self.id2agent[agent_id]
            post = agent.generate_post(self.cur_time)
            msg = {
                "type": "post",
                "agent_id": agent.user_prof["id"],
                "origin_agent_id": agent.user_prof["id"], # the original author of this post
                "post_author": agent.user_prof["name"],
                "post_content": post,
                "time": self.cur_time,
            }
            self.logs[round_cnt]["log_num"] += 1
            self.logs[round_cnt]["log_content"].append(msg)

        self.cur_time = add_interval(self.cur_time, self.interval) # time goes by
        logger.debug(self.logs[round_cnt])

    def run(self):
        for round_cnt in range(self.round_num):
            self.run_round(round_cnt)

    def reset(self):
        pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config/config.yaml", help="config file path")

    args = parser.parse_args()
    logger.debug(args)
    return args

def main():
    args = parse_args()
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    config.freeze()
    
    logger.info(config)
    simulator = Simulator(config)
    simulator.run()
    simulator.save()


if __name__ == "__main__":
    main()
