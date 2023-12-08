import os
import json
import random
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from yacs.config import CfgNode

from data.data import Data
from agents.snagent import SocialNetAgent
from llm.llm import LLM, ZhipuAi
# from recommender.recommender import Recommender

logging.basicConfig(format='---%(asctime)s %(levelname)s \n%(message)s ---\n', level=logging.DEBUG)


class Simulator():
    def __init__(self, config):
        self.config = config
        self.round_num = config["round_num"]
        self.round_cnt = 0
        self.log_folder = config["log_folder"]
        self.log = None
        self.data = None

        self.agents = []
        self.id2agent = {}
        self.recommender = None
        self.logs = [[] for _ in range(config["round_num"])]
        self.cur_time = 0

        self.load_data()
        self.load_agents()

    def load_data(self):
        self.data = Data(self.config)
        self.data.load_users()
        self.data.load_network()

    def load_agents(self):
        for user_prof in self.data.users:
            self.agents.append(
                SocialNetAgent(user_prof, ZhipuAi(self.config["llm_api_key"]))
            )
            self.id2agent[user_prof["id"]] = self.agents[-1]

    def create_recommender(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def run_round(self, round_cnt):
        self.logs[round_cnt] = [] # messages in this round

        messages = self.logs[round_cnt-1] if round_cnt > 0 else []
        # all agents receive messages and react to posts and reposts
        for msg in messages:
            if msg["type"] != "post" and msg["type"] != "repost":
                continue
            agent_id = msg["agent_id"]
            for neibor_agent_id in self.data.network[agent_id]:
                neibor_agent = self.id2agent[neibor_agent_id]
                neibor_agent.add_to_memory(msg, self.cur_time)
                reactions = neibor_agent.react_to_post(msg)
                for reaction in reactions:
                    if reaction == "repost":
                        new_msg = {
                            "type": "repost",
                            "agent_id": neibor_agent.user_prof["id"],
                            "origin_agent_id": msg["origin_agent_id"],
                            "content": msg["content"],
                            "time": self.cur_time
                        }
                    elif reaction == "follow":
                        new_msg = {
                            "type": "follow",
                            "agent_id": neibor_agent.user_prof["id"],
                            "target_agent_id": msg["agent_id"],
                            "time": self.cur_time
                        }
                    agent.add_to_memory(new_msg, self.cur_time)
                    self.logs[round_cnt].append(new_msg)
        # recommend system recommend posts to agents
        for agent in self.agents:
            pass

        # random sample some agents to generate posts
        sampled_agent_ids = random.sample([agent.user_prof["id"] for agent in self.agents], random.randint(int(len(self.agents) * self.config["sample_rate_low"]), int(len(self.agents) * self.config["sample_rate_high"])))
        for agent_id in sampled_agent_ids:
            agent = self.id2agent[agent_id]
            post = agent.generate_post()
            msg = {
                "type": "post",
                "agent_id": agent.user_prof["id"],
                "origin_agent_id": agent.user_prof["id"], # the original author of this post
                "content": post,
                "time": self.cur_time,
            }
            agent.add_to_memory(msg, self.cur_time)
            self.logs[round_cnt].append(msg)

        self.cur_time += 1 # time goes by
        logging.info(self.logs[round_cnt])


    def run(self):
        for round_cnt in range(self.round_num):
            self.run_round(round_cnt)

    def reset(self):
        pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config/config.yaml", help="config file path")

    args = parser.parse_args()
    logging.debug(args)
    return args

def main():
    args = parse_args()
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    config.freeze()
    
    logging.debug(config)
    simulator = Simulator(config)
    simulator.run()
    simulator.save()


if __name__ == "__main__":
    main()
