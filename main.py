import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from typing import List, Dict
from datetime import datetime
from yacs.config import CfgNode

from data.data import Data
from agents.agent import SocialNetAgent
from llm.llm import *
from utils.interval import *
# from recommender.recommender import Recommender
from environment.info_flow import feed_flow

logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)
# stdout handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter("---%(asctime)s %(levelname)s \n%(message)s ---\n\n"))
logger.addHandler(stdout_handler)


class Simulator():
    def __init__(self, config: dict):
        self.config: dict = config
        self.round_num: int = config["round_num"]
        self.round_cnt: int = 0
        self.log_folder: str = config["log_folder"]
        self.agents_folder: str = os.path.join(self.log_folder, config["agents_folder"])
        # create log folder and agents folder if not exists
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)
        if not os.path.exists(self.agents_folder):
            os.mkdir(self.agents_folder)
        self.cur_time: datetime = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval: Interval = parse_interval(config["interval"])
        self.data: Data = Data(config)
        self.agents: List[SocialNetAgent] = []
        self.id2agent: Dict[str, SocialNetAgent] = dict()
        self.logs: List[dict] = [{
            "round_cnt": i,
            "log_num": 0,
            "log_content": []
        } for i in range(config["round_num"])]
        self.id2agent_log: Dict[str, dict] = dict()
        self.agent_logs: List[dict] = []
        # self.recommender = None
        self.environment = None


        # init
        self.load_data()
        self.load_agents()
        self.create_environment()
        # self.create_recommender()
        
        # load checkpoint
        if config["load_checkpoint"]:
            print("load checkpoint from saved_simulator.json")
            self.load_from_dict()

    def load_data(self):
        self.data.load_users()
        self.data.load_network()
        # self.data.plot_network()
        # self.data.load_post_time()
        # self.data.load_content()

    def load_agents(self):
        if self.config["llm"] == "zhipuai":
            llm = ZhipuAi(self.config["zhipuai_api_key"])
        elif self.config["llm"] == "llama2":
            llm = Llama2(self.config["llama2_model_path"])
        elif self.config["llm"] == "chatglm3":
            llm = ChatGlm3(self.config["chatglm3_model_path"])
        elif self.config["llm"] == "openai":
            llm = OpenAi()
        for idx, user_prof in enumerate(self.data.users):
            self.agents.append(
                SocialNetAgent(user_prof, llm, self.config["reflection_thershold"])
            )
            self.id2agent[user_prof["id"]] = self.agents[-1]
        # init agent logs
        self.agent_logs = [{
            "origin_id": agent.user_prof["origin_id"],
            "agent_id": agent.user_prof["id"],
            "name": agent.user_prof["name"],
            "followers": ' '.join(self.data.network[agent.user_prof["id"]]),
            "log": [],
            "memory": {}
        } for agent in self.agents]
        self.id2agent_log = {agent_log["agent_id"]: agent_log for agent_log in self.agent_logs}

    # def create_recommender(self):
    #     if self.config["llm"] == "zhipuai":
    #         llm = ZhipuAi(self.config["zhipuai_api_key"])
    #     elif self.config["llm"] == "llama2":
    #         llm = Llama2(self.config["llama2_model_path"])
    #     elif self.config["llm"] == "chatglm3":
    #         llm = ChatGlm3(self.config["chatglm3_model_path"])
    #     elif self.config["llm"] == "openai":
    #         llm = OpenAi()
    #     self.recommender = Recommender(self.config, self.data, llm)


    def create_environment(self):
        if self.config["llm"] == "zhipuai":
            llm = ZhipuAi(self.config["zhipuai_api_key"])
        elif self.config["llm"] == "llama2":
            llm = Llama2(self.config["llama2_model_path"])
        elif self.config["llm"] == "chatglm3":
            llm = ChatGlm3(self.config["chatglm3_model_path"])
        elif self.config["llm"] == "openai":
            llm = OpenAi()
        self.environment = feed_flow(self.config, self.data, llm)
        self.environment.init_database()


    # def run_round(self, round_cnt):
    #     logger.info("Round %d: %s", round_cnt, format_time(self.cur_time))
    #
    #     # messages from last round
    #     msgs = self.logs[round_cnt - 1]["log_content"] if round_cnt > 0 else [{"type": "post", "agent_id": '25', "origin_agent_id": '25', "post_author": "Stig", "post_content": "社交网络模拟，启动！", "time": self.cur_time}]
    #     # all agents receive messages and react to posts and reposts
    #     print(self.data.network)
    #     for msg in msgs[:2]:
    #         if msg["type"] != "post" and msg["type"] != "repost":
    #             continue
    #         agent_id = msg["agent_id"]
    #         for neibor_agent_id in self.data.network[agent_id][:2]:
    #             logger.info("agent %s recieve msg from agent %s"%(neibor_agent_id, agent_id))
    #             neibor_agent = self.id2agent[neibor_agent_id]
    #             react = neibor_agent.react_to_post(msg, self.cur_time)
    #             # agent_logs
    #             self.id2agent_log[neibor_agent_id]["log"].append("recieve a post from agent %s: %s" % (agent_id, msg["post_content"]))
    #             if react["repost"]:
    #                 new_msg = {
    #                     "type": "repost",
    #                     "agent_id": neibor_agent.user_prof["id"],
    #                     "origin_agent_id": msg["origin_agent_id"],
    #                     "post_author": msg["post_author"],
    #                     "post_content": msg["post_content"],
    #                     "time": self.cur_time.strftime("%Y-%m-%d %H:%M:%S"),
    #                 }
    #                 self.logs[round_cnt]["log_num"] += 1
    #                 self.logs[round_cnt]["log_content"].append(new_msg)
    #                 # agent_logs
    #                 self.id2agent_log[neibor_agent_id]["log"].append("repost the post.")
    #             if react["follow"]:
    #                 pass # must have followed before
    #             # save agent log
    #             self.id2agent_log[neibor_agent_id]["memory"] = neibor_agent.memory.save_to_dict(with_embedding=False)
    #             with open(os.path.join(self.agents_folder, neibor_agent_id + ".json"), "w") as f:
    #                 f.write(json.dumps(self.id2agent_log[neibor_agent_id], indent=4, ensure_ascii=False).encode("utf-8").decode("utf-8"))
    #
    #         # recommend system recommend posts to agents
    #     for agent in self.agents[:2]:
    #         # print('why')
    #         sorted_content = self.recommender.get_full_sort_items(agent, -1)
    #         for msg in sorted_content:
    #             logger.info("agent %s receives msg from agent %s in recommender. The msg: %s" % (agent.user_prof["id"], msg["agent_id"], msg["post_content"]))
    #             react = agent.react_to_post(msg, self.cur_time)
    #             if react["repost"]:
    #                 new_msg = {
    #                     "type": "repost",
    #                     "agent_id": agent.user_prof["id"],
    #                     "origin_agent_id": msg["origin_agent_id"],
    #                     "post_author": msg["post_author"],
    #                     "post_content": msg["post_content"],
    #                     "time": self.cur_time
    #                 }
    #                 self.logs[round_cnt]["log_num"] += 1
    #                 self.logs[round_cnt]["log_content"].append(new_msg)
    #             if react["follow"]:
    #                 pass
    #
    # # random sample some agents to generate posts
    #     followers_count_sum = np.array([agent.user_prof["followers_count"] for agent in self.agents]).sum()
    #     sample_weights = np.array([agent.user_prof["followers_count"] for agent in self.agents]) / followers_count_sum
    #     sampled_agent_ids = np.random.choice([agent.user_prof["id"] for agent in self.agents], size=random.randint(int(len(self.agents) * self.config["sample_rate_low"]), int(len(self.agents) * self.config["sample_rate_high"])), p=sample_weights, replace=False)
    #     logger.info("%d sampled agents: " % len(sampled_agent_ids) + ",".join(sampled_agent_ids))
    #     for agent_id in sampled_agent_ids[:2]:
    #         agent = self.id2agent[agent_id]
    #         post = agent.generate_post(self.cur_time)
    #         msg = {
    #             "type": "post",
    #             "agent_id": agent.user_prof["id"],
    #             "origin_agent_id": agent.user_prof["id"], # the original author of this post
    #             "post_author": agent.user_prof["name"],
    #             "post_content": post,
    #             "time": self.cur_time.strftime("%Y-%m-%d %H:%M:%S"),
    #         }
    #         self.logs[round_cnt]["log_num"] += 1
    #         self.logs[round_cnt]["log_content"].append(msg)
    #         # agent_logs
    #         self.id2agent_log[agent_id]["log"].append("generate a post:" + post)
    #         self.id2agent_log[agent_id]["memory"] = agent.memory.save_to_dict(with_embedding=False)
    #         # save agent log
    #         with open(os.path.join(self.agents_folder, agent_id + ".json"), "w") as f:
    #             f.write(json.dumps(self.id2agent_log[agent_id], indent=4, ensure_ascii=False).encode("utf-8").decode("utf-8"))
    #
    #     for i in range(len(self.logs[round_cnt]["log_content"])):
    #         print('xixi')
    #         print(self.data.simulated_content)
    #         print(self.logs[round_cnt]["log_content"][i])
    #         if self.logs[round_cnt]["log_content"][i]["type"] == "post":
    #             self.data.simulated_content.append(self.logs[round_cnt]["log_content"][i])
    #
    #     self.cur_time = add_interval(self.cur_time, self.interval) # time goes by
    #     logger.debug(self.logs[round_cnt])


    def run_round(self, round_cnt):
        logger.info("Round %d: %s", round_cnt, format_time(self.cur_time))


        for agent in self.agents[:3]:
            rec_content, follow_content = self.environment.get_all_info(agent)
            rec_content_len = len(rec_content)
            msgs = rec_content + follow_content
            logger.info("all infos:")
            logger.info(msgs)
            for i,msg in enumerate(msgs):
                if i == 0:
                    logger.info("agent %s enters the recommender." % (agent.user_prof["id"]))
                    logger.info(rec_content)
                if i == rec_content_len:
                    logger.info("agent %s enters the following list." % (agent.user_prof["id"]))
                    logger.info(follow_content)
                logger.info("agent %s recieve msg from agent %s" % (agent.user_prof["id"], msg["agent_id"]))
                react = agent.react_to_post(msg, self.cur_time)
                self.id2agent_log[agent.user_prof["id"]]["log"].append("recieve a post from agent %s: %s" % (msg["agent_id"], msg["post_content"]))
                if react["repost"]:
                    new_msg = {
                        "type": "repost",
                        "agent_id": agent.user_prof["id"],
                        "origin_agent_id": msg["origin_agent_id"],
                        "post_author": msg["post_author"],
                        "post_content": msg["post_content"],
                        "time": self.cur_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self.logs[round_cnt]["log_num"] += 1
                    self.logs[round_cnt]["log_content"].append(new_msg)
                    # agent_logs
                    self.id2agent_log[agent.user_prof["id"]]["log"].append("repost the post.")
                if react["follow"]:
                    pass # must have followed before
                # save agent log
                self.id2agent_log[agent.user_prof["id"]]["memory"] = agent.memory.save_to_dict(with_embedding=False)
                with open(os.path.join(self.agents_folder, agent.user_prof["id"] + ".json"), "w") as f:
                    f.write(json.dumps(self.id2agent_log[agent.user_prof["id"]], indent=4, ensure_ascii=False).encode("utf-8").decode("utf-8"))

    # random sample some agents to generate posts
        followers_count_sum = np.array([agent.user_prof["followers_count"] for agent in self.agents]).sum()
        sample_weights = np.array([agent.user_prof["followers_count"] for agent in self.agents]) / followers_count_sum
        sampled_agent_ids = np.random.choice([agent.user_prof["id"] for agent in self.agents], size=random.randint(int(len(self.agents) * self.config["sample_rate_low"]), int(len(self.agents) * self.config["sample_rate_high"])), p=sample_weights, replace=False)
        logger.info("%d sampled agents: " % len(sampled_agent_ids) + ",".join(sampled_agent_ids))
        for agent_id in sampled_agent_ids[:2]:
            agent = self.id2agent[agent_id]
            post = agent.generate_post(self.cur_time)
            msg = {
                "type": "post",
                "agent_id": agent.user_prof["id"],
                "origin_agent_id": agent.user_prof["id"], # the original author of this post
                "post_author": agent.user_prof["name"],
                "post_content": post,
                "time": self.cur_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.logs[round_cnt]["log_num"] += 1
            self.logs[round_cnt]["log_content"].append(msg)
            # agent_logs
            self.id2agent_log[agent_id]["log"].append("generate a post:" + post)
            self.id2agent_log[agent_id]["memory"] = agent.memory.save_to_dict(with_embedding=False)
            # save agent log
            with open(os.path.join(self.agents_folder, agent_id + ".json"), "w") as f:
                f.write(json.dumps(self.id2agent_log[agent_id], indent=4, ensure_ascii=False).encode("utf-8").decode("utf-8"))

        for i in range(len(self.logs[round_cnt]["log_content"])):
            print('xixi')
            print(self.logs[round_cnt]["log_content"][i])
            if self.logs[round_cnt]["log_content"][i]["type"] == "post" or self.logs[round_cnt]["log_content"][i]["type"] == "repost":
                # self.data.simulated_content.append(self.logs[round_cnt]["log_content"][i])
                self.environment.database[int(self.logs[round_cnt]["log_content"][i]["agent_id"]) - 1].append(self.logs[round_cnt]["log_content"][i])

        self.cur_time = add_interval(self.cur_time, self.interval) # time goes by
        logger.debug(self.logs[round_cnt])



    def run(self):
        # reset log file
        with open(os.path.join(self.log_folder, "debug.log"), "r") as f:
            lines = f.readlines()
            lines = ''.join(lines)
            lines = ''.join(lines.split("Round " + str(self.round_cnt))[0].split("---")[:-1])
        with open(os.path.join(self.log_folder, "debug.log"), "w") as f:
            f.write(lines)
        with open(os.path.join(self.log_folder, "info.log"), "r") as f:
            lines = f.readlines()
            lines = ''.join(lines)
            lines = ''.join(lines.split("Round " + str(self.round_cnt))[0].split("---")[:-1])
        with open(os.path.join(self.log_folder, "info.log"), "w") as f:
            f.write(lines)
        # run
        while self.round_cnt < self.round_num:
            self.run_round(self.round_cnt)
            self.round_cnt += 1
            self.save()

    def reset(self):
        self.round_cnt = 0
        self.cur_time = datetime.now().replace(hour=8, minute=0, second=0)
        self.logs = [{
            "log_num": 0,
            "log_content": []
        } for _ in range(self.round_num)]
        for agent in self.agents:
            agent.reset()

    def save_to_dict(self):
        return {
            "round_cnt": self.round_cnt,
            "cur_time": self.cur_time.strftime("%Y-%m-%d %H:%M:%S"), 
            "logs": self.logs,
            "agents": [agent.save_to_dict() for agent in self.agents],
        }
    
    def save(self):
        with open(os.path.join(self.log_folder, "saved_simulator.json"), "w") as f:
            f.write(json.dumps(self.save_to_dict(), indent=4, ensure_ascii=False).encode("utf-8").decode("utf-8"))
        
    def load_from_dict(self):
        with open(os.path.join(self.log_folder, "saved_simulator.json"), "r") as f:
            dic = json.loads(f.read())
        self.round_cnt = dic["round_cnt"]
        self.cur_time = datetime.strptime(dic["cur_time"], "%Y-%m-%d %H:%M:%S")
        for log in dic["logs"]:
            self.logs[log["round_cnt"]] = log
        for agent_dic in dic["agents"]:
            agent = self.id2agent[agent_dic["user_prof"]["id"]]
            agent.load_from_dict(agent_dic)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config/config.yaml", help="config file path")
    parser.add_argument("-lc", "--load_checkpoint", action="store_true", help="load checkpoint with saved_simulator.json, must use same dataset")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    config["load_checkpoint"] = args.load_checkpoint
    config.freeze()
    
    # create log folder
    if not os.path.exists(config["log_folder"]):
        os.mkdir(config["log_folder"])
    # debug logfile handler
    logfile_handler = logging.FileHandler(os.path.join(config["log_folder"], "debug.log"))
    logfile_handler.setLevel(logging.DEBUG)
    logfile_handler.setFormatter(logging.Formatter("---%(asctime)s %(levelname)s \n%(message)s ---\n\n"))
    logger.addHandler(logfile_handler)
    # info logfile handler
    logfile_handler = logging.FileHandler(os.path.join(config["log_folder"], "info.log"))
    logfile_handler.setLevel(logging.INFO)
    logfile_handler.setFormatter(logging.Formatter("---%(asctime)s %(levelname)s \n%(message)s ---\n\n"))
    logger.addHandler(logfile_handler)

    print(config)
    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    a = [np.array([1,1,1]), np.array([2,2,2])]
    print(np.stack(a, 0))
    main()
