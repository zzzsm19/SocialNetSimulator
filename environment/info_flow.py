from environment.recommender.recommender import Recommender
from datetime import datetime

class feed_flow:
    def __init__(self, config, data, llm):
        self.config = config
        self.data = data
        self.llm = llm
        self.recommender = Recommender(self.config, llm)
        self.network = self.data.network
        # self.rec_info = [[[] for _ in range(self.config["round_num"])] for _ in range(len(self.data.users))]
        # self.follow_info = [[[] for _ in range(self.config["round_num"])] for _ in range(len(self.data.users))]
        self.database = [[] for _ in range(len(self.data.users))]

    def get_rec_info(self, user_agent):
        rec_content = self.recommender.get_full_sort_items(user_agent, self.database, -1)
        return rec_content

    def get_follow_info(self, user_agent):
        neighbours = self.network[user_agent.user_prof["id"]]
        follow_content = []
        for neighbour in neighbours:
            follow_content.extend(self.database[int(neighbour) - 1])
        return follow_content


    def get_all_info(self, user_agent):
        self.get_rec_info(user_agent)
        self.get_follow_info(user_agent)
        return self.get_rec_info(user_agent), self.get_follow_info(user_agent)

    def init_database(self, msg = None):
        if msg == None:
            self.database[24] = [{"type": "post", "agent_id": '25', "origin_agent_id": '25', "post_author": "Stig",
                                  "post_content": "社交网络模拟，启动！", "time": datetime.now().replace(hour=8, minute=0, second=0)}]

        else:
            self.database[int(msg["agent_id"]) - 1] = [msg]
