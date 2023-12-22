"""
Import the Data class and use it to load data.
"""
import logging
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger("MyLogger")

class Data:
    def __init__(self, config):
        self.user_path = config["user_path"]
        self.network_path = config["network_path"]

        self.hash = {}
        self.unhash = {}
        self.users = []
        self.network = {}

    def load_users(self):
        """
        load profile of users from user_path.
        profile file format:
            id, bi_followers_count, city, verified, followers_count, location, province, friends_count,
            name, gender, created_at, verified_type, statuses_count, description, [blank line] 
        (15 lines in total for each user).
        """
        with open(self.user_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
            assert len(lines) % 15 == 0
        for i in range(1, len(lines) // 15):
            user = {
                "origin_id": lines[i * 15].strip(),
                "hashed_id": str(i), # hashed id, start from 1
                "id": str(i), # id, equal to hashed id
                "bi_followers_count": int(lines[i * 15 + 1].strip()),
                "verified": lines[i * 15 + 3].strip(),
                "followers_count": int(lines[i * 15 + 4].strip()),
                "location": lines[i * 15 + 5].strip(),
                "name": lines[i * 15 + 8].strip(),
                "gender": "女" if lines[i * 15 + 9].strip() == "f" else "男" if lines[i * 15 + 9].strip() == "m" else "未知",
                "verified_type": lines[i * 15 + 11].strip(),
                "description": lines[i * 15 + 13].strip(),
            }
            self.users.append(user)
            # hash and unhash
            self.hash[user["origin_id"]] = user["hashed_id"]
            self.unhash[user["hashed_id"]] = user["origin_id"]
        print("load {} users".format(len(self.users)))
    
    def load_network(self):
        """
        load user follower network from follower network path, followers of each user.
        """
        with open(self.network_path, 'r') as f:
            [node_num, edge_num] = [int(_) for _ in f.readline().strip().split()]
            uids = f.readline().strip().split()
            lines = [line.strip().split() for line in f.readlines()]
        for i in range(len(uids)):
            self.network[self.hash[uids[i]]] = [self.hash[uids[j]] for j in range(len(lines[i])) if lines[i][j] == '1']
        print("network contains {} nodes and {} edges".format(node_num, edge_num))

    def plot_network(self):
        graph=nx.DiGraph()
        for user in self.users:
            graph.add_node(user["id"])
        for user in self.users:
            for follower_id in self.network[user["id"]]:
                graph.add_edge(follower_id, user["id"])
        
        plt.figure(figsize=(12, 12))
        pos=nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True)
        plt.savefig("network", dpi=1000, bbox_inches = 'tight')
