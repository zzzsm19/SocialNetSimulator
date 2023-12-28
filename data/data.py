"""
Import the Data class and use it to load data.
"""
import logging
import networkx as nx
import matplotlib.pyplot as plt
import collections

logger = logging.getLogger("MyLogger")

class Data:
    def __init__(self, config):
        self.user_path = config["user_path"]
        self.network_path = config["network_path"]

        self.hash = {}
        self.unhash = {}
        self.users = []
        self.network = {}

        self.word_table_path = config["word_table_path"]
        self.content_path = config["content_path"]
        self.post_time_path = config["post_time_path"]

        self.word_dict = {}
        self.content = collections.defaultdict(list)

        # self.simulated_content = []

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


    def load_word_table(self):
        with open(self.word_table_path, 'r', encoding='gbk') as f:
            lines = f.readlines()[1:]
            for line in lines:
                # print(line)
                line = line.strip('\n').split('\t')
                # print(line)
                self.word_dict[int(line[0])] = line[2]
            # print(lines)
            # print(len(lines))
        print(self.word_dict)


    def load_content(self):
        with open(self.content_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                content_id = lines[i]
                content = lines[i + 1]
                # print(content_id, content)
                self.content[int(content_id)].append(content)
        # print(len(self.content))
        # nums = 0
        del_keys = []
        for i, (key, value) in enumerate(self.content.items()):
            if len(value) != 4:
                del_keys.append(key)
        # print(nums)
        for key in del_keys:
            del self.content[key]
        for i, (key, value) in enumerate(self.content.items()):
            if i < 10:
                print(key, value)


    def load_post_time(self):
        with open(self.post_time_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
            # for i in range(1, l)
            for i in range(0, len(lines), 2):
                line = lines[i].strip('\n').split(' ')
                self.content[int(line[0])].append(datetime.strptime(line[1], '%Y-%m-%d-%H:%M:%S'))
                self.content[int(line[0])].extend(line[2:])

        # print('dd')
        # print(len(self.content))
        nums = 0
        for i, (key, value) in enumerate(self.content.items()):
            if len(value) != 3:
                nums += 1
        # print(nums)

    # def get_current_items(self):

    # def get_current_simulated_content(self, now):
    #     current_content = []
    #     for i in range(len(self.simulated_content)):
    #         if (self.simulated_content[i]["time"] - now).total_seconds() > 0:
    #             current_content.append(self.simulated_content[i])
    #     return current_content

    # def get_current_simulated_content(self):
    #     return self.simulated_content