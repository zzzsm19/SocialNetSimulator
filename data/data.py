"""
Import the Data class and use it to load data.
"""
import logging

Logger = logging.getLogger("MyLogger")

class Data:
    def __init__(self, config):
        self.user_path = config["user_path"]
        self.network_path = config["network_path"]

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
                "id": int(lines[i * 15].strip()),
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
            # Logger.debug(user)
        Logger.info("load {} users".format(len(self.users)))
    
    def load_network(self):
        """
        load user follower network from follower network path
        """
        with open(self.network_path, 'r') as f:
            [node_num, edge_num] = [int(_) for _ in f.readline().strip().split()]
            uids = [int(_) for _ in f.readline().strip().split()]
            lines = [line.strip().split() for line in f.readlines()]
        for i in range(len(uids)):
            self.network[uids[i]] = [uids[j] for j in range(len(lines[i])) if lines[i][j] == '1']
        Logger.info("network contains {} nodes and {} edges".format(node_num, edge_num))
