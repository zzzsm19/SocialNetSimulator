"""
Preprocess data
Extract a sub network from the whole dataset.
Extracted data are stored in 'extracted' folder,
extracted data include: network.txt(sub network), user_list.txt(list of users in sub network), user_profile.txt(users' profile).
"""

from tqdm import tqdm
import random
import numpy as np

def get_followee_lines():
    with open('../dataset/weibo_network/weibo_network.txt', 'r') as f:
        line = f.readline()
        N = int(line.split()[0])
        M = int(line.split()[1])
        print("followee network: {} nodes, {} edges".format(N, M))
        followee_lines = f.readlines()
    return followee_lines

def get_follower_lines():
    with open('../dataset/weibo_network/weibo_network_follower.txt', 'r') as f:
        line = f.readline()
        N = int(line.split()[0])
        M = int(line.split()[1])
        print("followee network: {} nodes, {} edges".format(N, M))
        follower_lines = f.readlines()
    return follower_lines


# follower_count = []
# followee_count = []
# follow_count = []
# for i in tqdm(range(N)):
#     followee_count.append((i, int(followee_lines[i].split('\t')[1])))
#     follower_count.append((i, int(follower_lines[i].split('\t')[1])))
#     follow_count.append((i, followee_count[-1][1] + follower_count[-1][1]))
# follow_count_sorted = [_ for _ in follow_count]
# follow_count_sorted.sort(key=lambda x: x[1])
# followee_count_sorted = [_ for _ in followee_count]
# follower_count_sorted = [_ for _ in follower_count]
# followee_count_sorted.sort(key=lambda x: x[1])
# follower_count_sorted.sort(key=lambda x: x[1])


def extract_sub_network(seeds, L, followee_lines, follower_lines):
    toexpand = set(seeds)
    expanding = set()
    expanded = set()
    print("seeds: ", seeds)
    for _ in range(L + 1):
        expanding = toexpand
        toexpand = set()
        # followee
        for i in tqdm(expanding):
            line = followee_lines[i].strip().split('\t')
            followee_num = int(line[1])
            for j in range(2, 2 * followee_num + 2, 2):
                followee = int(line[j])
                if followee not in expanding and followee not in expanded:
                    toexpand.add(followee)
        # # follower
        # for i in tqdm(expanding):
        #     line = follower_lines[i].strip().split('\t')
        #     follower_num = int(line[1])
        #     for j in range(2, 2 * follower_num + 2, 2):
        #         follower = int(line[j])
        #         if follower not in expanding and follower not in expanded:
        #             toexpand.add(follower)
        expanded = expanded.union(expanding)
    print("extracted length: ", len(expanded))
    with open('../dataset/extracted/user_list.txt', 'w') as f:
        f.write(str(len(expanded)) + '\n')
        f.write(' '.join([str(_) for _ in expanded]) + '\n')

    return expanded


def get_users():
    with open('../dataset/extracted/user_list.txt', 'r') as f:
        user_list = [int(_) for _ in f.readlines()[1].strip().split()]
    return user_list

def get_uids():
    with open('../dataset/uidlist.txt', 'r') as f:
        uid_list = [int(_.strip()) for _ in f.readlines()]
    return uid_list


def count_network(followee_lines):
    """
    Get the sub network of users in user_list.
    The format of network.txt is n * n zero-one matrix, where n is the number of users in user_list.
    """
    # sub network
    user_list = get_users()
    uid_list = get_uids()
    user_set = set(user_list)
    N = len(user_list)
    M = 0
    follow_map = np.array([0] * N * N).reshape(N, N)
    for i in tqdm(range(N)):
        followee_line = followee_lines[user_list[i]].strip().split('\t')
        followee_num = int(followee_line[1])
        # follower_line = follower_lines[nodes[i]].strip().split('\t')
        # follower_num = int(follower_line[1])
        for j in range(2, 2 * followee_num + 2, 2):
            followee = int(followee_line[j])
            if followee in user_set:
                follow_map[i][user_list.index(followee)] = 1
                M += 1
    with open('../dataset/extracted/network.txt', 'w') as f:
        f.write('{} {}\n'.format(N, M))
        f.write(' '.join([str(uid_list[_]) for _ in user_list]) + '\n')
        for i in tqdm(range(N)):
            f.write(' '.join([str(_) for _ in follow_map[i]]) + '\n')


def count_user_profile():
    # sub user profile
    """
        id, bi_followers_count, city, verified, followers_count, location, province, friends_count,
        name, gender, created_at, verified_type, statuses_count, description
    """
    user_list = get_users()
    uid_list = get_uids()
    user_set_unhashed = set([uid_list[i] for i in user_list])

    with open('../dataset/userProfile/user_profile1.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
        assert len(lines) % 15 == 0
    with open('../dataset/userProfile/user_profile2.txt', 'r', encoding='gbk') as f:
        lines += f.readlines()[15:]
        assert len(lines) % 15 == 0
    N = len(lines) // 15
    with open('../dataset/extracted/user_profile.txt', 'w', encoding='gbk') as f:
        for i in range(15):
            f.write(lines[i])
        for i in tqdm(range(1, N)):
            user_id = int(lines[15 * i].strip())
            if user_id in user_set_unhashed:
                for j in range(15):
                    f.write(lines[15 * i + j])


def main():
    followee_lines = get_followee_lines()
    follower_lines = get_follower_lines()
    L = 1 # hop num
    K = 3 # seed num
    # seeds = set(random.sample(range(N), K))
    seeds = {1423311} # one seed for test
    extract_sub_network(seeds, L, followee_lines, follower_lines)
    count_network(followee_lines)
    count_user_profile()


if __name__ == '__main__':
    main()
