"""
preprocess data
transform followee network to follower network (the original data file is 'weibo_network/wiebo_network.txt').
followee network: user_id, followee_num, followee1, followee2, ... (each line),
follower network: user_id, follower_num, follower1, follower2, ... (each line).
"""

from tqdm import tqdm
import multiprocessing
PART_SIZE = 200000

def get_network_lines():
    with open('../dataset/weibo_network/weibo_network.txt', 'r') as f:
        line = f.readline()
        N = int(line.split()[0])
        M = int(line.split()[1])
        lines = f.readlines()
    return N, M, lines


def count_network_lines_follower(lines):
    """
    count the number of followers for users from N lines part by part (each part contains PART_SIZE lines).
    If we count all the followers for all users at once, it will take too much time as the size of data is too large.
    """
    N = len(lines)
    for t in range(0, N, PART_SIZE):
        follower_lines = ['' for _ in range(N)]
        for i in tqdm(range(t, min(t + PART_SIZE, N))):
            line = lines[i].strip().split('\t')
            followee_num = int(line[1])
            for j in range(2, 2 * followee_num + 2, 2):
                followee = int(line[j])
                follower_lines[followee] += str(i) + '\t' + line[j + 1] + '\t'
        with open('../dataset/weibo_network/weibo_network_follower' + str(t) + '.txt', 'w') as f:
            for i in range(N):
                f.write(follower_lines[i].strip('\t') + '\n')


def combine_network_lines_follower(N, M):
    """
    combine the follower network lines from N // PART_SIZE parts to one file.
    """
    follower_lines = ['' for _ in range(N)]
    for t in range(0, N, PART_SIZE):
        with open('../dataset/weibo_network/weibo_network_follower' + str(t) + '.txt', 'r') as f:
            lines = f.readlines()
            assert len(lines) == N
            for i in tqdm(range(N)):
                if lines[i] == '\n':
                    continue
                follower_lines[i] += lines[i].strip('\n').strip('\t') + '\t'

    M_check = 0
    for i in tqdm(range(N)):
        M_check += len(follower_lines[i].split('\t')) // 2
        follower_lines[i] = str(i) + '\t' + str(len(follower_lines[i].split('\t')) // 2) + '\t' + follower_lines[i]
    assert M_check == M

    with open('../dataset/weibo_network/weibo_network_follower.txt', 'w') as f:
        f.write(str(N) + '\t' + str(M) + '\n')
        for line in follower_lines:
            f.write(line.strip('\t') + '\n')


def main():
    N, M, lines = get_network_lines()
    count_network_lines_follower(lines)
    combine_network_lines_follower(N, M)

if __name__ == '__main__':
    main()