import numpy as np

tal = 0


def new_P(x, re, flag):
    global tal
    for i in range(re.shape[0]):
        if re[x][i] != 0 and flag[i] == 0:
            flag[i] = 1
            tal = tal + 1
            new_P(i, re, flag);


if __name__ == '__main__':
    num = 7
    a = 5
    m = 6
    relation = np.zeros((7, 7))
    relation[1, 0] = 1
    relation[3, 1] = 1
    relation[4, 1] = 1
    relation[5, 3] = 1
    relation[6, 1] = 1
    relation[6, 3] = 1

    flag = np.zeros(7)
    flag[5] = 1
    new_P(a, relation, flag)
    print tal
