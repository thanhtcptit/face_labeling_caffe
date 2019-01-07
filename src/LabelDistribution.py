import numpy as np


def LabelDistribution(trans_label):
    kn, r, c = trans_label.shape
    dis_label = np.zeros(shape=(r, c, 3))
    for m in range(r):
        for n in range(c):
            for i in range(3):
                dis_label[m, n, i] = \
                    float(sum(trans_label[:, m, n] == i + 1)) / kn
    return dis_label
