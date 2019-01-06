import numpy as np

from utils.common import logistic, softmax


def T3_ImageRemap16(active_fc):
    r, c, ch, ml = active_fc.shape
    active_edge = np.zeros([r, c, ml])
    active_patch = np.zeros([ch - 1, r * c, ml])
    big_edge = np.zeros([4 * r + 1, 4 * c + 1])
    big_patch = np.zeros([4 * r + 1, 4 * c + 1, ch - 1])

    for m in range(ml):
        active_edge[:, :, m] = logistic(active_fc[:, :, -1, m])
        active = np.reshape(active_fc[:, :, :, m], [r * c, ch]).T
        active_patch[:, :, m] = softmax(active[:ch - 1, :])
        edge = active_edge[:, :, m].T
        patch = np.reshape(active_patch[:, :, m].T, [r, c, ch - 1])
        patch = np.transpose(patch, [1, 0, 2])

        iy, ix = np.unravel_index(m, [4, 4])
        x, y = np.meshgrid(list(range(ix + 1, 4 * r + 1, 4)),
                           list(range(iy + 1, 4 * c + 1, 4)))
        x = np.reshape(x, [-1])
        y = np.reshape(y, [-1])
        array_idx = np.ravel_multi_index([x, y], (4 * r + 1, 4 * c + 1))
        big_edge = np.reshape(big_edge, [-1])
        edge = np.reshape(edge, [-1])
        big_edge[array_idx] = edge
        big_edge = np.reshape(big_edge, [4 * r + 1, 4 * c + 1])

        for cc in range(ch - 1):
            big = big_patch[:, :, cc]
            big_shape = big.shape
            big = np.reshape(big, [-1])
            big[array_idx] = np.reshape(patch[:, :, cc], [-1])
            big = np.reshape(big, big_shape)
            big_patch[:, :, cc] = big

    big_edge = big_edge[:-1, :-1]
    big_patch = big_patch[:-1, :-1, :]
    return big_patch, big_edge
