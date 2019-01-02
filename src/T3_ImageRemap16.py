import numpy as np
from src.utils.common import logistic, softmax


def T3_ImageRemap16(active_fc):
    r, c, ch, ml = active_fc.shape
    active_edge = np.zeros([r, c, ml])
    active_patch = np.zeros([ch - 1, r * c, ml])
    bid_edge = np.zeros([4 * r + 1, 4 * c + 1])
    big_patch = np.zeros([4 * r + 1, 4 * c + 1, ch - 1])

    for m in range(ml):
        active_edge[:, :, m] = logistic(active_fc[:, :, -1, m])
        active = np.reshape(active_fc[:, :, :, m], [r * c, ch])

        active_patch[:, :, m] = softmax(active[:ch, :])
        edge = active_edge[:, :, m]
        patch = np.reshape(active_patch[:, :, m].T, [r, c, ch - 1])
        patch = np.transpose(patch, [1, 0, 2])

        iy, ix = np.unravel_index(m, [4, 4])
        x, y = np.meshgrid(list(range(ix + 1, 4 * r + 2, 4)),
                           list(range(iy + 1, 4 * c + 2, 4)))
        array_idx = np.ravel_multi_index(
            np.array([x, y]), [4 * r + 1, 4 * c + 1])
        big_edge[array_idx] = edge

        for cc in range(ch - 1):
            big = big_patch[:, :, cc]
            big[array_idx] = patch[:, :, cc]
            big_patch[:, :, cc] = big

    big_edge = big_edge[:-1, :-1]
    big_patch = big_patch[:-1, :-1, :]
    return big_patch, bid_edge
