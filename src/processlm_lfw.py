import numpy as np

from GeneratePrior import GeneratePrior
from T6_EPsharePadding import T6_EPsharePadding


def processIm_lfw(img, parm, shape):
    ex = 255 * GeneratePrior(shape)
    inp = np.concatenate((img, ex), 2)
    inp = T6_EPsharePadding(inp, parm)

    for k in range(6):
        inp[:, :, k, :] -= parm['mean'][k]

    # % permute from RGB to BGR
    inp = inp[:, :, [2, 1, 0, 3, 4, 5], :]
    inp = np.transpose(inp, [1, 0, 2, 3])
    return inp, ex
