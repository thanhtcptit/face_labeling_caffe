import numpy as np


def T6_EPsharePadding(img, parm):
    num = 16
    r, c, cc = img.shape
    patch_size = parm['patch_size'] + 2
    b_img = np.zeros([r + patch_size, c + patch_size, cc, num])
    b_img[:, :, 3, :] = 255

    xv, yv = np.meshgrid([0, -1, -2, -3], [0, -1, -2, -3])
    xv = np.reshape(xv, [-1])
    yv = np.reshape(yv, [-1])

    for k in range(num):
        board_x = int(np.ceil(patch_size / 2) + xv[k])
        board_y = int(np.ceil(patch_size / 2) + yv[k])
        b_img[board_y + 1: board_y + r + 1,
              board_x + 1: board_x + c + 1, :, k] = img

    return b_img
