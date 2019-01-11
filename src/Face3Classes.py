import os
import caffe
import numpy as np

from skimage.transform import resize

from processlm_lfw import processIm_lfw
from T3_ImageRemap16 import T3_ImageRemap16
from utils.common import save
from utils.path import Path

from PIL import Image
import scipy.io as sio


def Face3Classes(img, shape, parm):
    net = parm['net']
    inp, ex = processIm_lfw(img, parm, shape)
    active_fc = []
    for m in range(16):
        b_img = inp[:, :, :, m].astype(np.float32)
        b_img = np.transpose(b_img)
        active = net.forward(data=np.asarray([b_img]))
        active = np.transpose(np.squeeze(active['conv10']))
        # for k, v in net.blobs.items():
        #     print(k)
        #     num = 1
        #     for d in v.data.shape:
        #         num *= d
        #     print(np.sum(np.abs(v.data)) / num)

        active_fc.append(active)

    active_fc = np.transpose(np.array(active_fc), [1, 2, 3, 0])

    save(os.path.join(
        Path.DEBUG_DIR, 'results/py_out2.mat'),
        {'input': inp, 'active_fc': active_fc})

    lab = {}
    big_patch, big_edge = T3_ImageRemap16(active_fc)
    big_edge = resize(big_edge, (parm['imsize'] + 2, parm['imsize'] + 2),
                      order=1, preserve_range=True)
    lab['big_edge'] = big_edge[1: -1, 1: -1]
    big_patch = resize(big_patch, (parm['imsize'] + 2, parm['imsize'] + 2),
                       order=1, preserve_range=True)
    lab['big_patch'] = big_patch[1: -1, 1: -1, :]
    return lab
