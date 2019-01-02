import os
import caffe
import numpy as np

from skimage.transform import resize

from processlm_lfw import processIm_lfw
from T3_ImageRemap16 import T3_ImageRemap16
from src.utils.common import save
from src.utils.path import Path


def Face3Classes(img, shape, parm):
    net = parm['net']
    inp, ex = processIm_lfw(img, parm, shape)
    active_fc = []
    for m in range(16):
        b_img = inp[:, :, :, m]
        b_img = np.transpose(b_img, [2, 0, 1])
        # MATLAB: active = caffe('forward_test', {single(b_img)});
        net.blobs['data'].reshape(1, *b_img.shape)
        net.blobs['data'].data[...] = b_img
        net.forward()
        active = net.blobs['conv10'].data
        # MATLAB: active_fc(:,:,:,m) = squeezesc.(active{1});
        active_fc.append(np.squeeze(active))

    active_fc = np.transpose(np.array(active_fc), [2, 3, 0, 1])
    save(os.path.join(
        Path.RESOURCES_DIR, 'out2.mat'),
        {'input': inp, 'active_fc': active_fc})

    lab = {}
    big_patch, big_edge = T3_ImageRemap16(active_fc)
    big_edge = resize(big_edge, (parm['imsize'] + 2, parm['imsize'] + 2),
                      order=1, preserve_range=True)
    lab['big_edge'] = big_edge[2: -1, 2: -1]
    big_patch = resize(big_patch, (parm['imsize'] + 2, parm['imsize'] + 2),
                       order=1, preserve_range=True)
    lab['big_patch'] = big_patch[2: -1, 2: -1, :]

    return lab
