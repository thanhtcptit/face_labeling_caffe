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
        # for l in range(1, 7):
        #     # b_img = sio.loadmat(os.path.join(
        #     #  Path.DEBUG_DIR,
        #     #  'input/matlab/b_img{}.mat'.format(l)))['b_img'] * 255
        #     b_img = inp[:, :, l - 1, m] * 255
        #     b_img = b_img.astype('uint8')
        #     background = Image.fromarray(b_img)
        #     background.save(os.path.join(
        #         Path.DEBUG_DIR, 'input/matlab/b_img{}.png'.format(l)), 'PNG')
        # exit()
        # Code for pycaffe interface reprocedure:
        # MATLAB: active = caffe('forward_test', {single(b_img)});
        b_img = inp[:, :, :, m].astype(np.float32)
        b_img = np.transpose(b_img, [2, 0, 1])
        out = net.forward(data=np.asarray([b_img]))
        out = np.transpose(np.squeeze(out['conv10']), [1, 2, 0])
        # print(out[:, :, 0])
        # exit()
        active_fc.append(out)

    active_fc = np.transpose(np.array(active_fc), [1, 2, 3, 0])
    # print(active_fc[:, :, 0, 0])
    # exit()
    save(os.path.join(
        Path.DEBUG_DIR, 'results/out2.mat'),
        {'input': inp, 'active_fc': active_fc})
    # active_fc = sio.loadmat(os.path.join(
    #     Path.DEBUG_DIR,'results/out2.mat'))['active_fc']
    lab = {}
    big_patch, big_edge = T3_ImageRemap16(active_fc)
    big_edge = resize(big_edge, (parm['imsize'] + 2, parm['imsize'] + 2),
                      order=1, preserve_range=True)
    lab['big_edge'] = big_edge[1: -1, 1: -1]
    big_patch = resize(big_patch, (parm['imsize'] + 2, parm['imsize'] + 2),
                       order=1, preserve_range=True)
    lab['big_patch'] = big_patch[1: -1, 1: -1, :]
    # print(lab['big_patch'][:, :, 0])
    # print(lab['big_patch'].shape)
    # exit()
    return lab
