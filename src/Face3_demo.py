import os
import sys
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np

from skimage.io import imread

from Face3Init_01 import Face3Init_01
from Face3Classes import Face3Classes
from utils.common import load, save
from utils.path import Path


def main():
    solver = 'lfw03_solver_exemplar'
    proto_path = os.path.join(Path.MODEL_DIR, 'LFW_cvpr15')
    solver_file = os.path.join(proto_path, solver + '.prototxt')
    solver_mat = os.path.join(proto_path, solver + '.mat')

    solver, net = Face3Init_01(solver_file, solver_mat)

    lfw_ep_ex_mean_path = os.path.join(
        Path.RESOURCES_DIR, 'LFW_EP_EX_MEAN.mat')
    if os.path.exists(lfw_ep_ex_mean_path):
        LFW_EP_MEAN = load(lfw_ep_ex_mean_path, ['LFW_EP_MEAN'])[0][0]
    else:
        LFW_EP_MEAN = 100 * np.ones((1, 6))

    parm = {'net': net}
    parm['patch_size'] = 72
    parm['mini_batch'] = 1
    parm['imsize'] = 250
    parm['amp'] = 100
    parm['mean'] = LFW_EP_MEAN
    parm['result_path'] = os.path.join(Path.DEBUG_DIR, 'results')
    if not os.path.exists(parm['result_path']):
        os.makedirs(parm['result_path'])

    img = imread(os.path.join(Path.DEBUG_DIR, 'data', sys.argv[1]))
    img_name = os.path.splitext(sys.argv[1])[0]
    shape = []
    with open(os.path.join(Path.DEBUG_DIR, 'data', sys.argv[2])) as f:
        for line in f:
            x, y = line.split(' ')
            shape.append([int(x), int(y)])
    shape = np.array(shape)

    lab = Face3Classes(img, shape, parm)
    save(os.path.join(
        parm['result_path'], '{}_lab.mat'.format(img_name)), {'lab': lab})
    print('Result save to {}'.format(os.path.abspath(os.path.join(
        parm['result_path'], '{}_lab.mat'.format(img_name)))))


if __name__ == '__main__':
    main()
