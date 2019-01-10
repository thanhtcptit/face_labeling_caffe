import caffe
import os
import numpy as np

from T1_SolverParser import T1_SolverParser


def Face3Init_01(model_def_file, resume_file):
    if not os.path.exists(model_def_file):
        print('Need a network prototxt definition')
        exit()
    if not os.path.exists(resume_file):
        print('Need a resume file')
        exit()
    solver = T1_SolverParser(model_def_file, resume_file)
    if solver['solver_mode'] == 'GPU':
        caffe.set_mode_gpu()
        caffe.set_device(solver['device_id'])
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(solver['net'], caffe.TEST)
    layers = solver['model']

    # for k in net.blobs.keys():
    #     print(k)

    for i in range(len(layers)):
        layer_name = layers[i]['layer_names'][0][0]
        layer_weight = layers[i]['weights'][0][0][0].astype(np.float32)
        layer_bias = layers[i]['weights'][0][1][0].astype(np.float32)

        net.params[layer_name][0].data[...] = \
            np.transpose(layer_weight)
        net.params[layer_name][1].data[...] = \
            np.squeeze(layer_bias)

    return solver, net
