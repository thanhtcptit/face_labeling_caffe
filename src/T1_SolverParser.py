import os
import numpy as np

from utils.common import load, str2type


def T1_SolverParser(solver_def_file, resume_file):
    if not os.path.exists(solver_def_file):
        print('Need a solver definition file')
        exit()
    if os.path.exists(resume_file):
        solver = load(resume_file)
        solver = {'model': solver['Solver'][0, 0]['model']}
    else:
        solver = {}

    # Parse solver file to dict
    with open(solver_def_file, 'r') as f:
        for line in f:
            line = line.strip()
            ind = line.find('"')
            if ind != -1:
                field = line[ind + 1: -1]
                ind2 = line.find(':')
                name = line[: ind2]
            else:
                ind2 = line.find(':')
                if ind2 == -1:
                    print('Incorrect format')
                ctr = line[ind2 + 2:]
                _ctr = str2type(ctr, float)
                if _ctr is None:
                    field = ctr
                else:
                    field = _ctr
                name = line[:ind2]
            solver[name] = field

    if 'solver_mode' not in solver:
        solver['solver_mode'] = 'GPU'

    lnum = len(solver['model'])
    for ind in range(lnum):
        if solver['model'][ind]['layer_names'][0][0] == 'fc' + str(ind + 1):
            solver['model'][ind]['layer_names'][0] = \
                solver['model'][ind]['layer_names'][0].astype('<U6')
            solver['model'][ind]['layer_names'][0][0] = 'conv' + str(ind + 1)
            weights = solver['model'][ind]['weights'][0][0][0]
            s1, s2 = weights.shape
            _, _, _, ch = solver['model'][ind - 1]['weights'][0][0][0].shape
            filter_size = int(np.sqrt(s1 / ch))
            weights = np.reshape(weights, [filter_size, filter_size, ch, s2])
            solver['model'][ind]['weights'][0][0][0] = weights

    if solver['solver_mode'] == 'GPU' and 'device_id' not in solver:
        solver['device_id'] = 0

    return solver
