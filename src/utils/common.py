import numpy as np
import scipy.io


def text_read(path, fformats, num_line):
    fformats = fformats.split(' ')
    format_mapping = []
    for f in fformats:
        if f == '%s':
            format_mapping.append(str)
        elif f == '%d':
            format_mapping.append(int)
        else:
            format_mapping.append(float)

    file_data = [[] for _ in range(len(format_mapping))]
    with open(path) as f:
        for i, line in enumerate(f):
            if i == num_line:
                break
            data = line.strip().split(' ')
            for i in range(len(format_mapping)):
                file_data[i].append(format_mapping[i](data[i]))
    return file_data


def load(path, vars=None):
    mat = scipy.io.loadmat(path)
    if vars is None:
        return mat
    return [mat[v] for v in vars]


def save(path, data):
    scipy.io.savemat(path, data)


def softmax(eta):
    exp_eta = np.exp(eta)
    return exp_eta / np.sum(exp_eta)


def logistic(z):
    return 1 / (1 + np.exp(-z))


def str2type(value, ttype):
    try:
        return ttype(value)
    except ValueError:
        return None


if __name__ == '__main__':
    t = load('../resources/data/LFW/validate/Bruce_Springsteen_0002.mat')
    print(t)
    print(t['label'].shape)
