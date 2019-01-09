import unittest
import os
import numpy as np

from utils.path import Path
from im_toolbox import cp2tform_v1, cp2tform_v2, tformfwd


class TestTransformFunc(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTransformFunc, self).__init__(*args, **kwargs)
        self.num_test = 3

    def read(self, path):
        array = []
        with open(path) as f:
            for line in f:
                array.append(map(float, line.strip().split(',')))

        return np.array(array)

    def load_data(self, ind):
        input_points = self.read(os.path.join(
            Path.TEST_DIR, 'data/inputs/input_points_{}.txt'.format(ind)))
        base_points = self.read(os.path.join(
            Path.TEST_DIR, 'data/inputs/base_points_{}.txt'.format(ind)))
        trans = self.read(os.path.join(
            Path.TEST_DIR, 'data/results/trans_matrix_{}.txt'.format(ind)))
        trans_result = self.read(os.path.join(
            Path.TEST_DIR, 'data/results/trans_result_{}.txt'.format(ind)))

        return input_points, base_points, trans, trans_result

    def test_cpt2form(self):
        for i in range(self.num_test):
            uv, xy, trans, _ = self.load_data(i + 1)
            ACCEPT_DIFF = 1
            _trans, _ = cp2tform_v2(uv, xy)
            diff = np.sum(np.abs(trans - _trans)) / 6
            self.assertLess(diff, ACCEPT_DIFF)

    def test_tformfwd(self):
        for i in range(self.num_test):
            uv, xy, _, trans_result = self.load_data(i + 1)
            ACCEPT_DIFF = 1
            _trans, _ = cp2tform_v2(uv, xy)
            xy_m = tformfwd(_trans, uv)
            diff = np.sum(np.abs(trans_result - xy_m)) / (2 * len(uv))
            self.assertLess(diff, ACCEPT_DIFF)


if __name__ == '__main__':
    unittest.main()
