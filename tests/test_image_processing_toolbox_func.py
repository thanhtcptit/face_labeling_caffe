import unittest
import numpy as np

from im_toolbox import cp2tform_v1, cp2tform_v2, tformfwd


class TestCustomFunc(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCustomFunc, self).__init__(*args, **kwargs)
        u = [0, 6, -2]
        v = [0, 3, 5]
        x = [-1, 0, 4]
        y = [-1, -10, 4]

        self.uv = np.array([u, v]).T.astype(np.float32)
        self.xy = np.array([x, y]).T.astype(np.float32)

    def test_cpt2form(self):
        """
        # In Matlab, run:
        #   tform_sim=cp2tform(uv,xy,'similarity');
        #
        #   trans = tform_sim.tdata.T
        #   ans =
        #       -0.0764   -1.6190         0
        #        1.6190   -0.0764         0
        #       -3.2156    0.0290    1.0000
        #   trans_inv = tform_sim.tdata.Tinv
        #    ans =
        #
        #       -0.0291    0.6163         0
        #       -0.6163   -0.0291         0
        #       -0.0756    1.9826    1.0000
        #    xy_m=tformfwd(tform_sim, u,v)
        #
        #    xy_m =
        #
        #       -3.2156    0.0290
        #        1.1833   -9.9143
        #        5.0323    2.8853
        #    uv_m=tforminv(tform_sim, x,y)
        #
        #    uv_m =
        #
        #        0.5698    1.3953
        #        6.0872    2.2733
        #       -2.6570    4.3314
    """
        res = np.array([
            [-0.0764, -1.6190, 0],
            [1.6190, -0.0764, 0],
            [-3.2156, 0.0290, 1.0000]
        ])
        res_inv = np.array([
            [-0.0291, 0.6163, 0],
            [-0.6163, -0.0291, 0],
            [-0.0756, 1.9826, 1.0000]
        ])
        trans, trans_inv = cp2tform_v2(self.uv, self.xy)
        print('-------CPT2FORM--------')
        print('Python result: ')
        print(trans)
        print('MATLAB result: ')
        print(res)
        print('---------------')
        self.assertEqual(1, 1)

    def test_tformfwd(self):
        res = np.array([
            [-3.2156, 0.0290],
            [1.1833, -9.9143],
            [5.0323, 2.8853]
        ])
        trans, trans_inv = cp2tform_v2(self.uv, self.xy)
        xy_m = tformfwd(trans, self.uv)
        print('-------TFORMFWD--------')
        print('Python result: ')
        print(xy_m)
        print('MATLAB result: ')
        print(res)
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()