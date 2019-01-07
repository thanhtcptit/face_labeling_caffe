import numpy as np

from im_toolbox import cp2tform_v1, cp2tform_v2, tformfwd


def T1_GeneratingTrom(im_shape, mean_shape):
    points_idx = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    base_points = []
    input_points = []
    im_shape_flat = np.reshape(im_shape.T, [10, 1])
    for point in points_idx:
        x, y = point
        base_points.append([mean_shape[x], mean_shape[y]])
        input_points.append([im_shape_flat[x], im_shape_flat[y]])

    base_points = np.array(base_points)
    input_points = np.array(input_points)
    """
        # MATLAB snippet
        TFORM = cp2tform(input_points, base_points, 'similarity');

        [X, Y] = tformfwd(TFORM, im_shape(1:5), im_shape(6:10));
        IM_shape_new = [X',Y']';
    """
    base_points = np.squeeze(base_points)
    input_points = np.squeeze(input_points)
    trans, _ = cp2tform_v2(input_points, base_points)
    im_shape_new = tformfwd(trans, im_shape)
    im_shape_new = np.reshape(im_shape_new.T, [10, 1])
    return im_shape_new, trans
