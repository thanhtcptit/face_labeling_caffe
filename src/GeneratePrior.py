import os
import glob
import numpy as np

from T1_GeneratingTrom import T1_GeneratingTrom
from LabelDistribution import LabelDistribution
from utils.common import text_read, load
from utils.path import Path


def GeneratePrior(shape):
    data_folder = Path.LFW_DATA_DIR
    folder_valid = os.path.join(data_folder, 'validate')
    val_images = glob.glob(os.path.join(folder_valid, '*.txt'))
    valid_list, valid_num_list = text_read(
        os.path.join(data_folder, 'map.txt'), '%s %s', 500)

    load_vars = ['all_coef', 'mean_shape', 'U', 'S', 'all_tform']
    all_coef, mean_shape, U, S, all_tform = load(
        os.path.join(Path.RESOURCES_DIR, 'pca_shape4lfw_valid.mat'), load_vars)

    w = [0, 1, 1, 1, 1, 1, 1]
    kn = 10

    shape_new, _ = T1_GeneratingTrom(shape, mean_shape)
    coef = np.dot(U.T, shape_new)
    tile_coef = np.tile(coef, [1, all_coef.shape[1]])
    coef_dist = np.sqrt(np.dot(w, np.square(tile_coef - all_coef)))
    # rank of most similar poses in training set
    index = np.argsort(coef_dist)
    index = [201, 355, 1, 101, 320, 161, 264, 410, 390, 252]
    lab_names = [
        'Jackie_Chan_0012.mat',
        'Nathalia_Gillot_0001.mat',
        'Aaron_Peirsol_0001.mat',
        'Dan_Morales_0001.mat',
        'Mary_Catherine_Correll_0001.mat',
        'Gian_Marco_0002.mat',
        'Juliette_Lewis_0001.mat',
        'Roberto_Carlos_0001.mat',
        'Queen_Elizabeth_II_0011.mat',
        'John_Rigas_0001.mat'
    ]
    # print(index[:10])
    # exit()
    ex = []
    for n in range(kn):
        m = index[n]
        num_img = os.path.split(val_images[m])[1]
        num_img_short_ss = num_img[:-4]
        img_name_ss = valid_list[valid_num_list.index(
            num_img_short_ss + '.jpg')]
        img_name_short_ss = img_name_ss[:-4]
        lab_name_ss = img_name_short_ss + '.mat'
        lab_name_ss = lab_names[n]
        label_ss = load(os.path.join(folder_valid, lab_name_ss))
        ex.append(label_ss['label'])
    ex = LabelDistribution(np.array(ex))
    return ex
