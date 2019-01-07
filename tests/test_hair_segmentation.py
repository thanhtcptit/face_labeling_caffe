import os
import pandas as pd
import numpy as np
import scipy.io as sio
import skimage.io as skio
import matplotlib.pylab as plt
from PIL import Image

from utils.path import Path


def viz_landmarks(input_img, shapes, columns, rows=1, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(input_img)
    plt.scatter(x=shapes[:, 0], y=shapes[:, 1], c='w', s=20)
    plt.show()


def viz_labelled_face(input_img, landmark, patch_segment,
                      columns=2, rows=1, figsize=(12, 12)):
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(input_img)
        if i == 1:
            plt.scatter(x=landmark[:, 0], y=landmark[:, 1], c='w', s=20)
        else:
            hair_mask = patch_segment[:, :, 1]
            plt.imshow(hair_mask, alpha=0.8)
    plt.show()


def save_labelled_face(input_img, patch_segment):
    hair_patch = patch_segment[:, :, 1]
    print('read_label ', type(hair_patch),
          hair_patch.dtype, hair_patch.shape)
    hair_patch = hair_patch * 255
    hair_patch.astype('uint8')

    background = Image.fromarray(input_img)
    overlay = Image.fromarray(hair_patch)
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(background, overlay, .9)
    new_img = new_img.convert('RGB')
    new_img.save(os.path.join(Path.DEBUG_DIR, 'data/new.png'), 'PNG')


if __name__ == '__main__':
    input_img = skio.imread(os.path.join(Path.DEBUG_DIR, 'data/img2.png'))
    landmark = pd.read_csv(os.path.join(Path.DEBUG_DIR, 'data/img2_lm.txt'),
                           header=None, sep=" ").values
    # viz_landmarks(input_img, landmark, columns=1)
    lab = sio.loadmat(os.path.join(Path.DEBUG_DIR, 'results/lab.mat'))
    edge_segment, patch_segment = lab['lab'][0][0]
    # viz_labelled_face(input_img, landmark, patch_segment)
    save_labelled_face(input_img, patch_segment)
