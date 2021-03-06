import os
import pandas as pd
import numpy as np
import scipy.io as sio
import skimage.io as skio
import matplotlib.pylab as plt
import subprocess
from PIL import Image

from utils.path import Path


def get_result_from_server(file_name):
    subprocess.call([
        'scp',
        'ntq@192.168.1.200:' +
        '/home/ntq/thanhtc/hair_segmentation/debug/results/' + file_name,
        '/home/nero/py/nextsmarty/hair_segmentation/debug/results/'])
    # subprocess.call([
    #     'scp',
    #     'ntq@192.168.1.200:' +
    #     '/home/ntq/thanhtc/hair_segmentation/debug/results/py_out2.mat',
    #     '/home/nero/py/nextsmarty/hair_segmentation/debug/results/'])


def add_overlay(background, overlay):
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(background, overlay, 0.9)
    new_img = new_img.convert('RGB')
    return new_img


def viz_landmarks(input_img, shapes, columns, rows=1, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(input_img)
    plt.scatter(x=shapes[:, 0], y=shapes[:, 1], c='w', s=20)
    plt.show()


def viz_labelled_face(input_img, patch_segments, axis=0, figsize=(12, 12)):
    fig = plt.figure(figsize=figsize)
    hair_patchs = []
    for i, p in enumerate(patch_segments):
        if axis is None:
            hair_patch = p
        else:
            hair_patch = p[:, :, axis]

        hair_patch = hair_patch * 255
        hair_patch.astype('uint8')
        hair_patchs.append(hair_patch)

    background = Image.fromarray(input_img)
    fig.add_subplot(1, len(patch_segments) + 1, 1)
    plt.imshow(background)
    for i in range(len(patch_segments)):
        fig.add_subplot(1, len(patch_segments) + 1, i + 2)
        overlay = Image.fromarray(hair_patchs[i])
        new_img = add_overlay(background, overlay)
        plt.imshow(new_img)

    plt.show()


def viz_net_output(out2_files, figsize=(12, 12)):
    fig = plt.figure(figsize=figsize)
    actives = []
    for out2 in out2_files:
        out = sio.loadmat(out2)
        active = out['active_fc'][:, :, :, 0].astype('uint8')
        actives.append(active)

    for i in range(len(out2_files)):
        fig.add_subplot(1, len(out2_files), i + 1)
        img = Image.fromarray(actives[i])
        plt.imshow(img)

    plt.show()


def save_labelled_face(input_img, patch_segment, edge_segment):
    hair_patch = patch_segment[:, :, 1]
    hair_patch = hair_patch * 255
    hair_patch.astype('uint8')

    background = Image.fromarray(input_img)
    overlay = Image.fromarray(hair_patch)
    new_img = add_overlay(background, overlay)
    new_img.save(os.path.join(Path.DEBUG_DIR, 'data/new.png'), 'PNG')


if __name__ == '__main__':
    for i in range(1, 7):
        # get_result_from_server('{}_new_lab.mat'.format(i))
        input_img = skio.imread(os.path.join(
            Path.DEBUG_DIR, 'data/{}_new.jpg'.format(i)))
        lab_P = sio.loadmat(os.path.join(
            Path.DEBUG_DIR, 'results/{}_new_lab.mat'.format(i)))
        edge_segment_P, patch_segment_P = lab_P['lab'][0][0]
        viz_labelled_face(input_img, [patch_segment_P], axis=1)
    # landmark = pd.read_csv(os.path.join(Path.DEBUG_DIR, 'data/1_5pt.txt'),
    #                        header=None, sep=" ").values
    # viz_landmarks(input_img, landmark, columns=1)
    # lab_M = sio.loadmat(os.path.join(Path.DEBUG_DIR, 'results/mat_lab.mat'))

    # edge_segment_M, patch_segment_M = lab_M['lab'][0][0]
    # viz_labelled_face(input_img, [patch_segment_M, patch_segment_P], axis=1)
    # viz_labelled_face(input_img, [edge_segment_M, edge_segment_P], axis=None)
    # viz_labelled_face(input_img, [edge_segment_P], axis=None)
    # save_labelled_face(input_img, patch_segment_P, edge_segment_P)
    # viz_net_output([
    #    os.path.join(Path.DEBUG_DIR, 'results/mat_out2.mat'),
    #    os.path.join(Path.DEBUG_DIR, 'results/py_out2.mat')])
