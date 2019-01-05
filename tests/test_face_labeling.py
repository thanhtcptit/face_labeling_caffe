import scipy.io as sio


def read_label(fpath):
    lab = sio.loadmat(fpath)
    edge_segment, patch_segment = lab['lab'][0][0]
    hair_soft_mask = patch_segment[:, :, 1] * 255
    hair_soft_mask = hair_soft_mask.astype(np.uint8)

    print("read_label", type(hair_soft_mask),
          hair_soft_mask.dtype, hair_soft_mask.shape)
    return hair_soft_mask


def run_FaceLabelling(img_path, lm_path):
    os.chdir(Path.FACE_LABELING_ROOT)
    print(Path.FACE_LABELING_ROOT)
    print(os.getcwd())
    rc = subprocess.call([os.path.join(Path.FACE_LABELING_ROOT, 'run.sh'),
                          img_path, lm_path])
    print("FaceLabeling returns code", rc)
    os.chdir(Path.PROJECT_ROOT)
    if rc == 0 or rc == 137:
        print('Path.FACE_LABELING_OUT', Path.FACE_LABELING_OUT)
        label = read_label(fpath=os.path.join(Path.FACE_LABELING_OUT))
        return rc, label
    else:
        raise RuntimeError

