import numpy as np
from pathlib import Path
LUT_PATH = f"{Path(f'{__file__}').parent}"
SIGMOID_PATH = f"{LUT_PATH}/sigmoid_lookup.csv"

def get_sigmoid_lut():
    lvals = np.loadtxt(SIGMOID_PATH, delimiter=",")
    return np.asarray([i[1] for i in lvals])

def sigmoid_lut(val, minv=-256, maxv=255, div_size=128):
    val = np.floor(val/4).astype(np.int)
    val = val.clip(minv, maxv) + maxv + 1
    lut = get_sigmoid_lut()
    return np.floor(lut[val]/div_size)


def get_gaussian_lut():
    lvals = np.loadtxt(SIGMOID_PATH, delimiter=",")
    return np.asarray([i[1] for i in lvals])

def gaussian_lut(val, minv=-256, maxv=255):
    raise NotImplementedError