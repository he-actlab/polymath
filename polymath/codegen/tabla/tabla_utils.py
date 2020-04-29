import numpy as np
from pathlib import Path
LUT_PATH = f"{Path(f'{__file__}').parent}/lookup_tables"
SIGMOID_PATH = f"{LUT_PATH}/sigmoid.csv"

def get_sigmoid_lut():
    lvals = np.loadtxt(SIGMOID_PATH, delimiter=",")
    return np.asarray([i[1] for i in lvals])

def sigmoid_lut(val, minv=-256, maxv=255):
    val = val.astype(np.int)
    val = val.clip(minv, maxv) + 256
    lut = get_sigmoid_lut()
    return lut[val]


def get_gaussian_lut():
    lvals = np.loadtxt(SIGMOID_PATH, delimiter=",")
    return np.asarray([i[1] for i in lvals])

def gaussian_lut(val, minv=-256, maxv=255):
    raise NotImplementedError