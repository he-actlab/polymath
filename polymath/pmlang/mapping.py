from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np  # type: ignore
from polymath.srdfg.serialization.mgdfg_pb2 import Tensor

logger = logging.getLogger(__name__)

TENSOR_TYPE_TO_NP_TYPE = {
    int(Tensor.FLOAT): np.dtype('float32'),
    int(Tensor.UINT8): np.dtype('uint8'),
    int(Tensor.INT8): np.dtype('int8'),
    int(Tensor.UINT16): np.dtype('uint16'),
    int(Tensor.INT16): np.dtype('int16'),
    int(Tensor.INT32): np.dtype('int32'),
    int(Tensor.INT64): np.dtype('int64'),
    int(Tensor.BOOL): np.dtype('bool'),
    int(Tensor.FLOAT16): np.dtype('float16'),
    int(Tensor.DOUBLE): np.dtype('float64'),
    int(Tensor.COMPLEX64): np.dtype('complex64'),
    int(Tensor.COMPLEX128): np.dtype('complex128'),
    int(Tensor.UINT32): np.dtype('uint32'),
    int(Tensor.UINT64): np.dtype('uint64'),
    int(Tensor.STRING): np.dtype(np.object)
}

NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = {
    int(Tensor.FLOAT): int(Tensor.FLOAT),
    int(Tensor.UINT8): int(Tensor.INT32),
    int(Tensor.INT8): int(Tensor.INT32),
    int(Tensor.UINT16): int(Tensor.INT32),
    int(Tensor.INT16): int(Tensor.INT32),
    int(Tensor.INT32): int(Tensor.INT32),
    int(Tensor.INT64): int(Tensor.INT64),
    int(Tensor.BOOL): int(Tensor.INT32),
    int(Tensor.FLOAT16): int(Tensor.UINT16),
    int(Tensor.BFLOAT16): int(Tensor.UINT16),
    int(Tensor.DOUBLE): int(Tensor.DOUBLE),
    int(Tensor.COMPLEX64): int(Tensor.FLOAT),
    int(Tensor.COMPLEX128): int(Tensor.DOUBLE),
    int(Tensor.UINT32): int(Tensor.UINT32),
    int(Tensor.UINT64): int(Tensor.UINT64),
    int(Tensor.STRING): int(Tensor.STRING),
}

STORAGE_TENSOR_TYPE_TO_FIELD = {
    int(Tensor.FLOAT): 'float_data',
    int(Tensor.INT32): 'int32_data',
    int(Tensor.INT64): 'int64_data',
    int(Tensor.UINT16): 'int32_data',
    int(Tensor.DOUBLE): 'double_data',
    int(Tensor.COMPLEX64): 'float_data',
    int(Tensor.COMPLEX128): 'double_data',
    int(Tensor.UINT32): 'uint64_data',
    int(Tensor.UINT64): 'uint64_data',
    int(Tensor.STRING): 'string_data',
    int(Tensor.BOOL): 'int32_data',
}

STRING_TEXT_TO_TENSOR_TYPE = {
    'float' : int(Tensor.FLOAT),
    'int' : int(Tensor.INT32),
    'complex' : int(Tensor.COMPLEX64),
    'str' : int(Tensor.STRING),
    'bool' : int(Tensor.BOOL),
}

STRING_TEXT_TO_BINEXP = {"*": "mul",
                         "/": "div",
                         "+": "add",
                         "-": "sub",
                         "<": "tlt",
                         ">": "tgt",
                         "<=": "tlte",
                         ">=": "tgte",
                         "==": "teq",
                         "!=": "tne",
                          "^": "exp",
                         "%": "mod"
                         }

STRING_TEXT_TO_UNEXP = {"+" : "mov",
                        "-": "neg"}

STRING_TEXT_TO_FUNCTION = {"pi": "mov",
             "log": "log",
             "log2": "log2",
             "float": "cast",
             "int": "cast",
             "bin": "cast",
             "ceiling": "ceil",
             "floor": "floor",
             "e": "mov",
             "fread": "fread",
             "fwrite": "fwrite",
           "sigmoid" : "sigmoid"}
STRING_FUNCTION_TO_STRING_TYPE = {"pi": 'float',
                   "log": 'float',
                   "log2": 'float',
                   "float": 'float',
                   "int": 'int',
                   "bin": 'int',
                   "random": 'float',
                   "ceiling": 'float',
                   "floor": 'float',
                   "e": 'float',
                   "fread": 'str',
                   "fwrite": 'str',
                  "sigmoid" : "float"}