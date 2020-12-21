from google.protobuf.json_format import MessageToJson
from typing import Union, List, Dict
from pytools import memoize_method
from typing import TYPE_CHECKING, Dict

FUNC_WRAPPER_NODES = ["func_op", "slice_op", "sum", "prod",
                      "argmin", "argmax", "amin", "amax"]