from google.protobuf.json_format import MessageToJson
from typing import Union, List, Dict
from pytools import memoize_method
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from polymath.mgdfg.variables import Variable, Index
    from polymath.mgdfg.template import Template
    from polymath.mgdfg.graph_objects import Node, Edge
FUNC_WRAPPER_NODES = ["func_op", "slice_op", "sum", "prod",
                      "argmin", "argmax", "min", "max"]
