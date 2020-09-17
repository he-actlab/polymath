from __future__ import print_function, division, absolute_import

DEFAULT_SHAPES = [(1,), (1,)]
UNSET_SHAPE = tuple([])
SCALAR_IDX = (0,)
# TODO: Need to add all func operators here from base class
from polymath.mgdfg.domain import Domain
from polymath.mgdfg.base import Node, nodeop, func_op, contains,\
    import_, control_dependencies, pow_, EvaluationError, Graph, int_, \
    mul, sub, add, call, var_index
from polymath.mgdfg.nodes import variable, predicate, assert_, str_format, identity, lazy_constant, try_,\
    placeholder, temp, parameter, slice_op, input, state, output, write
from polymath.mgdfg.index import index, index_op
from polymath.mgdfg.group_nodes import GroupNode, sum, prod, max, min, argmin, argmax, bitreverse
from polymath.mgdfg.nonlinear import NonLinear, sigmoid, log2, exp, abs, sqrt, ceil, floor, cast, tanh
from polymath.mgdfg.template import Template
from polymath.mgdfg.util import Profiler, visualize, lower_graph, is_iterable
from polymath.mgdfg.serialization.serialize import pb_store, pb_load
from polymath.mgdfg.from_onnx.converter import from_onnx, get_attributes, get_value_info_shape
from polymath.mgdfg.from_onnx.node_definitions import linear_regressor_train,\
    svm_classifier_train, logistic_regressor_train, logistic_regressor, conv_bias, dense, relu, avg_pool2d,\
    batch_flatten, softmax, relu1d, dense_sigmoid, elem_mul, elem_sub, reduce_sum, matmul, batch_norm,\
    global_avg_pool, conv, max_pool, flatten, dropout, gemm, leaky_relu, avg_pool, lrn

from polymath.mgdfg.passes import register_pass, Pass
from polymath.mgdfg.passes.compiler_passes import NormalizeGraph, Lower, CountNodes, CountOpTypes
from polymath.codegen.tabla.tabla_translate import generate_tabla

try:
    from polymath.codegen.dnnweavergen.dnnweaver_translate import generate_dnnweaver
except ImportError:
    print(f"WARNING: DNNWeaver translation could not be imported because DNNWeaver is not currently installed.")
    generate_dnnweaver = None

try:
    from polymath.codegen.tvmgen.tvm_translate import generate_tvm
except ImportError:
    print(f"WARNING: TVM translation could not be imported because TVM is not currently installed.")
    generate_tvm = None