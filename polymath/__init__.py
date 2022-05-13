from __future__ import print_function, division, absolute_import

DEFAULT_SHAPES = [(1,), (1,)]
UNSET_SHAPE = tuple([])
SCALAR_IDX = (0,)
# TODO: Need to add all func operators here from base class
from polymath.srdfg.domain import Domain
from polymath.srdfg.base import Node, nodeop, func_op, contains,\
    import_, control_dependencies, pow_, EvaluationError, Graph, int_, \
    mul, sub, add, call, var_index
from polymath.srdfg.nodes import variable, predicate, assert_, str_format, identity, lazy_constant, try_,\
    placeholder, temp, parameter, slice_op, input, state, output, write
from polymath.srdfg.index import index, index_op
from polymath.srdfg.group_nodes import GroupNode, sum, prod, max, min, argmin, argmax, bitreverse
from polymath.srdfg.nonlinear import NonLinear, sigmoid, log2, log10, exp, abs, sqrt, ceil, \
    floor, cast, tanh, square, log, rsqrt, clip, logical_not, logical_or
from polymath.srdfg.template import Template
from polymath.srdfg.transformations import Transformation, unsqueeze, squeeze, flatten, gather, \
    reshape, gather_elements, transpose, pad, flip
from polymath.srdfg.util import Profiler, visualize, lower_graph, is_iterable
from polymath.srdfg.serialization.serialize import pb_store, pb_load

from polymath.srdfg.templates.data_analytics import linear_regressor_train,\
    svm_classifier_train, logistic_regressor_train, logistic_regressor

from polymath.srdfg.templates.dnn import conv_bias, depthwise_conv, depthwise_conv_bias, dense, relu, avg_pool2d,\
    batch_flatten, softmax, relu1d, dense_sigmoid, batch_norm,\
    global_avg_pool, conv, max_pool, dropout, leaky_relu, avg_pool, lrn, \
    elem_tanh, elem_sigmoid, elem_cast, conv_transpose, cross_entropy_loss, log_softmax, \
    nll_loss, conv_transpose_bias, elem_floor, elem_ceil, elem_clip, elem_exp, topk,\
    split, elem_if, elem_sqrt, elem_log, roi_align, elem_where, scatter_elements, \
    loop, nms, concat, one_hot, gelu, bias_add

from polymath.srdfg.templates.fused_dnn import conv_bias_relu,\
    conv_bias_relu_max_pool, \
    conv_bias_add_relu,\
    conv_bias_add_relu_global_avg_pool

from polymath.srdfg.templates.optimizers import sgd
from polymath.srdfg.templates.gradient_defs import gemm_grad, gemm_grad_no_bias, conv_grad, conv_grad_no_bias, \
    flatten_grad, elem_add_grad, relu_grad, batchnorm_grad, global_average_pool_grad, max_pool_grad,\
    cross_entropy_loss_grad, average_pool_grad, elem_tanh_grad

from polymath.srdfg.templates.gradient_defs import AUTODIFF_OPS

from polymath.srdfg.templates.math import elem_mul, elem_sub, elem_div, reduce_sum, matmul, gemm, \
    elem_add, elem_greater, lvmatmul, rvmatmul, gemm_no_bias, reduce_min, reduce_max, elem_min, elem_max,\
    elem_less, elem_not, elem_or, elem_and, elem_nonzero, reduce_prod, elem_equal, mean_var, reduce_mean,\
    elem_pow, reciprocal

from polymath.srdfg.templates.tensor_transformations import coarse_flatten, elem_gather, tensor_transpose, onnx_reshape, \
    onnx_squeeze, onnx_identity, onnx_resize, \
    onnx_unsqueeze, tensor_pad, tensor_flip, tensor_reshape, tensor_squeeze, resize

from polymath.srdfg.from_onnx.converter import from_onnx, get_attributes, get_value_info_shape, ONNX_OP_NAMES
from polymath.srdfg.from_pytorch.converter import from_pytorch, get_attributes, get_value_info_shape, PYTORCH_OP_NAMES
DNN_TRAINING_OPS = AUTODIFF_OPS + ONNX_OP_NAMES

from polymath.srdfg.passes import register_pass, Pass
from polymath.srdfg.passes.dnn_passes import UpdateBatchSize, CollectDNNShapes, RenameMultiDimOps, UpdateLayout, \
    FuseOps, SplitOps
from polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower, CountNodes, CountOpTypes
from polymath.srdfg.passes.autodiff import AutoDiffGraph, create_training_graph
from polymath.codegen.tabla.tabla_translate import generate_tabla

from polymath.tools.srdfg_helpers import print_graph_ops
