
from polymath.codegen.dnnweavergen.dnnweaver2.graph import Graph, get_default_graph

from polymath.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn import conv2D, maxPool, flatten, matmul, addBias, batch_norm, reorg, concat, leakyReLU
from polymath.codegen.dnnweavergen.dnnweaver2 import get_tensor
import logging
from polymath.codegen.dnnweavergen.dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

from polymath.codegen.dnnweavergen.dnnweaver2 import get_tensor
import pydot
import graphviz
from polymath.codegen.dnnweavergen.dnnweaver2.compiler import *


from polymath.codegen.dnnweavergen.dnnweaver2.simulator.accelerator import Accelerator

def yolo_convolution(tensor_in, filters=32, kernel_size=3,
        batch_normalize=True, act='leakyReLU',
        c_dtype=None, w_dtype=None,
        s_dtype=None, bn_dtype=None):

    input_channels = tensor_in.shape[-1]

    weights = get_tensor(shape=(filters, kernel_size, kernel_size, input_channels),
                         name='weights',
                         dtype=w_dtype)
    biases = get_tensor(shape=(filters),
                         name='biases',
                         dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    conv = conv2D(tensor_in, weights, biases, pad='SAME', dtype=c_dtype)

    if batch_normalize:
        with get_default_graph().name_scope('batch_norm'):
            mean = get_tensor(shape=(filters), name='mean', dtype=FixedPoint(16,c_dtype.frac_bits))
            scale = get_tensor(shape=(filters), name='scale', dtype=s_dtype)
            bn = batch_norm(conv, mean=mean, scale=scale, dtype=bn_dtype)
    else:
        bn = conv

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(bn, dtype=bn.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = bn
    else:
        raise ValueError('Unknown activation type {}'.format(act))

    return act

def get_graph_init(train=False):
    g = Graph('YOLOv2-Test: 16-bit', dataset='imagenet', log_level=logging.INFO)
    batch_size = 1

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size, 416, 416, 3), name='data', dtype=FQDtype.FXP16, trainable=False)
        with g.name_scope('conv0'):
            conv0 = yolo_convolution(i, filters=16, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 12),
                                     s_dtype=FixedPoint(16, 9), bn_dtype=FixedPoint(16, 8))
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1, 2, 2, 1), stride=(1, 2, 2, 1), pad='VALID')
        with g.name_scope('conv1'):
            conv1 = yolo_convolution(pool0, filters=32, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 8),
                                     s_dtype=FixedPoint(16, 14), bn_dtype=FixedPoint(16, 8))
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1, 2, 2, 1), stride=(1, 2, 2, 1), pad='VALID')
        with g.name_scope('conv2'):
            conv2 = yolo_convolution(pool1, filters=64, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     # batch_normalize=False, act='linear',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 10),
                                     s_dtype=FixedPoint(16, 13), bn_dtype=FixedPoint(16, 9))
        with g.name_scope('pool2'):
            pool2 = maxPool(conv2, pooling_kernel=(1, 2, 2, 1), stride=(1, 2, 2, 1), pad='VALID')
        with g.name_scope('conv3'):
            conv3 = yolo_convolution(pool2, filters=128, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 10),
                                     s_dtype=FixedPoint(16, 13), bn_dtype=FixedPoint(16, 10))
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1, 2, 2, 1), stride=(1, 2, 2, 1), pad='VALID')
        with g.name_scope('conv4'):
            conv4 = yolo_convolution(pool3, filters=256, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 11),
                                     s_dtype=FixedPoint(16, 13), bn_dtype=FixedPoint(16, 10))

        with g.name_scope('pool4'):
            pool4 = maxPool(conv4, pooling_kernel=(1, 2, 2, 1), stride=(1, 2, 2, 1), pad='VALID')
        with g.name_scope('conv5'):
            conv5 = yolo_convolution(pool4, filters=512, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 12),
                                     s_dtype=FixedPoint(16, 13), bn_dtype=FixedPoint(16, 11))
        with g.name_scope('pool5'):
            pool5 = maxPool(conv5, pooling_kernel=(1, 2, 2, 1), stride=(1, 1, 1, 1),
                            pad=((0, 0), (0, 1), (0, 1), (0, 0)))
        with g.name_scope('conv6'):
            conv6 = yolo_convolution(pool5, filters=1024, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 12),
                                     s_dtype=FixedPoint(16, 11), bn_dtype=FixedPoint(16, 9))
        with g.name_scope('conv7'):
            conv7 = yolo_convolution(conv6, filters=1024, kernel_size=3,
                                     batch_normalize=True, act='leakyReLU',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 11),
                                     s_dtype=FixedPoint(16, 14), bn_dtype=FixedPoint(16, 12))
        with g.name_scope('conv8'):
            conv8 = yolo_convolution(conv7, filters=125, kernel_size=1,
                                     batch_normalize=False, act='linear',
                                     w_dtype=FixedPoint(16, 14), c_dtype=FixedPoint(16, 11))

    return g


def get_graph(train=False):
    g = Graph('YOLOv2-Test: 16-bit', dataset='imagenet', log_level=logging.INFO)
    batch_size = 1

    with g.as_default(), g.name_scope('inputs'):
        i = get_tensor(shape=(batch_size,416,416,3), name='data', dtype=FQDtype.FXP16, trainable=False)
    with g.as_default():

        with g.name_scope('conv0'):
            conv0 = yolo_convolution(i, filters=16, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,12),
                    s_dtype=FixedPoint(16,9), bn_dtype=FixedPoint(16,8))
    with g.as_default():

        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
    with g.as_default():

        with g.name_scope('conv1'):
            conv1 = yolo_convolution(pool0, filters=32, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,8),
                    s_dtype=FixedPoint(16,14), bn_dtype=FixedPoint(16,8))
    with g.as_default():

        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
    with g.as_default():

        with g.name_scope('conv2'):
            conv2 = yolo_convolution(pool1, filters=64, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    # batch_normalize=False, act='linear',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,10),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,9))
    with g.as_default():

        with g.name_scope('pool2'):
            pool2 = maxPool(conv2, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
    with g.as_default():

        with g.name_scope('conv3'):
            conv3 = yolo_convolution(pool2, filters=128, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,10),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,10))
    with g.as_default():

        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
    with g.as_default():

        with g.name_scope('conv4'):
            conv4 = yolo_convolution(pool3, filters=256, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,11),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,10))

    with g.as_default():

        with g.name_scope('pool4'):
            pool4 = maxPool(conv4, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
    with g.as_default():

        with g.name_scope('conv5'):
            conv5 = yolo_convolution(pool4, filters=512, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,12),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,11))
    with g.as_default():

        with g.name_scope('pool5'):
            pool5 = maxPool(conv5, pooling_kernel=(1,2,2,1), stride=(1,1,1,1), pad=((0,0),(0,1),(0,1),(0,0)))
    with g.as_default():

        with g.name_scope('conv6'):
            conv6 = yolo_convolution(pool5, filters=1024, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,12),
                    s_dtype=FixedPoint(16,11), bn_dtype=FixedPoint(16,9))
    with g.as_default():

        with g.name_scope('conv7'):
            conv7 = yolo_convolution(conv6, filters=1024, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,11),
                    s_dtype=FixedPoint(16,14), bn_dtype=FixedPoint(16,12))
    with g.as_default():

        with g.name_scope('conv8'):
            conv8 = yolo_convolution(conv7, filters=125, kernel_size=1,
                    batch_normalize=False, act='linear',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,11))

    return g


def show_ops_tensors(graph):
    print('*' * 50)
    print('List of ops (nodes) in the graph_name')
    # print the ops in the yolo2_graph
    for op in graph.op_registry:
        print('\tOp name: {}'.format(op))
    print('*' * 50)

    print('*' * 50)
    print('List of tensors (edges) in the graph_name')
    # print the tensors in the yolo2_graph
    for key in graph.tensor_registry.keys():
        print('\t{}'.format(graph.tensor_registry[key]))
    print('*' * 50)

def generate_instructions(g):
    num_rows = 32
    num_cols = 32
    bram = {
        'ibuf': num_cols * 16 * 2048 / 2,
        'obuf': num_rows * 64 * 2048 / 2,
        'wbuf': num_cols * num_rows * 16 * 512 / 2,
        'bbuf': num_rows * 32 * 2048 / 2,
    }
    acc_obj = Accelerator(
        N=num_rows, M=num_cols,
        prec=16,
        mem_if_width=256,
        frequency=150e6,
        sram=bram
    )

    print(acc_obj.__str__())

def execute_graph(g):
    num_rows = 32
    num_cols = 32
    bram = {
        'ibuf': num_cols * 16 * 2048 / 2,
        'obuf': num_rows * 64 * 2048 / 2,
        'wbuf': num_cols * num_rows * 16 * 512 / 2,
        'bbuf': num_rows * 32 * 2048 / 2,
    }
    acc_obj = Accelerator(
        N=num_rows, M=num_cols,
        prec=16,
        mem_if_width=256,
        frequency=150e6,
        sram=bram
    )

    print(acc_obj.__str__())

    log_level = logging.INFO
    compiler = GraphCompiler(log_level=log_level)
    inst_binary = compiler.compile(graph=g, acc_obj=acc_obj)

    print('Number of instructions: {}'.format(inst_binary.size))
if __name__ == '__main__':
    g = get_graph()
    execute_graph(g)
    # show_ops_tensors(g)

# Yolo 1
# inputs / data[1, 416, 416, 3](FXP16(8, 8))
# conv0 / weights[16, 3, 3, 3](FXP16(2, 14))
# conv0 / biases[16](FXP32(10, 22))
# conv0 / Convolution[1, 416, 416, 16](FXP64(42, 22))
# conv0 / TypeCastOp[1, 416, 416, 16](FXP16(4, 12))
# conv0 / batch_norm / mean[16](FXP16(4, 12))
# conv0 / batch_norm / scale[16](FXP16(7, 9))
# conv0 / batch_norm / BatchNorm[1, 416, 416, 16](FXP32(11, 21))
# conv0 / batch_norm / TypeCastOp[1, 416, 416, 16](FXP16(8, 8))
# conv0 / leakyReLU / alpha[1](FP32)
# conv0 / leakyReLU / LeakyReLU[1, 416, 416, 16](FXP16(8, 8))
# pool0 / MaxPooling[1, 208, 208, 16](FXP16(8, 8))



# Yolo 2
# conv1 / weights[32, 3, 3, 16](FXP16(2, 14))
# conv1 / biases[32](FXP32(10, 22))
# conv1 / Convolution[1, 208, 208, 32](FXP64(42, 22))
# conv1 / TypeCastOp[1, 208, 208, 32](FXP16(8, 8))
# conv1 / batch_norm / mean[32](FXP16(8, 8))
# conv1 / batch_norm / scale[32](FXP16(2, 14))
# conv1 / batch_norm / BatchNorm[1, 208, 208, 32](FXP32(10, 22))
# conv1 / batch_norm / TypeCastOp[1, 208, 208, 32](FXP16(8, 8))
# conv1 / leakyReLU / alpha[1](FP32)
# conv1 / leakyReLU / LeakyReLU[1, 208, 208, 32](FXP16(8, 8))
# pool1 / MaxPooling[1, 104, 104, 32](FXP16(8, 8))
#
# Yolo 3
# conv2 / weights[64, 3, 3, 32](FXP16(2, 14))
# conv2 / biases[64](FXP32(10, 22))
# conv2 / Convolution[1, 104, 104, 64](FXP64(42, 22))
# conv2 / TypeCastOp[1, 104, 104, 64](FXP16(6, 10))
# conv2 / batch_norm / mean[64](FXP16(6, 10))
# conv2 / batch_norm / scale[64](FXP16(3, 13))
# conv2 / batch_norm / BatchNorm[1, 104, 104, 64](FXP32(9, 23))
# conv2 / batch_norm / TypeCastOp[1, 104, 104, 64](FXP16(7, 9))
# conv2 / leakyReLU / alpha[1](FP32)
# conv2 / leakyReLU / LeakyReLU[1, 104, 104, 64](FXP16(7, 9))
# pool2 / MaxPooling[1, 52, 52, 64](FXP16(7, 9))

# Yolo 4
# conv3 / weights[128, 3, 3, 64](FXP16(2, 14))
# conv3 / biases[128](FXP32(9, 23))
# conv3 / Convolution[1, 52, 52, 128](FXP64(41, 23))
# conv3 / TypeCastOp[1, 52, 52, 128](FXP16(6, 10))
# conv3 / batch_norm / mean[128](FXP16(6, 10))
# conv3 / batch_norm / scale[128](FXP16(3, 13))
# conv3 / batch_norm / BatchNorm[1, 52, 52, 128](FXP32(9, 23))
# conv3 / batch_norm / TypeCastOp[1, 52, 52, 128](FXP16(6, 10))
# conv3 / leakyReLU / alpha[1](FP32)
# conv3 / leakyReLU / LeakyReLU[1, 52, 52, 128](FXP16(6, 10))
# pool3 / MaxPooling[1, 26, 26, 128](FXP16(6, 10))

# Yolo 5
# conv4 / weights[256, 3, 3, 128](FXP16(2, 14))
# conv4 / biases[256](FXP32(8, 24))
# conv4 / Convolution[1, 26, 26, 256](FXP64(40, 24))
# conv4 / TypeCastOp[1, 26, 26, 256](FXP16(5, 11))
# conv4 / batch_norm / mean[256](FXP16(5, 11))
# conv4 / batch_norm / scale[256](FXP16(3, 13))
# conv4 / batch_norm / BatchNorm[1, 26, 26, 256](FXP32(8, 24))
# conv4 / batch_norm / TypeCastOp[1, 26, 26, 256](FXP16(6, 10))
# conv4 / leakyReLU / alpha[1](FP32)
# conv4 / leakyReLU / LeakyReLU[1, 26, 26, 256](FXP16(6, 10))
# pool4 / MaxPooling[1, 13, 13, 256](FXP16(6, 10))


# conv5 / weights[512, 3, 3, 256](FXP16(2, 14))
# conv5 / biases[512](FXP32(8, 24))
# conv5 / Convolution[1, 13, 13, 512](FXP64(40, 24))
# conv5 / TypeCastOp[1, 13, 13, 512](FXP16(4, 12))
# conv5 / batch_norm / mean[512](FXP16(4, 12))
# conv5 / batch_norm / scale[512](FXP16(3, 13))
# conv5 / batch_norm / BatchNorm[1, 13, 13, 512](FXP32(7, 25))
# conv5 / batch_norm / TypeCastOp[1, 13, 13, 512](FXP16(5, 11))
# conv5 / leakyReLU / alpha[1](FP32)
# conv5 / leakyReLU / LeakyReLU[1, 13, 13, 512](FXP16(5, 11))
# pool5 / MaxPooling[1, 13, 13, 512](FXP16(5, 11))

# Yolo 7
# conv6 / weights[1024, 3, 3, 512](FXP16(2, 14))
# conv6 / biases[1024](FXP32(7, 25))
# conv6 / Convolution[1, 13, 13, 1024](FXP64(39, 25))
# conv6 / TypeCastOp[1, 13, 13, 1024](FXP16(4, 12))
# conv6 / batch_norm / mean[1024](FXP16(4, 12))
# conv6 / batch_norm / scale[1024](FXP16(5, 11))
# conv6 / batch_norm / BatchNorm[1, 13, 13, 1024](FXP32(9, 23))
# conv6 / batch_norm / TypeCastOp[1, 13, 13, 1024](FXP16(7, 9))
# conv6 / leakyReLU / alpha[1](FP32)
# conv6 / leakyReLU / LeakyReLU[1, 13, 13, 1024](FXP16(7, 9))

# Yolo 8
# conv7 / weights[1024, 3, 3, 1024](FXP16(2, 14))
# conv7 / biases[1024](FXP32(9, 23))
# conv7 / Convolution[1, 13, 13, 1024](FXP64(41, 23))
# conv7 / TypeCastOp[1, 13, 13, 1024](FXP16(5, 11))
# conv7 / batch_norm / mean[1024](FXP16(5, 11))
# conv7 / batch_norm / scale[1024](FXP16(2, 14))
# conv7 / batch_norm / BatchNorm[1, 13, 13, 1024](FXP32(7, 25))
# conv7 / batch_norm / TypeCastOp[1, 13, 13, 1024](FXP16(4, 12))
# conv7 / leakyReLU / alpha[1](FP32)
# conv7 / leakyReLU / LeakyReLU[1, 13, 13, 1024](FXP16(4, 12))

# Yolo 9
# conv8 / weights[125, 1, 1, 1024](FXP16(2, 14))
# conv8 / biases[125](FXP32(6, 26))
# conv8 / Convolution[1, 13, 13, 125](FXP64(38, 26))
# conv8 / TypeCastOp[1, 13, 13, 125](FXP16(5, 11))


# Init data name: data
# Accelerator object
# 	Precision: 16
# 	Systolic array size: 32 -rows x 32 -columns
# 	IBUF size:   65,536.0 Bytes
# 	WBUF size:  524,288.0 Bytes
# 	OBUF size:  262,144.0 Bytes
# 	BBUF size:  131,072.0 Bytes
# Double buffering enabled. Sizes of SRAM are halved
# Number of instructions: 1433

# Accelerator object
# 	Precision: 16
# 	Systolic array size: 32 -rows x 32 -columns
# 	IBUF size:   65,536.0 Bytes
# 	WBUF size:  524,288.0 Bytes
# 	OBUF size:  262,144.0 Bytes
# 	BBUF size:  131,072.0 Bytes
# Double buffering enabled. Sizes of SRAM are halved
# Number of instructions: 1433





