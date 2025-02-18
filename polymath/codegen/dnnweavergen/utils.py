



from polymath.codegen.codegen_utils import get_func

from polymath.codegen.dnnweavergen.dnnweaver2.compiler import *
from polymath.codegen.dnnweavergen.dnnweaver2.simulator.accelerator import Accelerator
from polymath.codegen.dnnweavergen.dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
dtype_map = {
            "cout" : [12, 8, 10, 10, 11, 12, 12, 11, 11],
            "bn" : [8, 8, 9, 10, 10, 11, 9, 12],
             }

dtype_counters = {
    "cout" : 0,
    "bn": 0
}
def dnnweaver_init_weight(g,scope, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.get_tensor"

    with g.as_default():
        with g.name_scope(scope):
            return get_func(fname)(*args, dtype=FixedPoint(16,14), **kwargs)

def dnnweaver_init_bias(g, scope, *args, **kwargs):

    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.get_tensor"

    with g.as_default():
        with g.name_scope(scope):
            return get_func(fname)(*args, dtype=FixedPoint(32,22), **kwargs)

def dnnweaver_init_data(g,scope, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.get_tensor"
    print(f"Init data name: {kwargs['name']}")

    with g.as_default():
        with g.name_scope(scope):
            return get_func(fname)(*args, dtype=FQDtype.FXP16, **kwargs)

def dnnweaver_init_scale(g, scope,input_op, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.get_tensor"
    with g.as_default():
        with g.name_scope(scope):
            with g.name_scope(input_op):
                return get_func(fname)(*args, dtype=FixedPoint(16,9), **kwargs)


def dnnweaver_init_mean(g, scope,input_op, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.get_tensor"
    with g.as_default():
        with g.name_scope(scope):
            with g.name_scope(input_op):
                return get_func(fname)(*args, dtype=FixedPoint(16,9), **kwargs)

def dnnweaver_context(g,scope, fname, *args, **kwargs):

    with g.as_default():
        with g.name_scope(scope):
            return get_func(fname)(*args, **kwargs)


def dnnweaver_conv2d(g,scope, *args, **kwargs):

    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn.conv2D"
    with g.as_default():
        with g.name_scope(scope):
            dtype_counters['cout'] += 1
            return get_func(fname)(*args, dtype=FixedPoint(16, dtype_map['cout'][dtype_counters['cout'] - 1]), **kwargs)

def dnnweaver_batch_norm(g,scope, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn.batch_norm"

    with g.as_default():
        with g.name_scope(scope):
            with g.name_scope('batch_norm'):
                dtype_counters['bn'] +=1
                return get_func(fname)(*args, dtype=FixedPoint(16, dtype_map['bn'][dtype_counters['bn'] - 1]),**kwargs)

def dnnweaver_max_pool(g,scope, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn.maxPool"
    if 'pad' in kwargs.keys():
        kwargs['pad'] = ((0,0), (0, kwargs['pad'][0]), (0, kwargs['pad'][0]), (0,0))
    with g.as_default():
        with g.name_scope(scope):
            return get_func(fname)(*args, **kwargs)

def dnnweaver_leaky_relu(g,scope, *args, **kwargs):
    fname = "cmstack.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn.leakyReLU"

    with g.as_default():
        with g.name_scope(scope):
            return get_func(fname)(*args, **kwargs)

def dnnweaver_nyi(orig_func, *args, **kwargs):
    logging.error(f"Function {orig_func} has not yet been implemented. Exiting")
    exit(1)



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