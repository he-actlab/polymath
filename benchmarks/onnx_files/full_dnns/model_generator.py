import argparse
from onnxsim import simplify
import polymath as pm
import onnx
import tf2onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import io
from onnxsim import simplify
from onnx import optimizer
import tensorflow.compat.v1 as tf

def contains_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.is_contiguous(memory_format=torch.channels_last) and not t.is_contiguous():
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_cl(list(t)):
                return True
    return False


def print_inputs(args, indent=''):
    for t in args:
        if isinstance(t, torch.Tensor):
            print(indent, t.stride(), t.shape, t.device, t.dtype)
        elif isinstance(t, list) or isinstance(t, tuple):
            print(indent, type(t))
            print_inputs(list(t), indent=indent + '    ')
        else:
            print(indent, t)


def check_wrapper(fn):
    name = fn.__name__

    def check_cl(*args, **kwargs):
        was_cl = contains_cl(args)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            print('-------------------')
            raise e
        failed = False
        if was_cl:
            if isinstance(result, torch.Tensor):
                if result.dim() == 4 and not result.is_contiguous(memory_format=torch.channels_last):
                    print("`{}` got channels_last input, but output is not channels_last:".format(name),
                          result.shape, result.stride(), result.device, result.dtype)
                    failed = True
        if failed and True:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            raise Exception(
                'Operator `{}` lost channels_last property'.format(name))
        return result
    return check_cl

old_attrs = dict()

def attribute(m):
    old_attrs[m] = dict()
    for i in dir(m):
        e = getattr(m, i)
        exclude_functions = ['is_cuda', 'has_names', 'numel',
                             'stride', 'Tensor', 'is_contiguous', '__class__']
        if i not in exclude_functions and not i.startswith('_') and '__call__' in dir(e):
            try:
                old_attrs[m][i] = e
                setattr(m, i, check_wrapper(e))
            except Exception as e:
                print(i)
                print(e)


def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)

def create_lenet(optimize_model, training_mode, convert_data_format, to_polymath):
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                                   kernel_size=5, stride=1, padding=0, bias=False)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                                   kernel_size=5, stride=1, padding=0, bias=False)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                                   kernel_size=5, stride=1, padding=0, bias=False)
            self.linear1 = nn.Linear(120, 84)
            self.linear2 = nn.Linear(84, 10)
            self.tanh = nn.Tanh()
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.tanh(x)
            x = self.avgpool(x)
            x = self.conv2(x)
            x = self.tanh(x)
            x = self.avgpool(x)
            x = self.conv3(x)
            x = self.tanh(x)
            x = torch.flatten(x, 1)
            x = self.linear1(x)
            x = self.tanh(x)
            x = self.linear2(x)
            return x
    model = LeNet()
    input_var = torch.randn(1, 1, 32, 32)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "lenet", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_resnet18(optimize_model, training_mode, convert_data_format, to_polymath):
    model = models.resnet18(pretrained=not training_mode)
    input_var = torch.randn(1, 3, 224, 224)
    if not training_mode:
        output = model(input_var)
        model.eval()
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, "resnet18", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_resnet50(optimize_model, training_mode, convert_data_format, to_polymath):
    model = models.resnet50(pretrained=not training_mode)
    input_var = torch.randn(1, 3, 224, 224)
    if not training_mode:
        output = model(input_var)
        model.eval()
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, "resnet50", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath,
                        convert_data_format=False):
    f = io.BytesIO()
    mode = torch.onnx.TrainingMode.TRAINING if training_mode else torch.onnx.TrainingMode.EVAL
    torch.onnx.export(model,  # model being run
                      input_var,  # model input (or a tuple for multiple inputs)
                      f,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      training=mode,
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model_proto = onnx.ModelProto.FromString(f.getvalue())
    onnx.checker.check_model(model_proto)
    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    filepath = f"./{model_name}.onnx"
    # model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
    if optimize_model:
        model_proto, check = simplify(model_proto)
        assert check
    with open(filepath, "wb") as f:
        f.write(model_proto.SerializeToString())

    if to_polymath:
        graph = pm.from_onnx(filepath)
        pm.pb_store(graph, "./")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "resnet18", "lenet')

    argparser.add_argument('-o', '--optimize_model', type=str2bool, nargs='?', default=True,
                           const=True, help='Optimize the model')

    argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
                           const=True, help='Whether or not the model is in training mode')

    argparser.add_argument('-df', '--data_format_convert', type=str2bool, nargs='?', default=False,
                           const=True, help='Whether or not the model is in training mode')


    argparser.add_argument('-pm', '--to_polymath', type=str2bool, nargs='?', default=False,
                           const=True, help='Whether or not the model should be converted to PolyMath')
    args = argparser.parse_args()
    if args.benchmark == "lenet":
        create_lenet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
    elif args.benchmark == "resnet18":
        create_resnet18(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
    elif args.benchmark == "resnet50":
        create_resnet50(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
    else:
        raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
                           f"\"lenet\", \"resnet18\".")