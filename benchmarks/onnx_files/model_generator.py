import argparse
from onnxsim import simplify
import polymath as pm
import torch
import torch.nn as nn
from onnx_utils import extract_model
import onnx
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import io
from pathlib import Path
from onnxsim import simplify
from collections import namedtuple
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
import onnx
CWD = Path(f"{__file__}").parent

Targets = namedtuple('Targets', ['boxes', 'masks', 'labels'])


def get_image_from_url(url, size=None):
    import requests
    from PIL import Image
    from io import BytesIO
    from torchvision import transforms

    data = requests.get(url)
    image = Image.open(BytesIO(data.content)).convert("RGB")

    if size is None:
        size = (300, 200)
    image = image.resize(size, Image.BILINEAR)

    to_tensor = transforms.ToTensor()
    return to_tensor(image)


def get_test_images():
    image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
    image = get_image_from_url(url=image_url, size=(100, 320))

    image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
    image2 = get_image_from_url(url=image_url2, size=(250, 380))

    images = [image]
    test_images = [image2]
    return images, test_images

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
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                                   kernel_size=5, stride=1, padding=0, bias=True)
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


def create_custom_conv(optimize_model, training_mode, convert_data_format, to_polymath, input_shape, oc, ksize, stride, pad):
    n, ic, h, w = input_shape
    class CustomConv(nn.Module):
        def __init__(self):
            super(CustomConv, self).__init__()

            self.conv = nn.Conv2d(in_channels=ic, out_channels=oc,
                                   kernel_size=ksize, stride=stride, padding=pad, bias=True)

        def forward(self, x):
            x = self.conv(x)
            return x
    model = CustomConv()
    input_var = torch.randn(n, ic, h, w)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "custom_conv", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_custom_matmul(optimize_model, training_mode, convert_data_format, to_polymath, M, N, P, include_bias=False):
    class CustomMatmul(nn.Module):
        def __init__(self):
            super(CustomMatmul, self).__init__()

            self.mmul = nn.Linear(N,  P, bias=include_bias)

        def forward(self, x):
            x = self.mmul(x)
            return x
    model = CustomMatmul()
    input_var = torch.randn(M, N)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "custom_matmul", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_lenet_bn(optimize_model, training_mode, convert_data_format, to_polymath):
    class LeNetBN(nn.Module):
        def __init__(self):
            super(LeNetBN, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.bn1 = nn.BatchNorm2d(6)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.bn2 = nn.BatchNorm2d(16)

            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.bn3 = nn.BatchNorm2d(120)
            self.linear1 = nn.Linear(120, 84)
            self.linear2 = nn.Linear(84, 10)
            self.relu = nn.ReLU()
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = torch.flatten(x, 1)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    model = LeNetBN()
    input_var = torch.randn(3, 1, 32, 32)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "lenet_bn", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_resnet18(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.resnet18(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "resnet18"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        output = model(input_var)
        model.eval()
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_vgg16(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.vgg16(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "vgg16"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_inception(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.inception_v3(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 299, 299)
    model_name = "inceptionv3"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_mobilenet(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.mobilenet_v2(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "mobilenetv2"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_alexnet(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.alexnet(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "alexnet"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_efficientnet(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.alexnet(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "efficientnet"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_resnet50(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.resnet50(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "resnet50"

    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        output = model(input_var)
        model.eval()
    else:
        model_name = f"{model_name}_train"



    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def _make_empty_samples(N, C, H, W, training=False):

    img, other = get_test_images()
    t = Targets(boxes=torch.rand(0, 4), labels=torch.tensor([]).to(dtype=torch.int64),
                masks=torch.rand(0, H, W))

    return img, [t._asdict()]

def _make_mrcnn_samples():
    img, other = get_test_images()
    dummy_image = [torch.ones(3, 100, 100) * 0.3]
    return img, other, dummy_image


def create_maskrcnn(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=not training_mode, min_size=200, max_size=300)

    N, C, H, W = 1, 1, 300, 300
    # inputs = _make_empty_samples(N, C, H, W, training=training_mode)
    images, test_images, dummy_image = _make_mrcnn_samples()
    model_name = "mask_rcnn_vision"


    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        model.eval()
        model(images)
        input_var = [(images,), (test_images,), (dummy_image,)]
    else:
        model_name = f"{model_name}_train"
        model.train()
        input_var = [(images,)]

    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


class RPNModule(torch.nn.Module):
    def __init__(self):
        super(RPNModule, self).__init__()

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        out_channels = 256
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.0

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

    def forward(self, images, features):
        images = ImageList(images, [i.shape[-2:] for i in images])
        return self.rpn(images, features)

def create_maskrcnn_part(part_name, optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=not training_mode, min_size=200, max_size=300)

    images, test_images, dummy_image = _make_mrcnn_samples()
    model_name = f"mask_rcnn_vision_{part_name}"

    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        model.eval()
        transformed_input, targets = model.transform(images, None)
        features = model.backbone(transformed_input.tensors)

        proposals, proposal_losses = model.rpn(transformed_input, features, targets)
        detections, detector_losses = model.roi_heads(features, proposals, transformed_input.image_sizes, targets)
    else:
        model.eval()
        transformed_input, targets = model.transform(images, None)
        features = model.backbone(transformed_input.tensors)
        proposals, proposal_losses = model.rpn(transformed_input, features, targets)
        detections, detector_losses = model.roi_heads(features, proposals, transformed_input.image_sizes, targets)
    f = io.BytesIO()
    store_path = f"{CWD}/full_dnns/{model_name}.onnx"

    if part_name == "backbone":
        inputs = (transformed_input.tensors,)
        torch.onnx.export(model.backbone,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f,  # where to save the model (can be a file or file-like object)
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=["images_tensors"],
                          output_names=["feature_pool", "feature0", "feature1", "feature2", "feature3"],
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          )

    elif part_name == "rpn":
        rpn_model = RPNModule()
        rpn_model.eval()
        inputs = (transformed_input.tensors, features) + ({},)
        torch.onnx.export(rpn_model,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f,
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          )
    elif part_name == "roi":
        inputs = (features, proposals, transformed_input.image_sizes)
        torch.onnx.export(model.roi_heads,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f,
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          )
    model_proto = onnx.ModelProto.FromString(f.getvalue())

    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    with open(store_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    # convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)
    #


def _print_nodes(graph):
    nodes = []
    for n in graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                print(f"Found graph attribute for {n.op_type} - {n.name}\n"
                      f"Attribute name: {a.name}")
                nodes += _print_nodes(a.g)
        nodes.append(n.op_type)
    return nodes

def print_nodes(model_proto):
    nodes = _print_nodes(model_proto.graph)
    num_unique_nodes = len(list(set(nodes)))
    num_nodes_total = len(list(nodes))
    all_node_names = list(set(nodes))

    print(f"All node names: {all_node_names}\n"
          f"Unique operations: {num_unique_nodes}\n"
          f"Total Operations: {num_nodes_total}")

def convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath,
                        convert_data_format=False):
    f = io.BytesIO()
    mode = torch.onnx.TrainingMode.TRAINING if training_mode else torch.onnx.TrainingMode.EVAL
    if 'mask_rcnn' not in model_name:
        torch.onnx.export(model,  # model being run
                          input_var,  # model input (or a tuple for multiple inputs)
                          f,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          keep_initializers_as_inputs=True,
                          training=mode,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],
                          opset_version=12)
    else:
        model.eval()
        # input_var = [(input_var,)]
        if isinstance(input_var[0][-1], dict):
            input_var = input_var[0] + ({},)
        else:
            input_var = input_var[0]

        dynamic_axes = {"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                        "scores": [0], "masks": [0, 1, 2]}
        torch.onnx.export(model,  # model being run
                          input_var,  # model input (or a tuple for multiple inputs)
                          f,  # where to save the model (can be a file or file-like object)
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          # training=mode,
                          input_names=["images_tensors"],
                          output_names=["boxes", "labels", "scores", "masks"],
                          dynamic_axes=dynamic_axes,
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          # export_params=True,  # store the trained parameter weights inside the model file
                          # keep_initializers_as_inputs=True,
                          # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
                          )
        print(type(f.getvalue()))
    model_proto = onnx.ModelProto.FromString(f.getvalue())
    print_nodes(model_proto)
    onnx.checker.check_model(model_proto)
    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    filepath = f"{CWD}/{model_name}.onnx"
    if optimize_model:
        model_proto, check = simplify(model_proto)
        assert check
    model_proto = update_node_names(model_proto)
    model_proto = update_edge_names(model_proto)
    with open(filepath, "wb") as f:
        f.write(model_proto.SerializeToString())

    if to_polymath:
        graph = pm.from_onnx(filepath)
        pm.pb_store(graph, f"{CWD}/full_dnns/")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fix_original_onnx_model(batch_size):
    from pathlib import Path
    CWD = Path(f"{__file__}").parent

    input_path = f"{CWD}/full_dnns/mask_rcnn_zoo_original.onnx"
    output_path = f"{CWD}/full_dnns/mask_rcnn_zoo_original_updated.onnx"

    model_proto = onnx.load(input_path)
    new_start_idx = -1
    target_idx = -1
    for idx, n in enumerate(model_proto.graph.node):
        if n.name == '0':
            assert n.op_type == 'Unsqueeze'
            target_idx = idx
        elif n.name == '2':
            new_start_idx = idx
        elif new_start_idx != -1 and target_idx != -1:
            break

    assert target_idx != -1 and new_start_idx != -1
    target_shape = (batch_size, 3, 800, 800)
    dummy_tensor = onnx.helper.make_tensor_value_info("dummy", 1, target_shape)
    model_proto.graph.input[0].type.tensor_type.shape.CopyFrom(dummy_tensor.type.tensor_type.shape)
    model_proto.graph.node[new_start_idx].input[0] = model_proto.graph.input[0].name
    del model_proto.graph.node[target_idx]

    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    onnx.checker.check_model(output_path)

def update_node_names(model_proto):
    non_digit_nodes = []
    for n in model_proto.graph.node:
        if not n.name.isdigit():
            non_digit_nodes.append(n.name)
    for n in model_proto.graph.node:
        if n.name.isdigit():
            new_name = f"{n.op_type}{n.name}"
            assert new_name not in non_digit_nodes
            n.name = new_name
    return model_proto

def update_edge_names(model_proto):
    node_name_map = {}
    INPUT_NAMES = ['A', 'B', 'D', 'X', 'W']
    OUTPUT_NAMES = ['Y', 'Z', 'C', 'H', 'P']

    for n in model_proto.graph.node:
        for idx, i in enumerate(n.input):
            if i not in node_name_map:
                if i.isdigit():
                    assert idx < len(INPUT_NAMES)
                    new_name = f"{n.name.lower()}_{i}{INPUT_NAMES[idx]}"
                else:
                    new_name = i
                node_name_map[i] = new_name

        for idx, o in enumerate(n.output):
            if o not in node_name_map:
                if o.isdigit():
                    assert idx < len(OUTPUT_NAMES)
                    new_name = f"{n.name.lower()}_{o}{OUTPUT_NAMES[idx]}"
                else:
                    new_name = o
                node_name_map[o] = new_name

    for v in model_proto.graph.value_info:
        assert v.name in node_name_map
        v.name = node_name_map[v.name]

    for i in model_proto.graph.initializer:
        assert i.name in node_name_map
        i.name = node_name_map[i.name]

    for n in model_proto.graph.node:
        n.input[:] = [node_name_map[i] for i in n.input]
        n.output[:] = [node_name_map[o] for o in n.output]

    for i in model_proto.graph.input:
        i.name = node_name_map[i.name]

    for o in model_proto.graph.output:
        o.name = node_name_map[o.name]

    return model_proto

def simplify_mrcnn_zoo(batch_size=1):
    from pathlib import Path
    CWD = Path(f"{__file__}").parent
    initial_path = f"{CWD}/full_dnns/mask_rcnn_zoo_original_updated.onnx"
    filepath = f"{CWD}/full_dnns/mask_rcnn_zoo_test.onnx"
    model_proto = onnx.load(initial_path)
    model_proto = update_node_names(model_proto)
    model_proto = update_edge_names(model_proto)
    # onnx.checker.check_model(model_proto)
    # model_proto, check = simplify(model_proto)
    # assert check
    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    print_nodes(model_proto)
    # #
    with open(filepath, "wb") as f:
        f.write(model_proto.SerializeToString())

def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    return name, shape, shape_name

def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)

def parse_array(tensor_proto):
    np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
    return np_array

def collect_value_info(graph):
    node_info = {}
    for vi in graph.value_info:
        name, shape, shape_name = get_info(vi)
        node_info[name] = shape

    for init in graph.initializer:
        node_info[init.name] = parse_array(init).shape

    for inp in graph.input:
        name, shape, shape_name = get_info(inp)
        node_info[name] = shape

    for outp in graph.output:
        name, shape, shape_name = get_info(outp)
        node_info[name] = shape

    return node_info

def get_backbone_outputs(graph, node_info, node_output_map):
    num_max_pool = 0
    backbone_output_shape = [0, 256, 0, 0]
    backbone_kernel_shape = [256, 256, 3, 3]
    output_names = []
    for n in graph.node:
        if n.op_type == "MaxPool":
            if num_max_pool == 1:
                output_names.append(n.output[0])
                output_names.append(n.input[0])
            else:
                num_max_pool += 1
        elif n.op_type == "Conv" and node_output_map[n.input[0]][1] == 'Add':
            assert n.output[0] in node_info
            output_names.append(n.output[0])
    assert len(output_names) == 5
    return output_names

def get_rpn_outputs(graph, node_info, node_output_map):
    rpn_output_shape = [0, 4]
    output_names = []
    for n in graph.node:
        if n.op_type == "Gather":

            if n.input[0] in node_output_map and node_output_map[n.input[0]][1] == "Concat" and \
                    n.input[1] in node_output_map and node_output_map[n.input[1]][1] == "TopK" and \
                    node_info[n.output[0]] == rpn_output_shape:
                output_names = [n.output[0]]
                break
    assert len(output_names) == 1
    return output_names

def split_mrcnn(model_name, split_part):
    from pathlib import Path
    initial_path = f"{CWD}/full_dnns/{model_name}.onnx"
    filepath = f"{CWD}/full_dnns/{model_name}_simplified.onnx"
    node_output_map = {}
    model_proto = onnx.load(initial_path)
    # model_proto, _ = simplify(model_proto)

    # add_value_info_for_constants(model_proto)
    # model_proto = onnx.shape_inference.infer_shapes(model_proto)
    node_info = collect_value_info(model_proto.graph)
    for n in model_proto.graph.node:
        for o in n.output:
            node_output_map[o] = (n.name, n.op_type)
    node_output_map[model_proto.graph.input[0].name] = (None, None)

    if split_part == "backbone":
        input_path = initial_path
        output_path = f"{CWD}/full_dnns/{model_name}_{split_part}.onnx"
        input_names = ['image']
        output_names = get_backbone_outputs(model_proto.graph, node_info, node_output_map)
        extract_model(input_path, output_path, input_names, output_names)
    elif split_part == "rpn":
        input_path = initial_path
        output_path = f"{CWD}/full_dnns/{model_name}_{split_part}.onnx"
        input_names = ['image']
        output_names = get_rpn_outputs(model_proto.graph, node_info, node_output_map)
        input_names += get_backbone_outputs(model_proto.graph, node_info, node_output_map)
        extract_model(input_path, output_path, input_names, output_names)



def main():
    n = 1
    ic = 4
    oc = 8
    h = 33
    w = 33
    ksize = 3
    stride = 2
    pad = 0
    assert (w + 2 * pad - ksize) % stride == 0, 'width does not work'
    # input_shape = (n, ic, h, w)
    # input_var = torch.randn(*input_shape)
    # l = torch.nn.Conv2d(ic, oc, ksize, stride=stride, padding=pad)
    # out = l(input_var)
    M = 128
    N = 1024
    P = 2048
    size_limit1 = 518750
    size_limit = 4190000
    total_size = (M*N) + (N*P) + (M*P)*4 + P*4
    assert total_size <= size_limit, f"Total size {total_size} is greater than limit {size_limit}"
    # optimize_model, training_mode, convert_data_format, to_polymath, input_shape, oc, ksize, stride, pad
    # create_custom_conv(True, True, False, False, input_shape, oc, ksize, stride, pad)
    create_custom_matmul(True, True, False, False, M, N, P, include_bias=True)
    # print(out.shape)
    # output = F.conv2d(input_var)
    # oc, ksize, stride, pad
    # create_lenet_bn(True, True, False, False)
    # create_lenet(True, True, False, False)
    # benchmark = "mask_rcnn_zoo_original_updated_simplified"
    # training_mode = False
    # data_format_convert = False
    # to_polymath = False
    # optimize_model = True
    # batch_size = 1
    # split_part = "backbone"
    # # split_mrcnn(benchmark, split_part)
    # create_maskrcnn_part(split_part, optimize_model, training_mode, data_format_convert, to_polymath)

if __name__ == "__main__":
    main()
    # argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    # argparser.add_argument('-b', '--benchmark', required=True,
    #                        help='Name of the benchmark to create. One of "resnet18", "lenet')
    #
    # argparser.add_argument('-o', '--optimize_model', type=str2bool, nargs='?', default=True,
    #                        const=True, help='Optimize the model')
    #
    # argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model is in training mode')
    #
    # argparser.add_argument('-bs', '--batch_size', type=int, default=1, help='The batch size for the model')
    #
    # argparser.add_argument('-df', '--data_format_convert', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model is in training mode')
    #
    #
    # argparser.add_argument('-pm', '--to_polymath', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model should be converted to PolyMath')
    # args = argparser.parse_args()
    # if args.benchmark == "lenet":
    #     create_lenet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
    # elif args.benchmark == "lenetbn":
    #     create_lenet_bn(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
    # elif args.benchmark == "resnet18":
    #     create_resnet18(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "resnet50":
    #     create_resnet50(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "vgg16":
    #     create_vgg16(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "efficientnet":
    #     create_efficientnet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "alexnet":
    #     create_alexnet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "inception":
    #     create_inception(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "mobilenet":
    #     create_mobilenet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "maskrcnn":
    #     create_maskrcnn(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
    #                     batch_size=args.batch_size)
    # elif args.benchmark == "maskrcnn_simplify":
    #     simplify_mrcnn_zoo(batch_size=args.batch_size)
    # else:
    #     raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
    #                        f"\"lenet\", \"resnet18\".")
#