from polymath.srdfg.passes import register_pass, Pass, pass_registry
from polymath import UpdateBatchSize, CollectDNNShapes
import polymath as pm
import numpy as np
from itertools import product
from pathlib import Path

BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
CWD = Path(f"{__file__}").parent

ONNX_DNNS = f"{BENCH_DIR}/full_dnns/"


def test_resnet18_batchsize():
    batch_size = 32
    resnet18_path = f"{ONNX_DNNS}/resnet18.onnx"
    resnet18_graph = pm.from_onnx(resnet18_path)

    batch_size_pass = UpdateBatchSize(batch_size, resnet18_graph.op_name)
    updated_resnet18 = batch_size_pass(resnet18_graph)
    test_op_shape_pass = CollectDNNShapes()
    _ = test_op_shape_pass(updated_resnet18)

    ref_resnet18_path = f"{ONNX_DNNS}/resnet18_batch{batch_size}.onnx"
    #
    ref_resnet18_graph = pm.from_onnx(ref_resnet18_path)

    ref_op_shape_pass = CollectDNNShapes()
    _ = ref_op_shape_pass(ref_resnet18_graph)
    ref_shapes = ref_op_shape_pass.shape_tracker
    test_shapes = test_op_shape_pass.shape_tracker

    assert len(list(ref_shapes.keys())) == len(list(test_shapes.keys())), f"Reference keys: {list(ref_shapes.keys())}\n" \
                                                                          f"Test keys: {list(test_shapes.keys())}"
    for op_name, shapes in ref_shapes.items():
        for idx, s in enumerate(shapes):
            assert isinstance(s, tuple) and s == test_shapes[op_name][idx]






