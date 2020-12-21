import polymath as pm
import numpy as np
from polymath.srdfg.from_onnx.converter import get_value_info_shape
from .util import np_nms, onnx_nms, torch_nms, t_torch_nms
import pytest
from pathlib import Path

BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"
ONNX_FILE_DIR = Path(f"{Path(__file__).parent}/onnx_examples")
def test_lenet():
    pass
# @pytest.mark.parametrize('max_output_per_class, iou_threshold, score_threshold, center_point_box',[
#     (10, 0.5, 0.0, 0)
# ])
# def test_nms(max_output_per_class, iou_threshold, score_threshold, center_point_box):
#     # boxes = np.array([[
#     #     [1.0, 1.0, 0.0, 0.0],
#     #     [0.0, 0.1, 1.0, 1.1],
#     #     [0.0, 0.9, 1.0, -0.1],
#     #     [0.0, 10.0, 1.0, 11.0],
#     #     [1.0, 10.1, 0.0, 11.1],
#     #     [1.0, 101.0, 0.0, 100.0]
#     # ]]).astype(np.float32)
#     # scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
#     boxes = np.array([[
#         [1.0, 1.0, 0.0, 0.0],
#         [0.0, 0.1, 1.0, 1.1],
#         [0.0, 0.9, 1.0, -0.1],
#         [0.0, 10.0, 1.0, 11.0],
#         [1.0, 10.1, 0.0, 11.1],
#         [1.0, 101.0, 0.0, 100.0]
#     ]]).astype(np.float32)
#     scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
#     onnx_res, valid_res = onnx_nms(boxes, scores, max_output_per_class, iou_threshold, score_threshold)
#     # test_res = valid_res[:, 2]
#
#     np_res = np_nms(boxes, scores, max_output_per_class, iou_threshold, score_threshold)
#     torch_nms(boxes, scores, max_output_per_class, iou_threshold, score_threshold)
#     # onnx_nms(boxes, scores, max_output_per_class, iou_threshold, score_threshold)
#
# def test_mrcnn_backbone():
#     filename = f"backbone_mrcnn.onnx"
#     filepath = f"{BENCH_DIR}/full_dnns/mask_rcnn/{filename}"
#     assert Path(filepath).exists()
#     graph = pm.from_onnx(filepath)
#
# def test_mrcnn_ops():
#     # filename = f"mask_rcnn/backbone_mrcnn.onnx"
#     filename = f"lenet.onnx"
#     filepath = f"{BENCH_DIR}/full_dnns/{filename}"
#     assert Path(filepath).exists()
#
#     import onnx
#     model = onnx.load(filepath)
#     onnx.checker.check_model(model)
#     graph = onnx.shape_inference.infer_shapes(model).graph
#
#     val_info = {}
#
#     for v in graph.value_info:
#         # print(v)
#         val_info[v.name] = tuple([dim.dim_value for dim in v.type.tensor_type.shape.dim])
#
#     #     val_info[v.name] = get_value_info_shape(v)
#     # print(val_info)
#     for n in graph.node:
#         if n.op_type == "Pad":
#             print(f"Input shape: {val_info[n.input[0]]}\n"
#                   f"Output shape: {val_info[n.output[0]]}\n")
#             # print(n.input[0])
#             # print(n.output[0])
#             # print(n.name)
#             # print(f"\n")
#
#




