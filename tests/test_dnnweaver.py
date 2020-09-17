
# from dnnweaver2.compiler import FPGASpec, GraphCompiler
# from dnnweaver2.simulator.accelerator import Accelerator
# from polymath.codegen.dnnweavergen.dnnweaver_pass import DNNWEAVER_OPS
from pathlib import Path
import polymath as pm
from .util import tiny_yolo

DEFAULT_SRAM = {
        'ibuf': 16*32*512,
        'wbuf': 16*32*32*512,
        'obuf': 64*32*512,
        'bbuf': 16*32*512
    }
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"
ONNX_FILE_DIR = Path(f"{Path(__file__).parent}/onnx_examples")
#
# def test_load_yolo():
#     native_net = tiny_yolo(train=False)
#     filename = f"full_dnns/tiny_yolo.onnx"
#     filepath = f"{BENCH_DIR}/{filename}"
#     dnnw_path = f"{OUTPATH}/tiny_yolo_onnx_dnnw.json"
#
#     assert Path(filepath).exists()
#     graph = pm.from_onnx(filepath)
#
#     _, pm_net = pm.generate_dnnweaver(graph, {},
#                                    dnnw_path, debug=False,
#                                       context_dict={}, add_kwargs=True)
# #