from tvm.contrib import graph_runtime
from tvm import relay
import tvm
import numpy as np
from tvm.relay.testing import init

def benchmark_execution(mod,
                        params,
                        measure=True,
                        data_shape=(1, 3, 416, 416),
                        out_shape=(1, 125, 14, 14),
                        dtype='float32'):
    def get_tvm_output(mod, data, params, target, ctx, dtype='float32'):
        with relay.build_config(opt_level=1):

            graph, lib, params = relay.build(mod, target, params=params)

        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", data)
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))

        if measure:
            print("Evaluate graph_name runtime inference time cost...")
            ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=20)
            # Measure in millisecond.
            prof_res = np.array(ftimer().results) *1000
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))

        return out.asnumpy()

    # random input
    data = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    ctx = tvm.cpu(0)

    tvm_out = get_tvm_output(mod, tvm.nd.array(data.astype(dtype)), params,
                             target, ctx, dtype)

def execute_graph(net, print_mod=True, print_params=True, benchmark=True):

    mod, params = init.create_workload(net)

    if print_mod:
        print(f"Module: {mod}")

    if print_params:
        for p in params.keys():
            print(f"Key: {p}, shape: {params[p].shape}")

    if benchmark:
        # benchmark_execution(mod, params, data_shape=(1, 3, 416, 416), out_shape=(1, 125, 14, 14))
        # benchmark_execution(mod, params, data_shape=(1, 3, 416, 416), out_shape=(1, 125, 14, 14))
        benchmark_execution(mod, params, data_shape=(1, 3, 224, 224), out_shape=(1,1000))
