import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np


def benchmark_execution(mod,
                        params,
                        measure=False,
                        data_shape=(1, 1, 32, 32),
                        out_shape=(1, 10),
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
            prof_res = np.array(ftimer().results) * 1000
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))

        return out.asnumpy()

    def get_tvm_vm_output(mod, data, params, target, ctx, dtype='float32'):
        ex = relay.create_executor('vm', mod=mod, ctx=ctx)
        result = ex.evaluate()(data, **params)
        return result.asnumpy().astype(dtype)

    # random input
    data = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    ctx = tvm.cpu(0)

    tvm_out = get_tvm_output(mod, tvm.nd.array(data.astype(dtype)), params,
                             target, ctx, dtype)
    vm_out = get_tvm_vm_output(mod, tvm.nd.array(data.astype(dtype)), params,
                               target, ctx, dtype)
    tvm.testing.assert_allclose(vm_out, tvm_out, rtol=1e-5, atol=1e-5)