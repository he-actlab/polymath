from tvm import relay
from tvm.relay.testing import layers, init
from mxnet.gluon.model_zoo import vision
from polymath.codegen.tvmgen.test import benchmark
from tvm.relay import testing

import tvm
from tvm.contrib import graph_runtime
import numpy as np
# Mean inference time (std dev): 207.12 ms (22.97 ms)

def benchmark_execution(mod,
                        params,
                        measure=True,
                        data_shape=(1, 3, 224, 224),
                        out_shape=(1, 1000),
                        dtype='float32'):
    def get_tvm_output(mod, data, params, target, ctx, dtype='float32'):
        with relay.build_config(opt_level=0):

            graph, lib, params = relay.build(mod, target, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        print(m)

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

def gen_code(mod,
            params,
            measure=True,
            data_shape=(1, 3, 224, 224),
            out_shape=(1, 1000),
            dtype='float32'):
    data = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    ctx = tvm.cpu(0)
    with relay.build_config(opt_level=1):

        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_runtime.create(graph, lib, ctx)
    m.set_input("data", data)
    m.set_input(**params)
    m.run()
    out = m.get_output(0, tvm.nd.empty(out_shape, dtype))


def lenet(num_classes=10, data_shape=(1, 1, 32, 32),
               dtype='float32', alpha=1.0, is_shallow=False):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1 = layers.conv2d(data=data, channels=6, kernel_size=(5,5), name='conv1')
    conv1 = relay.nn.relu(conv1)
    pool2 = relay.nn.avg_pool2d(conv1, pool_size=(2,2), strides=(2,2))

    conv3 = layers.conv2d(data=pool2, channels=16, kernel_size=(5, 5), name='conv3')
    conv3 = relay.nn.relu(conv3)
    pool4 = relay.nn.avg_pool2d(conv3, pool_size=(2, 2), strides=(2, 2))
    flattened5 = relay.nn.batch_flatten(pool4)

    fcw5 = relay.var('fc5_weight')
    fc5 = relay.nn.dense(data=flattened5, weight=fcw5, units=120)
    fc5 = relay.nn.relu(fc5)

    fcw6 = relay.var('fc6_weight')
    fc6 = relay.nn.dense(data=fc5, weight=fcw6, units=84)
    fc6 = relay.nn.relu(fc6)

    fcw7 = relay.var('fc7_weight')
    fc7= relay.nn.dense(data=fc6, weight=fcw7, units=num_classes)
    fc7 = relay.nn.relu(fc7)

    softmax = relay.nn.softmax(data=fc7)
    fn = relay.Function(relay.analysis.free_vars(softmax), softmax)
    return fn

def test_vgg():
    for n in [11, 16]:
        mod, params = testing.vgg.get_workload(1, num_layers=n)
        benchmark_execution(mod, params)

def test_resnet():

    mod, params = testing.resnet.get_workload(batch_size=1, num_layers=18)


    # for p in params.keys():
    #     print(f"Key: {p}, shape: {params[p].shape}")

    # benchmark_execution(mod, params)

def test_mobilenet():
    mod, params = testing.mobilenet.get_workload(batch_size=1)
    benchmark_execution(mod, params, data_shape=(1, 3, 416, 416), out_shape=(1, 125, 14, 14))

def test_yolo():
    net = yolo()
    mod, params = init.create_workload(net)


    benchmark_execution(mod, params, data_shape=(1, 3, 416, 416), out_shape=(1, 125, 14, 14))

def test_yolo_fuse():
    net = testing.resnet.get_net(num_classes=1000,batch_size=1, num_layers=18, image_shape=(3,224,224))
    new_net = relay.testing.run_opt_pass(net, relay.transform.FuseOps())


def codegen_yolo():
    net = yolo()
    mod, params = init.create_workload(net)


    gen_code(mod, params, data_shape=(1, 3, 416, 416), out_shape=(1, 125, 14, 14))


def yolo_fused():
    inp = relay.var("data", shape=(1, 3, 416, 416))

    conv0 = layers.conv2d(name="conv0", data=inp, channels=16, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    bias0 = relay.nn.bias_add(conv0, relay.var("b0_bias"))
    bn0 = layers.batch_norm_infer(bias0, name="bn0")
    a1 = relay.nn.leaky_relu(bn0, alpha=0.1)
    p2 = relay.nn.max_pool2d(a1, pool_size=(2, 2), strides=(2, 2))

    conv3 = layers.conv2d(name="conv3", data=p2, channels=32, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    bias3 = relay.nn.bias_add(conv3, relay.var("b3_bias"))
    bn3 = layers.batch_norm_infer(bias3, name="bn3")
    a4 = relay.nn.leaky_relu(bn3, alpha=0.1)
    p5 = relay.nn.max_pool2d(a4, pool_size=(2, 2), strides=(2, 2))

    conv6 = layers.conv2d(name="conv6", data=p5, channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    bias6 = relay.nn.bias_add(conv6, relay.var("b6_bias"))
    bn6 = layers.batch_norm_infer(bias6, name="bn6")
    a7 = relay.nn.leaky_relu(bn6, alpha=0.1)
    p8 = relay.nn.max_pool2d(a7, pool_size=(2, 2), strides=(2, 2))

    conv9 = layers.conv2d(name="conv9", data=p8, channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    bias9 = relay.nn.bias_add(conv9, relay.var("b9_bias"))
    bn9 = layers.batch_norm_infer(bias9, name="bn9")
    a10 = relay.nn.leaky_relu(bn9, alpha=0.1)
    p11 = relay.nn.max_pool2d(a10, pool_size=(2, 2), strides=(2, 2))

    conv12 = layers.conv2d(name="conv12", data=p11, channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

    bias12 = relay.nn.bias_add(conv12, relay.var("b12_bias"))
    bn12 = layers.batch_norm_infer(bias12, name="bn12")
    a13 = relay.nn.leaky_relu(bn12, alpha=0.1)
    p14 = relay.nn.max_pool2d(a13, pool_size=(2, 2), strides=(2, 2))

    conv15 = layers.conv2d(name="conv15", data=p14, channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    bias15 = relay.nn.bias_add(conv15, relay.var("b15_bias"))
    bn15 = layers.batch_norm_infer(bias15, name="bn15")
    a16 = relay.nn.leaky_relu(bn15, alpha=0.1)
    p17 = relay.nn.max_pool2d(a16, pool_size=(2, 2), strides=(1, 1), padding=(0, 0))

    conv18 = layers.conv2d(name="conv18", data=p17, channels=1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

    bias18 = relay.nn.bias_add(conv18, relay.var("b18_bias"))
    bn18 = layers.batch_norm_infer(bias18, name="bn18")
    a19 = relay.nn.leaky_relu(bn18, alpha=0.1)

    conv20 = layers.conv2d(name="conv20", data=a19, channels=1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

    bias20 = relay.nn.bias_add(conv20, relay.var("b20_bias"))
    bn20 = layers.batch_norm_infer(bias20, name="bn20")
    a21 = relay.nn.leaky_relu(bn20, alpha=0.1)

    conv22 = layers.conv2d(name="conv22", data=a21, channels=125, kernel_size=(1, 1), strides=(1, 1), padding=(1, 1))

    bias22 = relay.nn.bias_add(conv22, relay.var("b22_bias"))
    final = relay.op.add(bias22, relay.const(1))


    fn = relay.Function(relay.analysis.free_vars(final), final)

    return fn

def yolo():
    inp = relay.var("data",shape=(1,3, 416,416))


    conv0 = layers.conv2d(name="conv0", data=inp, channels=16, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b0 = relay.var("b0_bias")
    bias0 = relay.nn.bias_add(conv0, b0)
    bn0 = layers.batch_norm_infer(bias0, name="bn0")
    a1 = relay.nn.leaky_relu(bn0, alpha=0.1)
    p2 = relay.nn.max_pool2d(a1, pool_size=(2,2), strides=(2,2))

    conv3 = layers.conv2d(name="conv3", data=p2, channels=32, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b3 = relay.var("b3_bias")
    bias3 = relay.nn.bias_add(conv3, b3)
    bn3 = layers.batch_norm_infer(bias3, name="bn3")
    a4 = relay.nn.leaky_relu(bn3, alpha=0.1)
    p5 = relay.nn.max_pool2d(a4, pool_size=(2,2), strides=(2,2))

    conv6 = layers.conv2d(name="conv6", data=p5, channels=64, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b6 = relay.var("b6_bias")
    bias6 = relay.nn.bias_add(conv6, b6)
    bn6 = layers.batch_norm_infer(bias6, name="bn6")
    a7 = relay.nn.leaky_relu(bn6, alpha=0.1)
    p8 = relay.nn.max_pool2d(a7, pool_size=(2,2), strides=(2,2))

    conv9 = layers.conv2d(name="conv9", data=p8, channels=128, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b9 = relay.var("b9_bias")
    bias9 = relay.nn.bias_add(conv9, b9)
    bn9 = layers.batch_norm_infer(bias9, name="bn9")
    a10 = relay.nn.leaky_relu(bn9, alpha=0.1)
    p11 = relay.nn.max_pool2d(a10, pool_size=(2,2), strides=(2,2))

    conv12 = layers.conv2d(name="conv12", data=p11, channels=256, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b12 = relay.var("b12_bias")
    bias12 = relay.nn.bias_add(conv12, b12)
    bn12 = layers.batch_norm_infer(bias12, name="bn12")
    a13 = relay.nn.leaky_relu(bn12, alpha=0.1)
    p14 = relay.nn.max_pool2d(a13, pool_size=(2,2), strides=(2,2))

    conv15 = layers.conv2d(name="conv15", data=p14, channels=512, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b15 = relay.var("b15_bias")
    bias15 = relay.nn.bias_add(conv15, b15)
    bn15 = layers.batch_norm_infer(bias15, name="bn15")
    a16 = relay.nn.leaky_relu(bn15, alpha=0.1)
    p17 = relay.nn.max_pool2d(a16, pool_size=(2,2), strides=(1,1), padding=(0,0))

    conv18 = layers.conv2d(name="conv18", data=p17, channels=1024, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b18 = relay.var("b18_bias")
    bias18 = relay.nn.bias_add(conv18, b18)
    bn18 = layers.batch_norm_infer(bias18, name="bn18")
    a19 = relay.nn.leaky_relu(bn18, alpha=0.1)

    conv20 = layers.conv2d(name="conv20", data=a19, channels=1024, kernel_size=(3,3), strides=(1,1), padding=(1,1))
    b20 = relay.var("b20_bias")
    bias20 = relay.nn.bias_add(conv20, b20)
    bn20 = layers.batch_norm_infer(bias20, name="bn20")
    a21 = relay.nn.leaky_relu(bn20, alpha=0.1)

    conv22 = layers.conv2d(name="conv22", data=a21, channels=125, kernel_size=(1,1), strides=(1,1), padding=(1,1))
    b22 = relay.var("b22_bias")
    bias22 = relay.nn.bias_add(conv22, b22)


    fn = relay.Function(relay.analysis.free_vars(bias22), bias22)

    return fn

def main():
    net = lenet()
    mod, params = init.create_workload(net)
    print(f"Module: {mod}")

    for p in params.keys():
        print(f"Key: {p}, shape: {params[p].shape}")
    # benchmark_execution(mod, params, data_shape=(1, 1, 32, 32),
    #                     out_shape=(1, 10), measure=True)



if __name__ == '__main__':
    import inspect
    # mname = "resnet18_v1"
    # test_yolo()
    main()
    # test_resnet()
    # with tvm.autotvm.tophub.context("llvm"):
    #     dtype_dict = {"data": 'float32'}
    #     shape_dict = {"data": (1,3,224,224)}
    #     model = vision.get_model(mname, pretrained=True)
    #     mod, params = relay.frontend.from_mxnet(model, shape_dict)
    #     print(f"Param keys: {params['resnetv10_batchnorm0_beta']}")




























# Execution statistics:
# 	inp_load_nbytes :          5549568
# 	wgt_load_nbytes :         12763136
# 	acc_load_nbytes :            30720
# 	uop_load_nbytes :             1177
# 	out_store_nbytes:          1680896
# 	gemm_counter    :          6623232
# 	alu_counter     :           572320





# 29953036
# 29951630
# 29953036
# 29956022
# Execution statistics:
# 	inp_load_nbytes :          5549568
# 	wgt_load_nbytes :         12763136
# 	acc_load_nbytes :            30720
# 	uop_load_nbytes :             4004
# 	out_store_nbytes:          1680896
# 	gemm_counter    :          6623232
# 	alu_counter     :           572320




