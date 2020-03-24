# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Deploy Pretrained ResNet Model from MxNet on VTA
================================================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This tutorial provides an end-to-end demo, on how to run ResNet-18 inference
onto the VTA accelerator design to perform ImageNet classification tasks.
It showcases Relay as a front end compiler that can perform quantization (VTA
only supports int8/32 inference) as well as graph_name packing (in order to enable
tensorization in the core) to massage the compute graph_name for the hardware target.
"""
from __future__ import absolute_import, print_function

from tvm.relay.testing import layers, init
from tvm.relay import testing

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user mxnet requests pillow
#
# Now return to the python code. Import packages.


import argparse, json, os, requests, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image
from ctypes import *
from mxnet.gluon.model_zoo import vision
import numpy as np
import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download

from tvm.relay.testing.darknet import __darknetffi__
from tvm.contrib.debugger import debug_runtime

import vta as vta
from vta.testing import simulator
from vta.top import graph_pack

def lenet(num_classes=10, data_shape=(1, 1, 32, 32),
               dtype='float32', alpha=1.0, is_shallow=False):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1 = layers.conv2d(data=data,channels=6, kernel_size=(5,5), name='conv1')
    conv1 = relay.nn.relu(conv1)
    pool2 = relay.nn.avg_pool2d(conv1,pool_size=(2,2), strides=(2,2))
    conv3 = layers.conv2d(data=pool2, channels=16, kernel_size=(5,5), name='conv3')
    conv3 = relay.nn.relu(conv3)
    pool4 = relay.nn.avg_pool2d(conv3,pool_size=(2,2), strides=(2,2))
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
    return init.create_workload(fn)

def mobilenet():
    with tvm.autotvm.tophub.context("llvm"):
        dtype_dict = {"data": 'float32'}
        shape_dict = {"data": (1,3,224,224)}
        model = vision.get_model("mobilenet0.25", pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
    print(mod)
    return mod,params

def resnet():

    return testing.resnet.get_workload(batch_size=env.BATCH, num_layers=18)

def test_resnet_mxnet(env):

    with tvm.autotvm.tophub.context("llvm"):
        dtype_dict = {"data": 'float32'}
        shape_dict = {"data": (env.BATCH,3,224,224)}
        model = vision.get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)

    return mod, params

def test_yolo_darknet():
    MODEL_NAME = "yolov2-tiny"
    CFG_NAME = MODEL_NAME + '.cfg'
    WEIGHTS_NAME = MODEL_NAME + '.weights'
    REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
    CFG_URL = 'https://github.com/pjreddie/darknet/raw/master/cfg/' + CFG_NAME + '?raw=true'
    WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

    cfg_path = download.download_testdata(CFG_URL, CFG_NAME, module="darknet")
    weights_path = download.download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")

    # Download and Load darknet library
    DARKNET_LIB = 'libdarknet2.0.so'

    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'


    lib_path = "/Users/seankinzer/.tvm_test_data/darknet/libdarknet2.0.so"
    print(lib_path)
    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)

    mod, params =  relay.frontend.from_darknet(net, dtype='float32', shape=(1,3,416,416))
    return mod, params

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
    final = relay.op.add(bias22, relay.const(1.0))


    fn = relay.Function(relay.analysis.free_vars(final), final)
    return init.create_workload(fn)

def main(model, start_pack, stop_pack, data_shape=(1, 3, 224, 224), dtype='float32'):
    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")


    ######################################################################
    # Define the platform and model targets
    # -------------------------------------
    # Execute on CPU vs. VTA, and define the model.

    # Load VTA parameters from the vta/config/vta_config.json file
    env = vta.get_env()

    # Set ``device=arm_cpu`` to run inference on the CPU
    # or ``device=vta`` to run inference on the FPGA.
    device = "vta"
    target = env.target if device == "vta" else env.target_vta_cpu

    # Name of Gluon model to compile
    # The ``start_pack`` and ``stop_pack`` labels indicate where
    # to start and end the graph_name packing relay pass: in other words
    # where to start and finish offloading to VTA.

    ######################################################################
    # Obtain an execution remote
    # ---------------------------------
    # When target is 'pynq', reconfigure FPGA and runtime.
    # Otherwise, if target is 'sim', execute locally.
    print(f"Target is {env.TARGET}")
    if env.TARGET in ["sim", "tsim"]:
        remote = rpc.LocalSession()
    else:
        print(f"Error, incorrect target for benchmarking: {env.TARGET}")

    # Get execution context from remote
    ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

    ######################################################################
    # Build the inference graph_name runtime
    # ---------------------------------
    # Grab ResNet-18 model from Gluon model zoo and compile with Relay.
    # The compilation steps are:
    #    1) Front end translation from MxNet into Relay module.
    #    2) Apply 8-bit quantization: here we skip the first conv layer,
    #       and dense layer which will both be executed in fp32 on the CPU.
    #    3) Perform graph_name packing to alter the data layout for tensorization.
    #    4) Perform constant folding to reduce number of operators (e.g. eliminate
    #       batch norm multiply).
    #    5) Perform relay build to object file.
    #    6) Load the object file onto remote (FPGA device).
    #    7) Generate graph_name runtime, `m`.

    # Load pre-configured AutoTVM schedules
    with autotvm.tophub.context(target):

        # Populate the shape and data type dictionary for ResNet input
        dtype_dict = {"data": 'float32'}
        shape_dict = {"data": data_shape}

        # Measure build start time
        build_start = time.time()

        # Start front end compilation
        if model == 'resnet':
            mod, params = test_resnet_mxnet(env)
        elif model == 'yolo':
            mod, params = test_yolo_darknet()
        elif model == 'lenet':
            mod, params = lenet()
        elif model == 'mobilenet':
            mod, params = mobilenet()
        else:
            print(f"Error, incorrect model name: {model}")

        ### Need to bind params




        # Update shape and type dictionary
        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            relay_prog = relay.quantize.quantize(mod['main'], params=params)

        print(f"Finishing quantizing graph_name")
        # Perform graph_name packing and constant folding for VTA target
        if target.device_name == "vta":
            assert env.BLOCK_IN == env.BLOCK_OUT
            relay_prog = graph_pack(
                relay_prog,
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=start_pack,
                stop_name=stop_pack)

        print(f"Finishing packing graph_name")


        # Compile Relay program with AlterOpLayout disabled
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            if target.device_name != "vta":
                graph, lib, params = relay.build(
                    relay_prog, target=target,
                    params=params, target_host=env.target_host)
            else:
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_prog, target=target,
                        params=params, target_host=env.target_host)

        # Measure Relay build time
        build_time = time.time() - build_start
        print(model + " inference graph_name built in {0:.2f}s!".format(build_time))

        # Send the inference library over to the remote RPC server
        temp = util.tempdir()
        lib.save(temp.relpath("graphlib.o"))
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # Graph runtime
        m = graph_runtime.create(graph, lib, ctx)
    #
    # # Set the network parameters and inputs
    data = np.random.uniform(size=data_shape).astype(dtype)

    m.set_input(**params)
    m.set_input('data', tvm.nd.array(data.astype(dtype)))

    # Perform inference and gather execution statistics
    # More on: https://docs.tvm.ai/api/python/module.html#tvm.module.Module.time_evaluator
    num = 1 # number of times we run module for a single measurement
    rep = 1 # number of measurements (we derive std dev from this)
    timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

    if env.TARGET in ["sim", "tsim"]:
        simulator.clear_stats()
        timer()
        sim_stats = simulator.stats()
        print("\nExecution statistics:")
        for k, v in sim_stats.items():
            # Since we execute the workload many times, we need to normalize stats
            # Note that there is always one warm up run
            # Therefore we divide the overall stats by (num * rep + 1)
            print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
    else:
        tcost = timer()
        std = np.std(tcost.results) * 1000
        mean = tcost.mean * 1000
        print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
        print("Average per sample inference time: %.2fms" % (mean/env.BATCH))



if __name__ == '__main__':
    model_name = "resnet"
    if model_name == "yolo":
        start_pack = "nn.conv2d"
        stop_pack = "split"
        data_shape = (1, 3, 416, 416)
    elif model_name == "lenet":
        start_pack = "nn.conv2d"
        stop_pack = "nn.softmax"
        data_shape=(1, 1, 32, 32)
    elif model_name == "resnet":
        start_pack = "nn.max_pool2d"
        stop_pack = "nn.global_avg_pool2d"
        data_shape = (1, 3, 224, 224)
    elif model_name == "mobilenet":
        start_pack = "nn.conv2d"
        stop_pack = "nn.global_avg_pool2d"
        data_shape = (1, 3, 224, 224)

    main(model_name, start_pack, stop_pack, data_shape)



# 29953533
# 6 - 19461432
