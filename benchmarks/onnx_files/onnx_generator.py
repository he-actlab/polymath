
import argparse
from onnxsim import simplify
import onnx
import tf2onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnxsim import simplify
from onnx import optimizer
import tensorflow.compat.v1 as tf


def create_backprop(l1, l2, l3, optimize_model):
    batch_size = 1
    model_name = f"backprop{l1}_{l2}_{l3}"
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(batch_size, l1), name='x')
        y = tf.placeholder(tf.float32, shape=(batch_size, l3), name='y')

        w1 = tf.placeholder(tf.float32, shape=(l1, l2), name='W1')
        w2 = tf.placeholder(tf.float32, shape=(l2, l3), name='W2')
        input_shapes = {"x:0": x.shape, "W1:0": w1.shape, "W2:0": w2.shape, "y:0": y.shape}

        mu = tf.constant(1, dtype=tf.float32)
        #     _ = tf.Variable(initial_value=np.random.rand(1))

        a1 = tf.math.sigmoid(tf.matmul(x, w1))
        a2 = tf.math.sigmoid(tf.matmul(a1, w2))

        d3 = a2 - y
        d2 = tf.matmul(d3, tf.transpose(w2)) * (a1 * (mu - a1))
        g1 = tf.transpose(x) * d2
        g2 = tf.transpose(a1) * d3

        w1 = tf.subtract(w1, mu * g1, name='W1_out')
        w2 = tf.subtract(w2, mu * g2, name='W2_out')
        sess.run(tf.initialize_all_variables())

        input_names = ['x:0', 'y:0']
        output_names = ['W1_out:0', 'W2_out:0']

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=input_names, output_names=output_names)
        model_proto = onnx_graph.make_model(model_name)
        onnx.checker.check_model(model_proto)
        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
        if optimize_model:
            model_proto, check = simplify(model_proto, input_shapes=input_shapes)
            assert check
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(w)


def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)


def create_lenet(features, optimize_model):
    RANDOM_SEED = 42
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    N_EPOCHS = 15

    IMG_SIZE = 32
    N_CLASSES = features

    class LeNet5(nn.Module):

        def __init__(self, n_classes, training=False):
            super(LeNet5, self).__init__()
            self.training = training
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                                   kernel_size=5, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                                   kernel_size=5, stride=1, padding=0)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                                   kernel_size=5, stride=1, padding=0)
            self.linear1 = nn.Linear(120, 84)
            self.linear2 = nn.Linear(84, n_classes)
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
            logits = self.linear2(x)

            probs = F.softmax(logits, dim=1)
            if self.training:
                return logits, probs
            else:
                return probs
    fname = "lenet.onnx"
    torch.manual_seed(RANDOM_SEED)
    x = torch.randn(1, 1, 32, 32)
    model = LeNet5(N_CLASSES).to('cpu')
    model.eval()
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fname,  # where to save the model (can be a file or file-like object)
                      #                   export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    onnx_model = onnx.load(fname)
    if optimize_model:
        model, check = simplify(onnx_model)
        assert check
    onnx.save(model, fname)

def create_linear(m, optimize_model):
    batch_size = 1
    model_name = f"linear_{m}"

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(m,), name='x')
        y = tf.placeholder(tf.float32, shape=(1,), name='y')

        w = tf.placeholder(tf.float32, shape=(m,), name='W')
        input_shapes = {"x:0": x.shape, "W:0": w.shape, "y:0": y.shape}
        mu = tf.constant(1, dtype=tf.float32, name="mu")
        h = tf.reduce_sum(tf.multiply(w, x))
        d = tf.subtract(h, y)
        g = tf.multiply(d, x)

        g = tf.multiply(mu, g)
        w = tf.subtract(w, g, name='w_out')

        input_names = ['x:0', 'y:0']
        output_names = ['w_out:0']

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=input_names, output_names=output_names)
        model_proto = onnx_graph.make_model(model_name)
        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
        if optimize_model:
            model_proto, check = simplify(model_proto, input_shapes=input_shapes)
            assert check
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def create_logistic(m, optimize_model):
    batch_size = 1
    model_name = f"logistic{m}"
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(m,), name='x')
        y = tf.placeholder(tf.float32, shape=(1,), name='y')

        w = tf.placeholder(tf.float32, shape=(m,), name='W')
        input_shapes = {"x:0": x.shape, "W:0": w.shape, "y:0": y.shape}

        mu = tf.constant(1, dtype=tf.float32, name="mu")
        h = tf.reduce_sum(tf.multiply(w, x))
        h = tf.math.sigmoid(h)
        d = tf.subtract(h, y)
        g = tf.multiply(d, x)

        g = tf.multiply(mu, g)
        w = tf.subtract(w, g, name='w_out')

        input_names = ['x:0', 'y:0']
        output_names = ['w_out:0']

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=input_names, output_names=output_names)

        model_proto = onnx_graph.make_model(model_name)

        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
        if optimize_model:
            model_proto, check = simplify(model_proto, input_shapes=input_shapes)
            assert check
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def create_svm(m, optimize_model):
    batch_size = 1
    model_name = f"svm{m}"


    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(m,), name='x')
        y = tf.placeholder(tf.float32, shape=(1,), name='y')

        w = tf.placeholder(tf.float32, shape=(m,), name='W')
        input_shapes = {"x:0": x.shape, "W:0": w.shape, "y:0": y.shape}

        mu = tf.constant(1, dtype=tf.float32, name="mu")

        h = tf.reduce_sum(w * x)
        c = y * h
        ny = 0 - y
        p = tf.cast((c > y), dtype=ny.dtype) * ny

        g = p * x

        w = tf.subtract(w, mu * g, name="W_out")
        sess.run(tf.initialize_all_variables())

        input_names = ['x:0', 'y:0']
        output_names = ['W_out:0']

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=input_names, output_names=output_names)
        model_proto = onnx_graph.make_model(model_name)
        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
        if optimize_model:
            model_proto, check = simplify(model_proto, input_shapes=input_shapes)
            assert check
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def create_reco(m, n, k, optimize_model):
    batch_size = 1
    model_name = f"reco{m}_{n}_{k}"


    with tf.Session() as sess:
        x1 = tf.placeholder(tf.float32, shape=(k, 1), name='x1')
        x2 = tf.placeholder(tf.float32, shape=(k, 1), name='x2')

        r1 = tf.placeholder(tf.float32, shape=(m, 1), name='r1')
        y1 = tf.placeholder(tf.float32, shape=(m, 1), name='y1')

        r2 = tf.placeholder(tf.float32, shape=(n, 1), name='r2')
        y2 = tf.placeholder(tf.float32, shape=(n, 1), name='y2')

        w1 = tf.placeholder(tf.float32, shape=(m, k), name='w1')
        w2 = tf.placeholder(tf.float32, shape=(n, k), name='w2')
        input_shapes = {"x1:0": x1.shape, "w1:0": w1.shape, "y1:0": y1.shape, "r1:0": r1.shape,
                        "x2:0": x2.shape, "w2:0": w2.shape, "y2:0": y2.shape, "r2:0": r2.shape,}

        mu = tf.constant(1, dtype=tf.float32, name="mu")
        h1_sum = tf.matmul(w1, x2)
        h1 = h1_sum * r1

        h2_sum = tf.matmul(w2, x1)
        h2 = h2_sum * r2
        d1 = h1 - y1
        d2 = h2 - y2
        g1 = d1 * tf.transpose(x2)
        g2 = d2 * tf.transpose(x1)

        w1_out = tf.subtract(w1, mu * g1, name="w1_out")
        w2_out = tf.subtract(w2, mu * g2, name="w2_out")
        sess.run(tf.initialize_all_variables())

        input_names = ['x1:0', 'y1:0', 'r1:0', 'x2:0', 'y2:0', 'r2:0']
        output_names = ['w1_out:0', 'w2_out:0']

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=input_names, output_names=output_names)
        model_proto = onnx_graph.make_model(model_name)
        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
        if optimize_model:
            model_proto, check = simplify(model_proto, input_shapes=input_shapes)
            assert check

        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

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
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')
    argparser.add_argument('-o', '--optimize_model', type=str2bool, nargs='?', default=True,
                           const=True, help='Optimize the model')
    args = argparser.parse_args()
    features = tuple([int(i) for i in args.feature_size])
    if args.benchmark == "linear":
        create_linear(*features, args.optimize_model)
    elif args.benchmark == "logistic":
        create_logistic(*features, args.optimize_model)
    elif args.benchmark == "reco":
        create_reco(*features, args.optimize_model)
    elif args.benchmark == "svm":
        create_svm(*features, args.optimize_model)
    elif args.benchmark == "backprop":
        create_backprop(*features, args.optimize_model)
    elif args.benchmark == "lenet":
        create_lenet(*features, args.optimize_model)
    else:
        raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
                           f"\"logistic\", \"linear\", \"reco\","
                                "or \"svm\".")