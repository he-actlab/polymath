
import argparse
import tf2onnx

from onnx import optimizer
import tensorflow as tf


def create_backprop(l1, l2, l3):
    batch_size = 1
    model_name = f"backprop{l1}_{l2}_{l3}"
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(batch_size, l1), name='x')
        y = tf.placeholder(tf.float32, shape=(batch_size, l3), name='y')

        w1 = tf.placeholder(tf.float32, shape=(l1, l2), name='W1')
        w2 = tf.placeholder(tf.float32, shape=(l2, l3), name='W2')

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
        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])

        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())


def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(w)


def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)


def create_lenet(features):
    with tf.Session() as sess:

        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        # y = tf.placeholder(tf.int32, (None), name='y')
        # one_hot_y = tf.one_hot(y, 10)
        # name:      conv5-6
        # structure: Input = 32x32x1. Output = 28x28x6.
        # weights:   (5*5*1+1)*6
        # connections: (28*28*5*5+28*28)*6
        conv1_W = init_weight((5, 5, 1, 6))
        conv1_b = init_bias(6)
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        # Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv5-16
        # input 14x14x6 Output = 10x10x16.
        # weights: (5*5*6+1)*16 ---real Lenet-5 is (5*5*3+1)*6+(5*5*4+1)*9+(5*5*6+1)*1
        conv2_W = init_weight((5, 5, 6, 16))
        conv2_b = init_bias(16)
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        # Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Input = 5x5x16. Output = 400.
        fc0 = tf.layers.flatten(conv2)

        # Input = 400. Output = 120.
        fc1_W = init_weight((400, 120))
        fc1_b = init_bias(120)
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

        # Input = 120. Output = 84.
        fc2_W = init_weight((120, 84))
        fc2_b = init_bias(84)
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)

        # Input = 84. Output = 10.
        fc3_W = init_weight((84, 10))
        fc3_b = init_bias(10)
        logits = tf.matmul(fc2, fc3_W) + fc3_b
        cross_entropy = tf.nn.softmax(logits=logits, name="out")
        sess.run(tf.compat.v1.global_variables_initializer())
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=['data'], output_names='out')
        model_proto = onnx_graph.make_model('lenet')
        model_proto = optimizer.optimize(model_proto, ['eliminate_identity'])
        with open(f"./lenet.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())
    return logits

def create_linear(m):
    batch_size = 1
    model_name = f"linear_{m}"

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(m,), name='x')
        y = tf.placeholder(tf.float32, shape=(1,), name='y')

        w = tf.placeholder(tf.float32, shape=(m,), name='W')

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
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def create_logistic(m):
    batch_size = 1
    model_name = f"logistic{m}"
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(m,), name='x')
        y = tf.placeholder(tf.float32, shape=(1,), name='y')

        w = tf.placeholder(tf.float32, shape=(m,), name='W')

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
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def create_svm(m):
    batch_size = 1
    model_name = f"svm{m}"


    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(m,), name='x')
        y = tf.placeholder(tf.float32, shape=(1,), name='y')

        w = tf.placeholder(tf.float32, shape=(m,), name='W')

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
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

def create_reco(m, n, k):
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
        with open(f"./{model_name}.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')
    args = argparser.parse_args()
    features = tuple([int(i) for i in args.feature_size])
    if args.benchmark == "linear":
        create_linear(*features)
    elif args.benchmark == "logistic":
        create_logistic(*features)
    elif args.benchmark == "reco":
        create_reco(*features)
    elif args.benchmark == "svm":
        create_svm(*features)
    elif args.benchmark == "backprop":
        create_backprop(*features)
    elif args.benchmark == "lenet":
        create_lenet(*features)
    else:
        raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
                           f"\"logistic\", \"linear\", \"reco\","
                                "or \"svm\".")