import pickle
import random
import threading

import pytest
import polymath as pm
import numpy as np


def numpy_helper(vals):
    w1 = vals["w1"]
    w2 = vals["w2"]
    x1 = vals["x1"].reshape(-1, vals["k"])
    x2 = vals["x2"].reshape(-1, vals["k"])
    r1 = vals["r1"].reshape(-1, vals["m"])
    y1 = vals["y1"].reshape(-1, vals["m"])
    r2 = vals["r2"].reshape(-1, vals["n"])
    y2 = vals["y2"].reshape(-1, vals["n"])
    test_out = np.empty(shape=w1.shape, dtype=w1.dtype)
    indices = []
    for i in range(test_out.shape[0]):
        for j in range(test_out.shape[1]):
            idx1 = (i,j)
            idx2 = tuple([j])
            both = tuple([idx1,idx2])
            indices.append(both)
            test_out[i][j] = w1[i][j]*x2[0][j]
    res = np.empty(shape=3, dtype=test_out.dtype)
    for i in range(3):
        res[i] = np.sum(test_out[i])
    bounds = tuple([[i for i in range(vals["k"])]])

    sum_bound = np.meshgrid(*bounds)

    h1 = np.sum(w1*x2, axis=-1)*r1
    h2 = np.sum(w2*x1, axis=-1)*r2
    d1 = h1 - y1
    d2 = h2 - y2

    g1 = d1.T * vals["x2"]
    g2 = d2.T * vals["x1"]
    w1 = vals["w1"] - g1
    w2 = vals["w2"] - g2
    return w1, w2

def test_multi_shapes():
    m_ = 5
    n_ = 4
    p_ = 3
    inp_ = np.random.randint(1, 5, (m_,p_))
    w_ = np.random.randint(1, 5, (p_,n_))
    mapping = {"m": m_, "n": n_, "p": p_, "in": inp_, "w": w_}

    numpy_res1 = np.empty(shape=(m_,p_,n_))
    indices = []
    for i in range(m_):
        for k in range(p_):
            for j in range(n_):
                numpy_res1[i][k][j] = inp_[i][k]*w_[k][j]
                indices.append(tuple([i,k,j]))
    numpy_res = np.sum(numpy_res1)

    with pm.Node(name="mmul") as graph:
        m = pm.placeholder("m")
        n = pm.placeholder("n")
        p = pm.placeholder("p")
        inp = pm.placeholder("in", shape=(m,p))
        wts = pm.placeholder("w", shape=(p,n))
        i = pm.index(0, m - 1, name="i")
        j = pm.index(0, n - 1, name="j")
        k = pm.index(0, p - 1, name="k")
        inp_ik = pm.var_index(inp, [i, k], name="in[i,k]")
        w_kj = pm.var_index(wts, [k, j], name="w[k,j]")
        slice_mul = (inp_ik*w_kj ).set_name("w[i,k]*in[k,j]")
        out = pm.sum([i, k, j], slice_mul, name="out")
    graph_res = graph("out", mapping)
    assert graph_res == numpy_res

def test_recommender():

    m_ = 3
    n_ = 3
    k_ = 2
    mu_ = 1
    x1_in = np.random.randint(1, 5, (k_))
    x2_in = np.random.randint(1, 5, (k_))

    r1_in = np.random.randint(1, 2, (m_))
    y1_in = np.random.randint(1, 5, (m_))

    r2_in = np.random.randint(1, 2, (n_))
    y2_in = np.random.randint(1, 5,(n_))

    w1_in = np.random.randint(1, 5, (m_,k_))
    w2_in = np.random.randint(1, 5, (n_,k_))
    input_dict = {"mu": mu_,
                   "n": n_,
                   "m": m_,
                   "k": k_,
                   "x1": x1_in,
                   "x2": x2_in,
                   "r1": r1_in,
                   "y1": y1_in,
                   "r2": r2_in,
                   "y2": y2_in,
                   "w1": w1_in,
                   "w2": w2_in
                   }
    with pm.Node(name="recommender") as graph:
        mu = pm.placeholder("mu")
        m = pm.placeholder("m")
        n = pm.placeholder("n")
        k = pm.placeholder("k")
        x1 = pm.placeholder("x1", shape=k)
        x2 = pm.placeholder("x2", shape=k)

        r1 = pm.placeholder("r1", shape=m)
        y1 = pm.placeholder("y1", shape=m)

        r2 = pm.placeholder("r2", shape=n)
        y2 = pm.placeholder("y2", shape=n)

        w1 = pm.placeholder("w1", shape=(m, k))
        w2 = pm.placeholder("w2", shape=(n, k))
        i = pm.index(0, m-1, name="i")
        j = pm.index(0, n-1, name="j")
        l = pm.index(0, k-1, name="l")
        h1_sum = pm.sum([l], (w1[i, l] * x2[l]).set_name("w1*x2")).set_name("h1_sum")
        h1 = (h1_sum[i] * r1[i]).set_name("h1")
        h2_sum = pm.sum([l], (x1[l] * w2[j, l]).set_name("x1*w2")).set_name("h2_sum")
        h2 = (h2_sum[j] * r2[j]).set_name("h2")
        #
        d1 = (h1[i] - y1[i]).set_name("d1")
        d2 = (h2[j] - y2[j]).set_name("d2")
        g1 = (d1[i] * x2[l]).set_name("g1")
        g2 = (d2[j] * x1[l]).set_name("g2")
        w1_ = (w1[i,l] - g1[i,l]).set_name("w1_")
        w2_ = (w2[i,l] - g2[i,l]).set_name("w2_")

    np_res = numpy_helper(input_dict)
    tout = graph(["w1_","w2_"], input_dict)
    np.testing.assert_allclose(tout[0], np_res[0])
    np.testing.assert_allclose(tout[1], np_res[1])


def test_consistent_context():
    with pm.Node() as graph:
        uniform = pm.func_op(random.uniform, 0, 1)
        scaled = uniform * 4
    _uniform, _scaled = graph([uniform, scaled])
    assert _scaled == 4 * _uniform


def test_context():
    with pm.Node() as graph:
        a = pm.placeholder(name='a')
        b = pm.placeholder(name='b')
        c = pm.placeholder(name='c')
        x = a * b + c
    actual = graph(x, {a: 4, 'b': 7}, c=9)
    assert actual == 37


def test_iter():
    with pm.Node(name="outer") as graph:
        pm.variable('abc', name='alphabet', shape=3)
    a, b, c = graph['alphabet']
    assert graph([a, b, c]) == tuple('abc')


def test_getattr():
    with pm.Node() as graph:
        imag = pm.parameter(default=1 + 4j).imag
    assert graph(imag) == 4


class MatmulDummy:
    """
    Dummy implementing matrix multiplication (https://www.python.org/dev/peps/pep-0465/) so we don't
    have to depend on numpy for the tests.
    """
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, pm.Node):
            return NotImplemented
        return self.value * other


@pytest.fixture(params=[
    ('+', 1, 2),
    ('-', 3, 7.0),
    ('*', 2, 7),
    ('@', MatmulDummy(3), 4),
    ('/', 3, 2),
    ('//', 8, 3),
    ('%', 8, 5),
    ('&', 0xff, 0xe4),
    ('|', 0x01, 0xf0),
    ('^', 0xff, 0xe3),
    ('**', 2, 3),
    ('<<', 1, 3),
    ('>>', 2, 1),
    ('!=', 3, 7),
    ('>', 4, 8),
    ('>=', 9, 2),
    ('<', 7, 1),
    ('<=', 8, 7),
])
def binary_operators(request):
    operator, a, b = request.param
    expected = expected = eval('a %s b' % operator)
    return operator, a, b, expected


def test_binary_operators_left(binary_operators):
    operator, a, b, expected = binary_operators
    with pm.Node() as graph:
        _a = pm.parameter(default=a)
        operation = eval('_a %s b' % operator)

    actual = graph(operation)
    assert actual == expected, "expected %s %s %s == %s but got %s" % \
        (a, operator, b, expected, actual)


def test_binary_operators_right(binary_operators):
    operator, a, b, expected = binary_operators
    with pm.Node() as graph:
        _b = pm.parameter(default=b)
        operation = eval('a %s _b' % operator)

    actual = graph(operation)
    assert actual == expected, "expected %s %s %s == %s but got %s" % \
        (a, operator, b, expected, actual)


@pytest.mark.parametrize('operator, value', [
    ('~', True),
    ('~', False),
    ('-', 3),
    ('+', 5),
])
def test_unary_operators(value, operator):
    expected = eval('%s value' % operator)
    with pm.Node() as graph:
        operation = eval('%s pm.parameter(default=value)' % operator)
    actual = graph(operation)
    assert actual == expected, "expected %s %s = %s but got %s" % \
        (operator, value, expected, actual)


def test_contains():
    with pm.Node() as graph:
        test = pm.placeholder()
        alphabet = pm.variable('abc')
        contains = pm.contains(alphabet, test)

    assert graph(contains, {test: 'a'})
    assert not graph(contains, {test: 'x'})


def test_abs():
    with pm.Node() as graph:
        absolute = abs(pm.parameter(default=-5))

    assert graph(absolute) == 5


def test_reversed():
    with pm.Node() as graph:
        rev = reversed(pm.variable('abc'))

    assert list(graph(rev)) == list('cba')


def test_name_change():
    with pm.Node() as graph:
        operation = pm.parameter(default=None, name='operation1')
        pm.parameter(default=None, name='operation3')

    assert 'operation1' in graph.nodes
    operation.name = 'operation2'
    assert 'operation2' in graph.nodes
    assert graph['operation2'] is operation
    # We cannot rename to an existing operation
    with pytest.raises(ValueError):
        operation.name = 'operation3'

def test_conditional():
    with pm.Node() as graph:
        x = pm.parameter(default=4)
        y = pm.placeholder(name='y')
        condition = pm.placeholder(name='condition')
        z = pm.predicate(condition, x, y)

    assert graph(z, condition=False, y=5) == 5
    # We expect a value error if we evaluate the other branch without a placeholder
    with pytest.raises(ValueError):
        print(graph(z, condition=False))


def test_conditional_with_length():
    def f(a):
        return a, a

    with pm.Node() as graph:
        x = pm.parameter(default=4)
        y = pm.placeholder(name='y')
        condition = pm.placeholder(name='condition')

        z1, z2 = pm.predicate(condition, pm.func_op(f, x).set_name("xfunc"), pm.func_op(f, y).set_name("yfunc"), shape=2, name="predfunc")

    assert graph([z1, z2], condition=True) == (4, 4)
    assert graph([z1, z2], condition=False, y=5) == (5, 5)


@pytest.mark.parametrize('message', [None, "x should be smaller than %d but got %d"])
def test_assert_with_dependencies(message):
    with pm.Node() as graph:
        x = pm.placeholder(name='x')
        if message:
            assertion = pm.assert_(x < 10, message, 10, x)
        else:
            assertion = pm.assert_(x < 10)
        with pm.control_dependencies([assertion]):
            y = 2 * x

    assert len(y.dependencies) == 1
    assert graph(y, x=9) == 18
    with pytest.raises(AssertionError) as exc_info:
        graph(y, x=11)

    if message:
        exc_info.match(message % (10, 11))


def test_assert_with_value():
    with pm.Node() as graph:
        x = pm.placeholder(name='x')
        assertion = pm.assert_(x < 10, val=2 * x)

    assert graph(assertion, x=9) == 18
    with pytest.raises(AssertionError):
        graph(assertion, x=11)


@pytest.mark.parametrize('format_string, args, kwargs', [
    ("hello {}", ["world"], {}),
    ("hello {world}", [], {"world": "universe"}),
])
def test_str_format(format_string, args, kwargs):
    with pm.Node() as graph:
        output = pm.str_format(format_string, *args, **kwargs)

    assert graph(output) == format_string.format(*args, **kwargs)


def test_call():
    class Adder:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def compute(self):
            return self.a + self.b

        def __call__(self):
            return self.compute()

    with pm.Node() as graph:
        adder = pm.identity(Adder(3, 7))
        op1 = adder()
        op2 = adder.compute()

    assert graph([op1, op2]) == (10, 10)


def test_lazy_constant():
    import time

    def target():
        time.sleep(1)
        return 12345

    with pm.Node() as graph:
        value = pm.lazy_constant(target)

    start = time.time()
    assert graph(value) == 12345
    assert time.time() - start > 1

    start = time.time()
    assert graph(value) == 12345
    assert time.time() - start < 0.01


def test_lazy_constant_not_callable():
    with pytest.raises(ValueError):
        with pm.Node() as graph:
            pm.lazy_constant(None)


def test_graph_pickle():
    with pm.Node() as graph:
        x = pm.placeholder('x')
        y = pm.pow_(x, 3, name='y')

    _x = random.uniform(0, 1)
    desired = graph('y', x=_x)

    pickled = pickle.dumps(graph)
    graph = pickle.loads(pickled)
    actual = graph('y', x=_x)
    assert desired == actual

def test_import():
    with pm.Node() as graph:
        os_ = pm.import_('os')
        isfile = os_.path.isfile(__file__)

    assert graph(isfile)


def test_tuple():
    expected = 13
    with pm.Node() as graph:
        a = pm.parameter(default=expected)
        b = pm.identity((a, a))
    actual, _ = graph(b)
    assert actual is expected, "expected %s but got %s" % (expected, actual)

def test_np_size():
    with pm.Node() as graph:
        # Only load the libraries when necessary
        imageio = pm.import_('imageio')
        ndimage = pm.import_('scipy.ndimage')
        np = pm.import_('numpy')

        x = np.random.randint(0, 255, 3)
        y = np.random.randint(0, 255, 1)[0]
        w = np.random.randint(0, 255, 3)
        t = (x*w).set_name("t")
    t = graph["t"].nodes

def test_list():
    expected = 13
    with pm.Node() as graph:
        a = pm.parameter(default=expected)
        b = pm.identity([a, a])
    actual, _ = graph(b)
    assert actual is expected, "expected %s but got %s" % (expected, actual)


def test_dict():
    expected = 13
    with pm.Node() as graph:
        a = pm.parameter(default=expected)
        b = pm.identity({'foo': a})
    actual = graph(b)['foo']
    assert actual is expected, "expected %s but got %s" % (expected, actual)


def test_slice():
    with pm.Node() as graph:
        a = pm.variable(range(100))
        b = pm.parameter(default=1)
        c = a[b:]
    assert len(graph(c)) == 99


def test_bool():
    with pm.Node() as graph:
        a = pm.placeholder()

    assert a


@pytest.mark.parametrize('context, expected', [
    ({'a': 1, 'b': 0}, 'zero-division'),
    ({'a': 1, 'b': 2}, 0.5),
])
def test_try(context, expected):
    finally_reached = []

    with pm.Node() as graph:
        a = pm.placeholder('a')
        b = pm.placeholder('b')
        c = pm.try_(
            a / b,
            [(ZeroDivisionError, 'zero-division')],
            pm.func_op(lambda: finally_reached.append('done'))
        )

    assert graph(c, context) == expected
    assert finally_reached


def test_try_not_caught():
    with pm.Node() as graph:
        a = pm.placeholder()
        b = pm.placeholder()
        c = pm.try_(a / b, [(ValueError, 'value-error')])

    with pytest.raises(ZeroDivisionError):
        graph(c, {a: 1, b: 0})


def test_invalid_fetches():
    with pm.Node():
        a = pm.placeholder()
    graph = pm.Node()

    with pytest.raises(RuntimeError):
        graph(a)

    with pytest.raises(KeyError):
        graph('a')

    with pytest.raises(ValueError):
        graph(123)


def test_invalid_context():
    with pytest.raises(ValueError):
        pm.Node().run([], "not-a-mapping")

    with pytest.raises(ValueError):
        pm.Node().run([], {123: None})


def test_duplicate_value():
    with pm.Node() as graph:
        a = pm.placeholder('a')

    with pytest.raises(ValueError):
        graph([], {a: 1}, a=1)

    with pytest.raises(ValueError):
        graph([], {a: 1, 'a': 1})


def test_conditional_callback():
    with pm.Node() as graph:
        a = pm.parameter(default=1)
        b = pm.parameter(default=2)
        c = pm.placeholder()
        d = pm.predicate(c, a, b + 1)

    # Check that we have "traced" the correct number of operation evaluations
    tracer = pm.Profiler()
    assert graph(d, {c: True}, callback=tracer) == 1
    assert len(tracer.times) == 3
    tracer = pm.Profiler()
    assert graph(d, {c: False}, callback=tracer) == 3
    assert len(tracer.times) == 4


def test_try_callback():
    with pm.Node() as graph:
        a = pm.placeholder('a')
        b = pm.assert_((a > 0).set_name('condition'), val=a, name='b')
        c = pm.try_(b, [
            (AssertionError, (pm.identity(41, name='41') + 1).set_name('alternative'))
        ])

    tracer = pm.Profiler()
    graph(c, {a: 3}, callback=tracer) == 3
    assert len(tracer.times) == 4

    graph(c, {a: -2}, callback=tracer) == 42
    assert len(tracer.times) == 6


def test_stack_trace():
    with pm.Node() as graph:
        a = pm.placeholder()
        b = pm.placeholder()
        c = a / b

    try:
        graph(c, {a: 1, b: 0})
        raise RuntimeError("did not raise ZeroDivisionError")
    except ZeroDivisionError as ex:
        assert isinstance(ex.__cause__, pm.EvaluationError)


def test_thread_compatibility():
    def work(event):
        with pm.Node() as graph:
            event.wait()
    event = threading.Event()
    thread = threading.Thread(target=work, args=(event,))
    thread.start()
    try:
        with pm.Node() as graph:
            event.set()
    except AssertionError as e:
        event.set()
        raise e


def test_new():
    test_a = np.array([1,2,3,4])
    test_b = np.array([5,6,7,8])
    test_placeholder = pm.placeholder("hello")
    with pm.Node(name="main") as graph:
        a = pm.parameter(default=6, name="a")
        b = pm.parameter(default=5, name="b")
        a = (a + b).set_name("a_mul_b")
        with pm.Node(name="graph2") as graph2:
            c = pm.variable([[[0, 1,  2], [3,  4,  5], [6,  7,  8]],
                          [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                          [[18, 19, 20], [21, 22, 23], [24, 25, 26]]],
                         name="c")
            c_2 = (c*2).set_name(name="c2")
            e = pm.parameter(default=4, name="e")
            l = pm.placeholder("test")
            x = (l*e).set_name("placeholdermult")
            i = pm.index(0, 1, name="i")
            j = pm.index(0, 1, name="j")
            k = pm.index(0, 2, name="k")
            e_i = pm.var_index(c, [i, j, k], "e_i")

def test_second():
    test_a = np.array([1,2,3,4])
    test_b = np.array([5,6,7,8])
    with pm.Node(name="main") as graph:
        a = pm.parameter(default=6, name="a")
        b = pm.parameter(default=5, name="b")
        a = (a + b).set_name("a_mul_b")
        with pm.Node(name="graph2") as graph2:
            n = pm.placeholder("n")
            b = pm.placeholder("b")
            e = pm.parameter(default=6, name="e")
            l = pm.state("test", shape=(n, b))
            i = pm.index(0, graph2["n"] - 1)
            j = pm.index(0, graph2["b"] - 1)
            lij = pm.var_index(l, [i, j], "lij")

            x = (l*e).set_name("placeholdermult")

        _ = graph2("test", {l: np.arange(16).reshape((-1,4))})
        _ = graph2("lij", {l: np.arange(16).reshape((-1,4))})
        _ = graph2("placeholdermult", {l: np.arange(16).reshape((-1,4))})

def test_linear_embedded():

    with pm.Node(name="linear_reg") as graph:
        m = pm.placeholder("m")
        x = pm.placeholder("x", shape=(m), type_modifier="input")
        y = pm.placeholder("y", type_modifier="input")
        w = pm.placeholder("w", shape=(m), type_modifier="state")
        mu = pm.parameter(name="mu", default=1.0)
        i = pm.index(0, (graph["m"]-1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]).set_name("x*w"), name="h")
        d = (h-y).set_name("h-y")
        g = (d*x).set_name("d*x")
        w_ = (w - (mu*g).set_name("mu*g")).set_name("w-mu*g")
    x = np.random.randint(0, 10, 5)
    y = np.random.randint(0, 10, 1)[0]
    w = np.random.randint(0, 10, 5)

    graph_res = graph("w-mu*g", {"x": x, "y": y, "w": w})
    actual_res = w - ((np.sum(x*w) - y)*x)*1.0
    np.testing.assert_allclose(graph_res, actual_res)

@pytest.mark.parametrize('lbound, ubound, stride', [
    (0, 223, 2),
    (0, 223, 1),
])
def test_strided_index(lbound, ubound, stride):

    with pm.Node(name="strided") as graph:
        idx = pm.index(lbound, ubound-1, stride=stride, name="i")

    ref = np.arange(lbound, ubound, stride)
    res = graph("i", {})

    np.testing.assert_allclose(ref, res)

