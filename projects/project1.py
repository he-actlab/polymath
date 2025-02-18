import polymath as pm
import numpy as np
import gpflow
from scipy.signal import butter
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_data(N):
    x = np.random.randint(-5, 5, N)
    return x

def np_test(x, w):
    fft_out = np.abs(np.fft.fft(x))
    output = sigmoid(fft_out.dot(w))
    return output

def get_graph():

    with pm.Node("fft") as graph:
        N = pm.parameter("N")
        x = pm.input("x", shape=(N))
        n1 = pm.index(0, N-1, name="n1")
        n2 = pm.index(0, N-1, name="n2")

        X = pm.output("X", shape=(N))

        M = pm.temp("M", shape=(N,N))
        M[n1, n2] = (n1 * n2)
        M[n1, n2] = pm.exp(-2j * np.pi * M[n1,n2]/N)
        X[n1] = pm.abs(pm.sum([n2], M[n1, n2]* x[n2]))
        W = pm.input("W", shape=(N))
        Y = pm.output("Y", shape=(1,))
        pm.logistic_regressor(X, W, Y, 1.0, N)


    return graph

def matern_kernel(xa, xb, lscale=1):
    s_ = np.sum((xa-xb)**2)
    r = np.sqrt(s_)
    out = (1 + np.sqrt(3)*r/lscale)*np.exp(-np.sqrt(3)*r/lscale)
    return out

def matern(xa_, xb_, lscale_):
    with pm.Node("matern32") as graph:
        m = pm.parameter("m")
        lscale = pm.parameter("lscale")
        xa = pm.input("xa", shape=(m))
        xb = pm.input("xb", shape=(m))
        out = pm.output("y", shape=(1,))
        i = pm.index(0, m-1, name="i")
        r = pm.sqrt(pm.sum([i], (xa[i] - xb[i])**2)).set_name("r")
        t = (1 + pm.sqrt(3) * r/lscale) * pm.exp(-pm.sqrt(3)*r/lscale)
        out.write(t)
    return graph("y", {"xa": xa_, "xb": xb_, "lscale": lscale_})

def dsp(signal):
    return np.abs(np.fft.fft(signal))

def logreg(x, w):
    return sigmoid(x.dot(w.T).reshape(-1))

def datagen(n_rows, n_cols):
    data = np.random.uniform(0, 1, (n_rows, n_cols))
    return data

def mpc(X, Y):
    k1 = Matern(length_scale=[np.std(X)])
    # k1 = gpflow.kernels.Matern32(np.std(Y)/np.sqrt(2), active_dims=[0], lengthscales=[np.std(X)])
    ll_var = np.std(Y)/np.sqrt(2)
    m = GaussianProcessRegressor(kernel=Matern(length_scale=[np.std(X)], nu=1.5), alpha=ll_var)
    # m = gpflow.models.GPR((X, Y.T), k1, noise_variance=ll_var)
    return m


    # k = Matern(length_scale=lscale, nu=v)



def test_mpc(X, Y):
    y = Y - 0.0


def test_other():
    n_rows = 1
    n_cols = 128
    all_data = []
    burn_in = 20  # number of initial runs to get 5 pairs of data points to feed into the optimizer
    Y = np.empty((burn_in, n_rows))
    X = np.empty((n_rows, burn_in))
    weights = np.random.uniform(-0.5, 0.5, n_cols).astype(np.float)

    for i in range(burn_in):
        param = np.float64(np.random.randint(0, 100))
        # component one: random data generator
        data_in = datagen(n_rows, n_cols)

        # Compnent two: signal processing component
        # You can replace it with your fft implementation from signal processing module
        data_out_signal = dsp(data_in)
        logreg_out = logreg(data_out_signal, weights)
        Y[i] = logreg_out
        X[0][i] = param
    # res = mpc(X.T, Y.T)
    # param = np.asarray([np.float64(np.random.randint(0, 75, (1,)))]).reshape(-1,1)
    # data_in = datagen(n_rows, n_cols)
    # data_out_signal = dsp(data_in)
    # mpc_inp = logreg(data_out_signal, weights)
    # print(res.predict(param))
    # print(res._build_predict(data_in))
    lscale = [np.std(X)]
    v = 1.5
    print(matern_kernel(X, lscale, 1.5))


def optimize(data, n_rows, n_cols, weights, niter=10):

    for i in range(niter):
        param = np.float64(np.random.randint(0, 100))
        # component one: random data generator
        data_in = datagen(n_rows, n_cols)

        # Compnent two: signal processing component
        # You can replace it with your fft implementation from signal processing module
        data_out_signal = dsp(data_in)
        logreg_out = logreg(data_out_signal, weights)
        # Y[i] = logreg_out
        # X[0][i] = param




if __name__ == "__main__":
    # test_other()
    data = np.random.uniform(0, 1, (1, 100))
    t = Matern(length_scale=[np.std(data)], nu=1.5)
    tdata = np.random.uniform(0, 0.2, (1, 100))
    tout = t(tdata, data)
    np_test = matern_kernel(tdata, data, np.std(data))

    pm_test = matern(tdata[0], data[0], np.std(data))
    # print(tout)
    print(np_test)
    print(pm_test)
    # x, w = create_data(16)
    #
    # graph = get_graph()
    #
    # output = graph("Y", {"x": x, "W": w})
    # np_output = np_test(x, w)
    # print(f"{np.allclose(np_output, output)}")


