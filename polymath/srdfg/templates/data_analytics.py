import polymath as pm
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools


class svm_classifier_train(pm.Template):
    def define_graph(self, x, w, y, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        c = (y*h).set_name("c")
        ny = (0 - y).set_name("ny")
        p = ((c > 1)*ny).set_name("p")
        g = (p * x[i]).set_name("g")
        w[i] = w[i] - mu * g[i]

class logistic_regressor_train(pm.Template):

    def define_graph(self, x, w, y, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]


class linear_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        y_pred.write(pm.sum([i], (x[i] * w[i]), name="h"))


class logistic_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        y_pred.write(pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h")))


class mc_logistic_regressor_train(pm.Template):

    def define_graph(self, x, w, y, y_pred, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.temp(name="h", shape=(m))
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]

class mc_logistic_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))

class linear_regressor_train(pm.Template):

    def define_graph(self, x, w, y, mu, m):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]




class ppo(pm.Template):
    def define_graph(self, obs, action, states,
                     gamma=0.99,
                     clip=0.2,
                     ent_coeff=0.01,
                     lam=0.95,
                     adam_eps=1e-5):
        pass




# TODO: Add reshape operator, constant operator, gemm

