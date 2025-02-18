import polymath as pm
import numpy as np



class matern32(pm.Template):
    def define_graph(self, x, y, variance, lengthscale):
        sqrt3 = pm.sqrt(3.0)
        i = pm.index(0, x.shape[0]-1, "i")
