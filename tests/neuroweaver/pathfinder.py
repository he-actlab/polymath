import numpy as np
import polymath as pm

def pathfinder_impl():
    with pm.Node(name="pathfinder") as graph:

        rows = pm.parameter("rows")
        cols = pm.parameter("cols")

        data = pm.input("data", shape=(rows, cols))
        wall = pm.input("wall", shape=(rows, cols))
        min_var = pm.input("min_var", shape=(2, cols))
        result = pm.output("result", shape=(2, cols))

        iR = pm.index(0, rows - 1, name="iR")
        iC = pm.index(0, cols - 1, name="iC")
        jt = pm.index(0, 1, name="jt")

        # wall[iR, iC] = wall[iR,iC] + data[iR,iC]
        tmp = wall[iR,iC] + data[iR,iC]

        #t = pm.index(0, rows - 1, name="t")
        n = pm.index(0, cols - 1, name="n")
        j = pm.index(0, 1, name="j")
        #
        # min_var = (result[j,n] + 1).set_name("min_var1")        ## ideally, it uses arg n-1
        #
        # min_var = (((wall[j,n] > min_var[j,n]) * min_var[j,n]) + ((wall[j,n] <= min_var[j,n]) * wall[j,n])).set_name("min2")
        mvar_temp = (((tmp[j,n] > min_var[j,n]) * min_var[j,n]) + ((tmp[j,n] <= min_var[j,n]) * tmp[j,n])).set_name("min2")
        # min_var = ((((wall[iR,iC] + data[iR,iC]) > min_var[j,iC]) * min_var[j,iC]) + (((wall[iR,iC] + data[iR,iC]) <= min_var[j,iC]) * (wall[iR,iC] + data[iR,iC]))).set_name("min2")
        #
        result[j, n] = (tmp[j,n] + mvar_temp[j,n]).set_name("result1")     ## ideally uses t+1, but polymath fails
        #
        # result_final[j,iC] = (result).set_name("result_final1")

    return graph
