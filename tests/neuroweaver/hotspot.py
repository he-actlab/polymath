import polymath as pm
import numpy as np
BLOCK_SIZE = 16
BLOCK_SIZE_C = BLOCK_SIZE
BLOCK_SIZE_R = BLOCK_SIZE

STR_SIZE = 256

'''
MAX_PD	= 3000000
PRECISION =	0.001
SPEC_HEAT_SI = 1.75e6
K_SI = 100
FACTOR_CHIP = 0.5

t_chip = 0.0005
chip_height = 0.016
chip_width = 0.016
amb_temp = 80.0
'''

MAX_PD	= 3000000
PRECISION =	1
SPEC_HEAT_SI = 2
K_SI = 100
FACTOR_CHIP = 1

t_chip = 1
chip_height = 1
chip_width = 1
amb_temp = 80
## pass result and temp as 2D matrix
def hotspot_impl():
    with pm.Node(name="hotspot") as graph:
        # num_iterations = pm.parameter("num_iterations")
        row = pm.parameter("row")
        col = pm.parameter("col")

        result = pm.output("result", shape=(row, col))
        temp = pm.input("temp", shape=(row, col))
        power = pm.input("power", shape=(row, col))


        grid_height = chip_height * row
        grid_width = chip_width * col

        Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
        Rx = grid_width * (2 * K_SI * t_chip * grid_height)
        Ry = grid_height * (2 * K_SI * t_chip * grid_width)
        Rz = t_chip * (K_SI * grid_height * grid_width)

        max_slope = MAX_PD * (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
        step = PRECISION * max_slope * 1000
        Rx_1 = 1 * Rx
        Ry_1 = 1 * Ry
        Rz_1 = 1 * Rz
        Cap_1 = step * Cap

        r = pm.index(1, (row - 2), name="r")
        c = pm.index(1, (col - 2), name="c")
        rt = pm.index(0, (row - 1), name="rt")
        ct = pm.index(0, (col - 1), name="ct")
        result[rt, ct] = power[rt, ct] + 1
        temp1 = temp[r, c]
        temp2 = temp[r + 1, c]
        temp4 = temp[r, c + 1]
        temp5 = temp[r - 1, c]
        temp7 = temp[r, c - 1]
        delta = (power[r,c] + ((temp2 + temp5 - 2 * temp1)).set_name("tempd1") * Ry_1).set_name("delta0")
        delta1 = (delta + (temp4 + temp7 - 2*temp1).set_name("tempd2")*Rx_1).set_name("delta1")
        delta2 = (Cap_1*(delta1 + (amb_temp - temp1) * Rz_1)).set_name("delta2")
        result[r, c] = delta2 + temp1

    return graph
