import polymath as pm
import numpy as np
BLOCK_SIZE = 16
BLOCK_SIZE_C = BLOCK_SIZE
BLOCK_SIZE_R = BLOCK_SIZE

STR_SIZE = 256

MAX_PD	= 3000000
PRECISION =	1
SPEC_HEAT_SI = 2
K_SI = 100
FACTOR_CHIP = 1

t_chip = 1
chip_height = 1
chip_width = 1
amb_temp = 80

## pass result and temp as 3D matrix
def impl_3D():
    with pm.Node(name="hotspot_3d") as graph:
        numCols = pm.parameter("numCols")
        numRows = pm.parameter("numRows")
        layers = pm.parameter("layers")

        tIn = pm.input("temp", shape=(numCols, numRows, layers))
        pIn = pm.input("power", shape=(numCols, numRows, layers))
        tOut = pm.output("result", shape=(numCols, numRows, layers))

        dx = chip_height * numRows
        dy = chip_width * numCols
        dz = t_chip * layers

        Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy
        Rx = dy * (2 * K_SI * t_chip * dx)
        Ry = dx * (2 * K_SI * t_chip * dy)
        Rz = dz * (K_SI * dx * dy)

        max_slope = MAX_PD * (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
        dt = PRECISION * max_slope

        stepDivCap = dt * Cap
        ce =stepDivCap * Rx
        cn =stepDivCap * Ry
        ct =stepDivCap * Rz
        cw =stepDivCap * Rx
        cs =stepDivCap * Ry
        cb =stepDivCap * Rz

        cc = 1 - (2*ce + 2*cn + 3*ct)
        z = pm.index(1, (numCols - 2), name="z")
        y = pm.index(1, (numRows - 2), name="y")
        x = pm.index(1, (layers - 2), name="x")

        zt = pm.index(0, (numCols - 1), name="zt")
        yt = pm.index(0, (numRows - 1), name="yt")
        xt = pm.index(0, (layers - 1), name="xt")

        tIn1 = (tIn[z, y, x]).set_name("tin1")
        tIn2 = (tIn[z, y-1, x]).set_name("tin2")
        tIn3 = (tIn[z, y+1, x]).set_name("tin3")
        tIn4 = (tIn[z, y, x-1]).set_name("tin4")
        tIn5 = (tIn[z, y, x+1]).set_name("tin5")
        tIn6 = (tIn[z-1, y, x]).set_name("tin6")
        tIn7 = (tIn[z+1, y, x]).set_name("tin7")

        delta12 = (tIn2 * cn).set_name("delta12")
        delta1 = ((tIn1 * cc).set_name("delta11") + delta12).set_name("delta1")
        delta2 = (delta1 + (tIn3 * cs) + (tIn5 * ce)).set_name("delta2")
        delta3 = (delta2 + (tIn4 * cw) + (tIn7 * ct)).set_name("delta3")
        delta4 = (delta3 + (tIn6 * cb) + (ct * amb_temp)).set_name("delta4")
        delta5 = ((dt * Cap) * pIn[z, y, x]).set_name("delta5")
        delta6 = (delta4 + delta5).set_name("delta6")
        #tOut[z,y,x] = tIn[z,y,x]*cc + tIn[z,y,x]*cn + tIn[z,y,x]*cs + tIn[z,y,x]*ce + tIn[z,y,x]*cw + tIn[z,y,x]*ct + tIn[z,y,x]*cb + (dt*Cap) * pIn[z,y,x] + ct*amb_temp
        tOut[zt, yt, xt] = pIn[zt, yt, xt] + 1
        tOut[z, y, x] = delta6
    return graph



