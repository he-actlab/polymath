from polymath.codegen.dnnweavergen.dnnweaver2.scalar.dtypes import Dtype

class ScalarOp(object):
    def __init__(self, op_str, dtype):
        self.op_str = op_str
        self.dtype = dtype
    def __str__(self):
        if isinstance(self.dtype, Dtype):
            return '{}({})'.format(self.op_str, self.dtype.__str__())
        else:
            ret = str(self.op_str)
            ret += '('
            ret += ','.join([x.__str__() for x in self.dtype])
            ret += ')'
            return ret


class ScalarOpTypes(object):
    def __init__(self):
        self.MulOp = {}
        self.MacOp = {}
        self.SqrOp = {}
        self.CmpOp = {}
        self.AddOp = {}
        self.SubOp = {}
        self.RshiftOp = {}
    def MUL(self, dtypes):
        assert len(dtypes) == 2
        dtype_str = tuple(d.__str__() for d in dtypes)
        if dtype_str not in self.MulOp.keys():
            self.MulOp[dtype_str] = ScalarOp('Multiply', dtypes)
        return self.MulOp[dtype_str]

    def MAC(self, dtypes):
        assert len(dtypes) == 3
        dtype_str = tuple(d.__str__() for d in dtypes)
        if dtype_str not in self.MacOp.keys():
            self.MacOp[dtype_str] = ScalarOp('Multiply-Accumulate', dtypes)
        return self.MacOp[dtype_str]

    def SQR(self, dtypes):
        assert isinstance(dtypes, Dtype)

        dtype_str = dtypes.__str__()

        if dtype_str not in self.SqrOp.keys():
            self.SqrOp[dtype_str] = ScalarOp('Square', dtypes)
        return self.SqrOp[dtype_str]
    def CMP(self, dtypes):
        assert isinstance(dtypes, Dtype), 'Got Dtypes: {}'.format(dtypes)
        dtype_str = dtypes.__str__()

        if dtype_str not in self.CmpOp.keys():
            self.CmpOp[dtype_str] = ScalarOp('Compare', dtypes)
        return self.CmpOp[dtype_str]

    def ADD(self, dtypes):
        assert len(dtypes) == 2
        dtype_str = tuple(d.__str__() for d in dtypes)

        if dtype_str not in self.AddOp.keys():
            self.AddOp[dtype_str] = ScalarOp('Addition', dtypes)
        return self.AddOp[dtype_str]
    def SUB(self, dtypes):
        assert len(dtypes) == 2
        dtype_str = tuple(d.__str__() for d in dtypes)

        if dtype_str not in self.SubOp.keys():
            self.SubOp[dtype_str] = ScalarOp('Subtract', dtypes)
        return self.SubOp[dtype_str]
    def RSHIFT(self, dtypes):
        assert isinstance(dtypes, Dtype), 'Got Dtypes: {}'.format(dtypes)
        dtype_str = dtypes.__str__()

        if dtype_str not in self.RshiftOp.keys():
            self.RshiftOp[dtype_str] = ScalarOp('Rshift', dtypes)
        return self.RshiftOp[dtype_str]


Ops = ScalarOpTypes()
