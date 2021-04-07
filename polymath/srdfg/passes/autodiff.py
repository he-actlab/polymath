from polymath.srdfg.passes import register_pass, Pass
import polymath as pm
from collections import deque
from collections import defaultdict

NON_DNN_NODE_OPS = (pm.write, pm.placeholder, pm.index, pm.var_index,
                    pm.slice_op, pm.func_op, pm.GroupNode, pm.NonLinear)
OPTIMIZERS = {'sgd': pm.sgd}
LOSS_FUNCS = {'cross_entropy': pm.cross_entropy_loss}

@register_pass
class AutoDiffGraph(Pass):
    GRAD_FUNCS = {}

    def __init__(self, loss_func, optimizer, optimizer_args, optimizer_kwargs=None):
        self.loss_func = LOSS_FUNCS[loss_func]
        self.optimizer = OPTIMIZERS[optimizer]
        self.optimizer_kwargs = optimizer_kwargs or {}
        assert isinstance(optimizer_args, tuple)
        self.optimizer_args = optimizer_args
        self.tape = deque()
        self.grad_map = {}
        self.param_map = {}

        super(AutoDiffGraph, self).__init__()

    def apply_pass(self, node:pm.Node, ctx):
        if node.op_name in pm.ONNX_OP_NAMES:
            assert node.op_name in AutoDiffGraph.GRAD_FUNCS
            self.tape.append(node)
        return node

    def package_pass(self, node, ctx):
        with node:
            out_loss = pm.output(name=f"{self.tape[0].outputs[0].name}_loss")
            target = pm.input(name="target", shape=self.tape[0].outputs[0].shape[0])
            self.loss_func(self.tape[0].outputs[0], target, out_loss)
        self.grad_map[self.tape[0].outputs[0].name] = out_loss
        while self.tape:
            n = self.tape.pop()
            assert n.graph is not None and n.op_name in AutoDiffGraph.GRAD_FUNCS
            with n.graph:
                AutoDiffGraph.GRAD_FUNCS[n.op_name](self, n)
    
    def conv_bias_grad(self, node):
        with node.graph:
            pm.conv_transpose()
    
    def elem_add_grad(self, node):
        pass
    
    def conv_grad(self, node):
        conv_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0])
    
    def batch_norm_grad(self, node):
        pass
    
    def relu_grad(self, node):
        pass
    
    def gemm_grad(self, node):
        node_output_grad = self.grad_map[node.outputs[0].name]
        gemm_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        gemm_weight_grad = pm.output(name=f"{node.inputs[1].name}_grad", shape=node.inputs[1].shape)

        if len(node.inputs) > 2:
            gemm_bias_grad = pm.output(name=f"{node.inputs[2].name}_grad", shape=node.inputs[2].shape)
    
        with node.graph:
            # Input grad
            pm.gemm_no_bias(node_output_grad, node.inputs[1], gemm_weight_grad)
            # Weight grad
            pm.gemm_no_bias(node_output_grad, node.inputs[0], gemm_inp_grad, transA=True)

            # Weight update
            # self.optimizer(node.inputs[1], gemm_weight_grad, *self.optimizer_args, **self.optimizer_kwargs)

            # Bias grad
            if len(node.inputs) > 2:
                pm.reduce_sum(node_output_grad, gemm_bias_grad)


    
    
    def max_pool_grad(self, node):
        pass
    
    def global_avg_pool_grad(self, node):
        pass
    
    def flatten_grad(self, node):
        pass

    GRAD_FUNCS['conv'] = conv_grad
    GRAD_FUNCS['conv_bias'] = conv_bias_grad
    GRAD_FUNCS['relu'] = relu_grad
    GRAD_FUNCS['batch_norm'] = batch_norm_grad
    GRAD_FUNCS['max_pool'] = max_pool_grad
    GRAD_FUNCS['global_avg_pool'] = global_avg_pool_grad
    GRAD_FUNCS['gemm'] = gemm_grad
    GRAD_FUNCS['elem_add'] = elem_add_grad
    GRAD_FUNCS['coarse_flatten'] = flatten_grad

