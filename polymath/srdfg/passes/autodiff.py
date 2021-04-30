from polymath.srdfg.passes import register_pass, Pass
import polymath as pm
from collections import deque
from collections import defaultdict

NON_DNN_NODE_OPS = (pm.write, pm.placeholder, pm.index, pm.var_index,
                    pm.slice_op, pm.func_op, pm.GroupNode, pm.NonLinear)
OPTIMIZERS = {'sgd': pm.sgd}
LOSS_FUNCS = {'cross_entropy': pm.cross_entropy_loss, 'test_loss_fn': pm.elem_sub}

@register_pass
class AutoDiffGraph(Pass):
    GRAD_FUNCS = {}

    def __init__(self, loss_func, optimizer, optimizer_kwargs=None):
        self.loss_func = LOSS_FUNCS[loss_func]
        self.optimizer_name = optimizer
        self.optimizer = OPTIMIZERS[optimizer]
        self.optimizer_kwargs = optimizer_kwargs or {}
        assert isinstance(self.optimizer_kwargs, dict)

        self.tape = deque()
        self.grad_map = {}
        self.param_map = {}

        super(AutoDiffGraph, self).__init__()

    def apply_pass(self, node:pm.Node, ctx):
        if node.op_name in pm.ONNX_OP_NAMES:
            if node.op_name not in AutoDiffGraph.GRAD_FUNCS:
                raise RuntimeError(f"No definition for operation {node.op_name}")
            self.tape.append(node)
        return node

    def package_pass(self, node, ctx):
        node.name = f"{node.name}_training"

        final_node = self.tape[-1]
        if final_node.op_name not in LOSS_FUNCS:
            with node:
                out_loss = pm.output(name=f"{final_node.outputs[0].name}_loss", shape=final_node.outputs[0].shape)
                target = pm.input(name=f"target_{final_node.outputs[0].name}", shape=final_node.outputs[0].shape[0])
                loss_node = self.loss_func(final_node.outputs[0], target, out_loss)
                self.grad_map[out_loss.name] = out_loss
                AutoDiffGraph.GRAD_FUNCS[loss_node.op_name](self, loss_node)

        while self.tape:
            n = self.tape.pop()
            assert n.graph is not None and n.op_name in AutoDiffGraph.GRAD_FUNCS
            with node:
                AutoDiffGraph.GRAD_FUNCS[n.op_name](self, n)
        return node
    
    def elem_add_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]

        a_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        b_grad = pm.output(name=f"{node.inputs[1].name}_grad", shape=node.inputs[1].shape)
        pm.elem_add_grad(node.inputs[0], node.inputs[1], grad, a_grad, b_grad)
        self.grad_map[node.inputs[0].name] = a_grad
        self.grad_map[node.inputs[1].name] = b_grad

    
    def conv_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        conv_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad_{node.name}", shape=node.inputs[0].shape)
        conv_weight_grad = pm.output(name=f"{node.inputs[1].name}_grad_{node.name}", shape=node.inputs[1].shape)

        if len(node.inputs) > 2:
            conv_bias_grad = pm.output(name=f"{node.inputs[2].name}_grad", shape=node.inputs[2].shape)
            pm.conv_grad(node.inputs[0], node.inputs[1], node.inputs[2], grad,
                         conv_inp_grad, conv_weight_grad, conv_bias_grad,
                         self.optimizer_name, self.optimizer_kwargs,
                         stride=node.kwargs['stride'],
                         pad=node.kwargs['pad'],
                         dilation=node.kwargs['dilation'])

        else:
            pm.conv_grad_no_bias(node.inputs[0], node.inputs[1], grad, conv_inp_grad, conv_weight_grad,
                                 self.optimizer_name, self.optimizer_kwargs, stride=node.kwargs['stride'],
                                 pad=node.kwargs['pad'], dilation=node.kwargs['dilation'])

        self.grad_map[node.inputs[0].name] = conv_inp_grad

    def batch_norm_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        bn_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        bn_scale_grad = pm.output(name=f"{node.inputs[1].name}_grad_{node.name}", shape=node.inputs[1].shape)
        bn_bias_grad = pm.output(name=f"{node.inputs[2].name}_grad_{node.name}", shape=node.inputs[2].shape)
        inp = node.inputs[0]
        scale = node.inputs[1]
        bias = node.inputs[2]
        mean = node.inputs[3]
        var = node.inputs[4]
        pm.batchnorm_grad(inp, scale, bias, mean, var, grad, bn_inp_grad, bn_scale_grad, bn_bias_grad,
                          self.optimizer_name, self.optimizer_kwargs)
        self.grad_map[node.inputs[0].name] = bn_inp_grad


    def relu_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        out_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.relu_grad(node.inputs[0], grad, out_grad)
        self.grad_map[node.inputs[0].name] = out_grad

    def elem_tanh_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        out_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.elem_tanh_grad(node.inputs[0], grad, out_grad)
        self.grad_map[node.inputs[0].name] = out_grad

    def gemm_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]

        gemm_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        gemm_weight_grad = pm.output(name=f"{node.inputs[1].name}_grad", shape=node.inputs[1].shape)

        if len(node.inputs) > 2:
            gemm_bias_grad = pm.output(name=f"{node.inputs[2].name}_grad", shape=node.inputs[2].shape)
            pm.gemm_grad(node.inputs[0], node.inputs[1], node.inputs[2], grad,
                         gemm_inp_grad, gemm_weight_grad, gemm_bias_grad,
                         self.optimizer_name, self.optimizer_kwargs)

        else:
            pm.gemm_grad_no_bias(node.inputs[0], node.inputs[1], grad, gemm_inp_grad, gemm_weight_grad,
                                 self.optimizer_name, self.optimizer_kwargs)

        self.grad_map[node.inputs[0].name] = gemm_inp_grad

    def max_pool_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        mpool_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.max_pool_grad(node.inputs[0], grad, mpool_grad, node.kernel_size[0], node.kernel_size[1],
                         node.stride, node.pad)
        self.grad_map[node.inputs[0].name] = mpool_grad

    def average_pool_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        apool_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.average_pool_grad(node.inputs[0], grad, apool_grad, node.kernel_size[0], node.kernel_size[1],
                         node.stride, node.pad)
        self.grad_map[node.inputs[0].name] = apool_grad

    def global_avg_pool_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        global_average_pool_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.global_average_pool_grad(node.inputs[0], grad, global_average_pool_grad)
        self.grad_map[node.inputs[0].name] = global_average_pool_grad
    
    def flatten_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        flatten_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.flatten_grad(node.inputs[0], grad, flatten_inp_grad)
        self.grad_map[node.inputs[0].name] = flatten_inp_grad

    def cross_entropy_grad(self, node):
        grad = self.grad_map[node.outputs[0].name]
        inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.cross_entropy_loss_grad(node.inputs[0], node.inputs[1], grad, inp_grad)
        self.grad_map[node.inputs[0].name] = inp_grad


    GRAD_FUNCS['conv'] = conv_grad
    GRAD_FUNCS['conv_bias'] = conv_grad
    GRAD_FUNCS['relu'] = relu_grad
    GRAD_FUNCS['batch_norm'] = batch_norm_grad
    GRAD_FUNCS['max_pool'] = max_pool_grad
    GRAD_FUNCS['elem_tanh'] = elem_tanh_grad
    GRAD_FUNCS['avg_pool'] = average_pool_grad
    GRAD_FUNCS['global_avg_pool'] = global_avg_pool_grad
    GRAD_FUNCS['gemm'] = gemm_grad
    GRAD_FUNCS['elem_add'] = elem_add_grad
    GRAD_FUNCS['coarse_flatten'] = flatten_grad
    GRAD_FUNCS['cross_entropy_loss'] = cross_entropy_grad


def create_training_graph(graph, loss_func="cross_entropy", optimizer="sgd", **optimizer_kwargs):
    autodiff_pass = pm.AutoDiffGraph(loss_func, optimizer, optimizer_kwargs)
    train_graph = autodiff_pass(graph)
    lower_pass = pm.Lower(pm.DNN_TRAINING_OPS)
    lowered_train_graph = lower_pass(train_graph)
    return lowered_train_graph
