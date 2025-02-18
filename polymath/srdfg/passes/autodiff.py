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
        if node.op_name in AutoDiffGraph.GRAD_FUNCS:
            self.tape.append(node)
        return node

    def package_pass(self, node, ctx):

        if "train" not in node.name:
            node.name = f"{node.name}_train"

        final_node = self.tape[-1]
        if final_node.op_name not in LOSS_FUNCS:
            with node:
                out_loss = pm.output(name=f"{final_node.outputs[0].name}_loss", shape=final_node.outputs[0].shape[0])
                target = pm.input(name=f"target_{final_node.outputs[0].name}", shape=final_node.outputs[0].shape[0])
                loss_node = self.loss_func(final_node.outputs[0], target, out_loss)
                self.update_grad_map(out_loss, out_loss, node)
                AutoDiffGraph.GRAD_FUNCS[loss_node.op_name](self, loss_node)

        while self.tape:
            n = self.tape.pop()
            assert n.graph is not None and n.op_name in AutoDiffGraph.GRAD_FUNCS
            with node:
                AutoDiffGraph.GRAD_FUNCS[n.op_name](self, n)
        return node

    def update_grad_map(self, input_node, gradient_node, parent_node):
        if input_node.name in self.grad_map:
            assert gradient_node.shape == self.grad_map[input_node.name].shape
            grad_name = f"{self.grad_map[input_node.name].name},{gradient_node.name}"
            acc_grad = pm.output(name=grad_name, shape=gradient_node.shape)
            pm.elem_add(self.grad_map[input_node.name], gradient_node, acc_grad)
            self.grad_map[input_node.name] = acc_grad
        else:
            self.grad_map[input_node.name] = gradient_node

    def get_gradient(self, parent_node):
        if len(parent_node.outputs) > 1:
            grads = []
            for o in parent_node.outputs:
                assert o.name in self.grad_map
                grads.append(self.grad_map[o.name])
            return tuple(grads)
        else:
            assert len(parent_node.outputs) == 1 and parent_node.outputs[0].name in self.grad_map
            return self.grad_map[parent_node.outputs[0].name]


    def elem_add_grad(self, node):
        grad = self.get_gradient(node)
        self.update_grad_map(node.inputs[0], grad, node)
        self.update_grad_map(node.inputs[1], grad, node)

    def conv_grad(self, node):
        grad = self.get_gradient(node)
        conv_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad_{node.name}", shape=node.inputs[0].shape)
        conv_weight_grad = pm.output(name=f"{node.inputs[1].name}_grad_{node.name}", shape=node.inputs[1].shape)

        if len(node.inputs) > 2:
            conv_bias_grad = pm.output(name=f"{node.inputs[2].name}_grad", shape=node.inputs[2].shape)

            pm.conv_grad(node.inputs[0], node.inputs[1], node.inputs[2], grad,
                         conv_inp_grad, conv_weight_grad, conv_bias_grad,
                         self.optimizer_name, self.optimizer_kwargs,
                         stride=int(node.kwargs['stride']),
                         pad=int(node.kwargs['pad'][0]),
                         dilation=int(node.kwargs['dilation']))

        else:
            pm.conv_grad_no_bias(node.inputs[0], node.inputs[1], grad, conv_inp_grad, conv_weight_grad,
                                 self.optimizer_name, self.optimizer_kwargs, stride=int(node.kwargs['stride']),
                                 pad=int(node.kwargs['pad'][0]), dilation=int(node.kwargs['dilation']))

        self.update_grad_map(node.inputs[0], conv_inp_grad, node)

    def mean_var_grad(self, node):
        pass

    def batch_norm_grad(self, node):
        grad = self.get_gradient(node)
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
        self.update_grad_map(node.inputs[0], bn_inp_grad, node)

    def relu_grad(self, node):
        grad = self.get_gradient(node)
        out_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.relu_grad(node.inputs[0], grad, out_grad)
        self.update_grad_map(node.inputs[0], out_grad, node)

    def elem_tanh_grad(self, node):
        grad = self.get_gradient(node)
        out_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.elem_tanh_grad(node.inputs[0], grad, out_grad)

        self.update_grad_map(node.inputs[0], out_grad, node)

    def gemm_grad(self, node):
        grad = self.get_gradient(node)
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

        self.update_grad_map(node.inputs[0], gemm_inp_grad, node)

    def max_pool_grad(self, node):
        grad = self.get_gradient(node)
        mpool_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.max_pool_grad(node.inputs[0], grad, mpool_grad, node.kernel_size[0], node.kernel_size[1],
                         node.stride, node.pad)
        self.update_grad_map(node.inputs[0], mpool_grad, node)

    def average_pool_grad(self, node):
        grad = self.get_gradient(node)
        apool_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.average_pool_grad(node.inputs[0], grad, apool_grad, node.kernel_size[0], node.kernel_size[1],
                         node.stride, node.pad)
        self.update_grad_map(node.inputs[0], apool_grad, node)


    def global_avg_pool_grad(self, node):
        grad = self.get_gradient(node)
        gavg_pool_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.global_average_pool_grad(node.inputs[0], grad, gavg_pool_grad)
        self.update_grad_map(node.inputs[0], gavg_pool_grad, node)

    
    def flatten_grad(self, node):
        grad = self.get_gradient(node)
        flatten_inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.flatten_grad(node.inputs[0], grad, flatten_inp_grad)
        self.update_grad_map(node.inputs[0], flatten_inp_grad, node)

    def cross_entropy_grad(self, node):
        grad = self.get_gradient(node)

        inp_grad = pm.output(name=f"{node.inputs[0].name}_grad", shape=node.inputs[0].shape)
        pm.cross_entropy_loss_grad(node.inputs[0], node.inputs[1], grad, inp_grad)
        self.update_grad_map(node.inputs[0], inp_grad, node)

    GRAD_FUNCS['conv'] = conv_grad
    GRAD_FUNCS['conv_bias'] = conv_grad
    GRAD_FUNCS['relu'] = relu_grad
    GRAD_FUNCS['batch_norm'] = batch_norm_grad
    GRAD_FUNCS['mean_var'] = mean_var_grad
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
