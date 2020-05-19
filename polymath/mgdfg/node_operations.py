# import functools
# import operator
# import builtins
# import importlib
# import numpy as np
# from numbers import Integral
# import contextlib
# from .domain import Domain
# from .base import Node
# from . import nodes
#
# from .util import _flatten_iterable, _is_node_type_instance
#
#
# class func_op(Node):  # pylint: disable=C0103,R0903
#     """
#     Node wrapper for stateless functions.
#
#     Parameters
#     ----------
#     target : callable
#         function to evaluate the node
#     args : tuple
#         positional arguments passed to the target
#     kwargs : dict
#         keywoard arguments passed to the target
#     """
#     def __init__(self, target, *args, **kwargs):
#         kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs \
#             else f"{target.__name__}"
#         if "domain" in kwargs:
#             dom = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
#         elif len(args) == 2:
#             all_args = _flatten_iterable(args)
#             slice1_var, slice1_idx, slice2_var, slice2_idx = self.get_index_nodes(all_args[0], all_args[1])
#             dom = slice1_idx.combine_set_domains(slice2_idx)
#
#         else:
#             dom = Domain(tuple([]))
#         self._target = None
#         super(func_op, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", domain=dom, **kwargs)
#         self.target = target
#         self.added_attrs += ["domain", "target"]
#
#     @property
#     def target(self):
#         return self._target
#
#     @target.setter
#     def target(self, fnc):
#         self._target = fnc
#         self.op_name = f"{fnc.__name__}"
#         self.kwargs["target"] = f"{fnc.__module__}.{fnc.__name__}"
#
#     @property
#     def domain(self):
#         return self.kwargs["domain"]
#
#     def get_index_nodes(self, slice1_var=None, slice2_var=None):
#         if slice1_var is None and slice2_var is None:
#             slice1_var, slice2_var = self.args
#
#         if isinstance(slice1_var, (slice_op, nodes.var_index)) or _is_node_type_instance(slice1_var, "GroupNode"):
#             slice1_idx = slice1_var.domain
#         else:
#             slice1_idx = Domain(tuple([]))
#
#         if isinstance(slice2_var, (slice_op, nodes.var_index)) or _is_node_type_instance(slice2_var, "GroupNode"):
#             slice2_idx = slice2_var.domain
#         else:
#             slice2_idx = Domain(tuple([]))
#         return slice1_var, slice1_idx, slice2_var, slice2_idx
#
#     def _evaluate(self, *args, **kwargs):
#
#         for aa in list(kwargs.keys()):
#             if aa in self.added_attrs:
#                 kwargs.pop(aa)
#         return self.target(*args, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#         return call(self, *args, **kwargs)
#
#     def __repr__(self):
#         return "<func_op '%s' target=%s args=<%d items>>" % \
#             (self.name, self.kwargs["target"], len(self.args))
#
#     def __add__(self, other):
#         return add(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__radd__(self)
#
#     def __radd__(self, other):
#         return add(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__add__(self)
#
#     def __sub__(self, other):
#         return sub(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rsub__(self)
#
#     def __rsub__(self, other):
#         return sub(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__sub__(self)
#
#     def __pow__(self, other):
#         return pow_(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rpow__(self)
#
#     def __rpow__(self, other):
#         return pow_(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rpow__(self)
#
#     def __matmul__(self, other):
#         return matmul(self, other, graph=self.graph)
#
#     def __rmatmul__(self, other):
#         return matmul(other, self, graph=self.graph)
#
#     def __mul__(self, other):
#         return mul(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rmul__(self)
#
#     def __rmul__(self, other):
#         return mul(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__mul__(self)
#
#     def __truediv__(self, other):
#         return truediv(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__truediv__(self)
#
#     def __rtruediv__(self, other):
#         return truediv(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rtruediv__(self)
#
#     def __floordiv__(self, other):
#         return floordiv(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rfloordiv__(self)
#
#     def __rfloordiv__(self, other):
#         return floordiv(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__floordiv__(self)
#
#     def __mod__(self, other):
#         return mod(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rmod__(self)
#
#     def __rmod__(self, other):
#         return mod(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__mod__(self)
#
#     def __lshift__(self, other):
#         return lshift(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rlshift__(self)
#
#     def __rlshift__(self, other):
#         return lshift(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__lshift__(self)
#
#     def __rshift__(self, other):
#         return rshift(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rrshift__(self)
#
#     def __rrshift__(self, other):
#         return rshift(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rshift__(self)
#
#     def __and__(self, other):
#         return and_(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rand__(self)
#
#     def __rand__(self, other):
#         return and_(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__and__(self)
#
#     def __or__(self, other):
#         return or_(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__ror__(self)
#
#     def __ror__(self, other):
#         return or_(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__or__(self)
#
#     def __xor__(self, other):
#         return xor(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rxor__(self)
#
#     def __rxor__(self, other):
#         return xor(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__xor__(self)
#
#     def __lt__(self, other):
#         return lt(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__gt__(self)
#
#     def __le__(self, other):
#         return le(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__ge__(self)
#
#     def __eq__(self, other):
#         return hash(self) == hash(other)
#
#     def __ne__(self, other):
#         return ne(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__ne__(self)
#
#     def __gt__(self, other):
#         return gt(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__lt__(self)
#
#     def __ge__(self, other):
#         return ge(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__le__(self)
#
#     def __invert__(self):
#         return inv(self, graph=self.graph)
#
#     def __neg__(self):
#         return neg(self, graph=self.graph)
#
#     def __abs__(self):
#         return abs_(self, graph=self.graph)
#
#     def __pos__(self):
#         return pos(self, graph=self.graph)
#
#     def __reversed__(self):
#         return reversed_(self, graph=self.graph)
#
# class slice_op(Node):
#     """
#     Node representing multi-dimensional operations performed on a node.
#
#     Parameters
#     ----------
#     target : cal
#         The multi-dimensional variable used for indexing into.
#     idx : tuple
#         Tuple of either integer values or index/index_op nodes.
#     """
#     def __init__(self, target, *args, **kwargs):
#
#         if "domain" in kwargs:
#             dom = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
#         else:
#             all_args = _flatten_iterable(args)
#             slice1_var, slice1_idx, slice2_var, slice2_idx = self.get_index_nodes(all_args[0], all_args[1])
#             dom = slice1_idx.combine_set_domains(slice2_idx)
#
#         if "op_name" in kwargs:
#             kwargs.pop("op_name")
#
#         target_name = f"{target.__module__}.{target.__name__}"
#         super(slice_op, self).__init__(*args, target=target_name, domain=dom, op_name=f"slice_{target.__name__}", **kwargs)
#         self.target = target
#
#     @property
#     def domain(self):
#         return self.kwargs["domain"]
#
#     def __getitem__(self, key):
#         if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
#             return self
#         elif self.is_shape_finalized() and len(self.nodes) > 0:
#             if isinstance(key, (int, Node)):
#                 key = tuple([key])
#             assert len(key) == len(self.shape)
#             name = f"{self.name}{key}"
#             ret = self.nodes[name]
#             return ret
#         else:
#             name = []
#             if isinstance(key, Node):
#                 name.append(key.name)
#             elif hasattr(key, "__len__") and not isinstance(key, str):
#                 for k in key:
#                     if isinstance(k, Node):
#                         name.append(k.name)
#                     else:
#                         name.append(str(k))
#             else:
#                 name.append(key)
#             name = self.var.name + "[" + "][".join(name) + "]"
#             if name in self.graph.nodes:
#                 return self.graph.nodes[name]
#             elif isinstance(key, (list)):
#                 return nodes.var_index(self, key, name=name, graph=self.graph)
#             elif isinstance(key, tuple):
#                 return nodes.var_index(self, list(key), name=name, graph=self.graph)
#             else:
#                 return nodes.var_index(self, [key], name=name, graph=self.graph)
#
#     def set_shape(self, shape=None, init=False):
#         s = []
#         assert isinstance(shape, (tuple, list))
#         if all([isinstance(sv, Integral) for sv in shape]) and len(self.domain) == np.product(shape) and len(shape) > 0:
#             self._shape = shape if isinstance(shape, tuple) else tuple(shape)
#         else:
#
#             for idx, d in enumerate(self.domain.dom_set):
#                 if shape and isinstance(shape[idx], (func_op, Integral)):
#                     s.append(shape[idx])
#                 elif shape and isinstance(shape[idx], float):
#                     s.append(int(shape[idx]))
#                 elif isinstance(d, float):
#                     s.append(int(d))
#                 elif isinstance(d, nodes.var_index):
#                     s.append(d.domain)
#                 else:
#                     s.append(d)
#
#             self._shape = tuple(s)
#
#     def is_scalar(self, val):
#         return not isinstance(val, np.ndarray) or (len(val.shape) == 1 and val.shape[0] == 1)
#
#     def _evaluate(self, op1, op2, context=None, **kwargs):
#         if self.is_scalar(op1) or self.is_scalar(op2):
#             value = self.target(op1, op2)
#         else:
#             op1_idx = self.domain.map_sub_domain(self.args[0].domain) if isinstance(self.args[0], Node) else tuple([])
#             op2_idx = self.domain.map_sub_domain(self.args[1].domain) if isinstance(self.args[1], Node) else tuple([])
#             op1 = np.asarray(list(map(lambda x: op1[x], op1_idx))).reshape(self.domain.computed_shape)
#             op2 = np.asarray(list(map(lambda x: op2[x], op2_idx))).reshape(self.domain.computed_shape)
#             value = self.target(op1, op2)
#
#         return value
#
#     def get_index_nodes(self, slice1_var=None, slice2_var=None):
#         if slice1_var is None and slice2_var is None:
#             slice1_var, slice2_var = self.args
#
#         if isinstance(slice1_var, (slice_op, nodes.var_index)) or _is_node_type_instance(slice1_var, "GroupNode"):
#             slice1_idx = slice1_var.domain
#         else:
#             slice1_idx = Domain(tuple([]))
#
#         if isinstance(slice2_var, (slice_op, nodes.var_index)) or _is_node_type_instance(slice2_var, "GroupNode"):
#             slice2_idx = slice2_var.domain
#         else:
#             slice2_idx = Domain(tuple([]))
#         return slice1_var, slice1_idx, slice2_var, slice2_idx
#
#     def __add__(self, other):
#         return slice_op(operator.add, self, other, graph=self.graph)
#
#     def __radd__(self, other):
#         return slice_op(operator.add, other, self, graph=self.graph)
#
#     def __sub__(self, other):
#         return slice_op(operator.sub, self, other, graph=self.graph)
#
#     def __rsub__(self, other):
#         return slice_op(operator.sub, other, self, graph=self.graph)
#
#     def __pow__(self, other):
#         return slice_op(builtins.pow, self, other, graph=self.graph)
#
#     def __rpow__(self, other):
#         return slice_op(builtins.pow, other, self, graph=self.graph)
#
#     def __mul__(self, other):
#         return slice_op(operator.mul, self, other, graph=self.graph)
#
#     def __rmul__(self, other):
#         return slice_op(operator.mul, other, self, graph=self.graph)
#
#     def __truediv__(self, other):
#         return slice_op(operator.truediv, self, other, graph=self.graph)
#
#     def __rtruediv__(self, other):
#         return slice_op(operator.truediv, other, self, graph=self.graph)
#
#     def __floordiv__(self, other):
#         return slice_op(operator.floordiv, self, other, graph=self.graph)
#
#     def __rfloordiv__(self, other):
#         return slice_op(operator.floordiv, other, self, graph=self.graph)
#
#     def __mod__(self, other):
#         return slice_op(operator.mod, self, other, graph=self.graph)
#
#     def __rmod__(self, other):
#         return slice_op(operator.mod, other, self, graph=self.graph)
#
#     def __lshift__(self, other):
#         return slice_op(operator.lshift, self, other, graph=self.graph)
#
#     def __rlshift__(self, other):
#         return slice_op(operator.lshift, other, self, graph=self.graph)
#
#     def __rshift__(self, other):
#         return slice_op(operator.rshift, self, other, graph=self.graph)
#
#     def __rrshift__(self, other):
#         return slice_op(operator.rshift, other, self, graph=self.graph)
#
#     def __and__(self, other):
#         return slice_op(operator.and_, self, other, graph=self.graph)
#
#     def __rand__(self, other):
#         return slice_op(operator.and_, other, self, graph=self.graph)
#
#     def __or__(self, other):
#         return slice_op(operator.or_, self, other, graph=self.graph)
#
#     def __ror__(self, other):
#         return slice_op(operator.or_, other, self, graph=self.graph)
#
#     def __xor__(self, other):
#         return slice_op(operator.xor, self, other, graph=self.graph)
#
#     def __rxor__(self, other):
#         return slice_op(operator.xor, other, self, graph=self.graph)
#
#     def __lt__(self, other):
#         return slice_op(operator.lt, self, other, graph=self.graph)
#
#     def __le__(self, other):
#         return slice_op(operator.lt, other, self, graph=self.graph)
#
#     def __ne__(self, other):
#         return slice_op(operator.ne, self, other, graph=self.graph)
#
#     def __gt__(self, other):
#         return slice_op(operator.gt, self, other, graph=self.graph)
#
#     def __ge__(self, other):
#         return slice_op(operator.ge, self, other, graph=self.graph)
#
#     def __repr__(self):
#         return "<slice_%s '%s'>" % (self.target.__name__, self.name)
#
# def nodeop(target=None, **kwargs):
#     """
#     Decorator for creating nodes from functions.
#     """
#     # This is called when the decorator is used with arguments
#     if target is None:
#         return functools.partial(nodeop, **kwargs)
#
#     # This is called when the decorator is used without arguments
#     @functools.wraps(target)
#     def _wrapper(*args, **kwargs_inner):
#         return func_op(target, *args, **kwargs_inner, **kwargs)
#     return _wrapper
#
# @nodeop
# def call(func, *args, **kwargs):
#     """
#     Call `func` with positional arguments `args` and keyword arguments `kwargs`.
#
#     Parameters
#     ----------
#     func : callable
#         Function to call when the node is executed.
#     args : list
#         Sequence of positional arguments passed to `func`.
#     kwargs : dict
#         Mapping of keyword arguments passed to `func`.
#     """
#     return func(*args, **kwargs)
#
# @contextlib.contextmanager
# def control_dependencies(dependencies, graph=None):
#     """
#     Ensure that all `dependencies` are executed before any nodes in this scope.
#
#     Parameters
#     ----------
#     dependencies : list
#         Sequence of nodes to be evaluted before evaluating any nodes defined in this
#         scope.
#     """
#     # Add dependencies to the graph
#     graph = Node.get_active_graph(graph)
#     graph.dependencies.extend(dependencies)
#     yield
#     # Remove dependencies from the graph
#     del graph.dependencies[-len(dependencies):]
#
#
# @nodeop
# def identity(value):
#     """
#     Node returning the input value.
#     """
#     return value
#
# @nodeop
# def assert_(condition, message=None, *args, val=None):  # pylint: disable=keyword-arg-before-vararg
#     """
#     Return `value` if the `condition` is satisfied and raise an `AssertionError` with the specified
#     `message` and `args` if not.
#     """
#     if message:
#         assert condition, message % args
#     else:
#         assert condition
#
#     return val
#
# @nodeop
# def str_format(format_string, *args, **kwargs):
#     """
#     Use python's advanced string formatting to convert the format string and arguments.
#
#     References
#     ----------
#     https://www.python.org/dev/peps/pep-3101/
#     """
#     return format_string.format(*args, **kwargs)
# # pylint: disable=C0103
# abs_ = nodeop(builtins.abs)
# dict_ = nodeop(builtins.dict)
# help_ = nodeop(builtins.help)
# min_ = nodeop(builtins.min)
# setattr_ = nodeop(builtins.setattr)
# all_ = nodeop(builtins.all)
# dir_ = nodeop(builtins.dir)
# hex_ = nodeop(builtins.hex)
# next_ = nodeop(builtins.next)
# slice_ = nodeop(builtins.slice)
# any_ = nodeop(builtins.any)
# divmod_ = nodeop(builtins.divmod)
# id_ = nodeop(builtins.id)
# object_ = nodeop(builtins.object)
# sorted_ = nodeop(builtins.sorted)
# ascii_ = nodeop(builtins.ascii)
# enumerate_ = nodeop(builtins.enumerate)
# input_ = nodeop(builtins.input)
# oct_ = nodeop(builtins.oct)
# staticmethod_ = nodeop(builtins.staticmethod)
# bin_ = nodeop(builtins.bin)
# eval_ = nodeop(builtins.eval)
# int_ = nodeop(builtins.int)
# open_ = nodeop(builtins.open)
# str_ = nodeop(builtins.str)
# bool_ = nodeop(builtins.bool)
# exec_ = nodeop(builtins.exec)
# isinstance_ = nodeop(builtins.isinstance)
# ord_ = nodeop(builtins.ord)
# sum_ = nodeop(builtins.sum)
# bytearray_ = nodeop(builtins.bytearray)
# filter_ = nodeop(builtins.filter)
# issubclass_ = nodeop(builtins.issubclass)
# pow_ = nodeop(builtins.pow)
# super_ = nodeop(builtins.super)
# bytes_ = nodeop(builtins.bytes)
# float_ = nodeop(builtins.float)
# iter_ = nodeop(builtins.iter)
# print_ = nodeop(builtins.print)
# tuple_ = nodeop(builtins.tuple)
# callable_ = nodeop(builtins.callable)
# format_ = nodeop(builtins.format)
# len_ = nodeop(builtins.len)
# property_ = nodeop(builtins.property)
# type_ = nodeop(builtins.type)
# chr_ = nodeop(builtins.chr)
# frozenset_ = nodeop(builtins.frozenset)
# list_ = nodeop(builtins.list)
# range_ = nodeop(builtins.range)
# vars_ = nodeop(builtins.vars)
# classmethod_ = nodeop(builtins.classmethod)
# getattr_ = nodeop(builtins.getattr)
# locals_ = nodeop(builtins.locals)
# repr_ = nodeop(builtins.repr)
# zip_ = nodeop(builtins.zip)
# compile_ = nodeop(builtins.compile)
# globals_ = nodeop(builtins.globals)
# map_ = nodeop(builtins.map)
# reversed_ = nodeop(builtins.reversed)
# complex_ = nodeop(builtins.complex)
# hasattr_ = nodeop(builtins.hasattr)
# max_ = nodeop(builtins.max)
# round_ = nodeop(builtins.round)
# delattr_ = nodeop(builtins.delattr)
# hash_ = nodeop(builtins.hash)
# memoryview_ = nodeop(builtins.memoryview)
# set_ = nodeop(builtins.set)
# add = nodeop(operator.add)
# and_ = nodeop(operator.and_)
# attrgetter = nodeop(operator.attrgetter)
# concat = nodeop(operator.concat)
# contains = nodeop(operator.contains)
# countOf = nodeop(operator.countOf)
# delitem = nodeop(operator.delitem)
# eq = nodeop(operator.eq)
# floordiv = nodeop(operator.floordiv)
# ge = nodeop(operator.ge)
# getitem = nodeop(operator.getitem)
# gt = nodeop(operator.gt)
# index = nodeop(operator.index)
# indexOf = nodeop(operator.indexOf)
# inv = nodeop(operator.inv)
# invert = nodeop(operator.invert)
# ior = nodeop(operator.ior)
# ipow = nodeop(operator.ipow)
# irshift = nodeop(operator.irshift)
# is_ = nodeop(operator.is_)
# is_not = nodeop(operator.is_not)
# itemgetter = nodeop(operator.itemgetter)
# le = nodeop(operator.le)
# length_hint = nodeop(operator.length_hint)
# lshift = nodeop(operator.lshift)
# lt = nodeop(operator.lt)
# matmul = nodeop(operator.matmul)
# methodcaller = nodeop(operator.methodcaller)
# mod = nodeop(operator.mod)
# mul = nodeop(operator.mul)
# ne = nodeop(operator.ne)
# neg = nodeop(operator.neg)
# not_ = nodeop(operator.not_)
# or_ = nodeop(operator.or_)
# pos = nodeop(operator.pos)
# rshift = nodeop(operator.rshift)
# setitem = nodeop(operator.setitem)
# sub = nodeop(operator.sub)
# truediv = nodeop(operator.truediv)
# truth = nodeop(operator.truth)
# xor = nodeop(operator.xor)
#
# import_ = nodeop(importlib.import_module)
