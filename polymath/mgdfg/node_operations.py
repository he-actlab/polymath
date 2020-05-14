# import functools
# import operator
# import builtins
# from polymath.mgdfg import base, domain
# from polymath.mgdfg.util import _flatten_iterable
#
# class func_op(base.Node):  # pylint: disable=C0103,R0903
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
#             domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
#         elif len(args) == 2:
#             all_args = _flatten_iterable(args)
#             slice1_var, slice1_idx, slice2_var, slice2_idx = self.get_index_nodes(all_args[0], all_args[1])
#             domain = slice1_idx.combine_set_domains(slice2_idx)
#
#         else:
#             domain = domain.Domain(tuple([]))
#         self._target = None
#         super(func_op, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", domain=domain, **kwargs)
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
#         if isinstance(slice1_var, (base.slice_op, base.var_index)) or _is_node_type_instance(slice1_var, "GroupNode"):
#             slice1_idx = slice1_var.domain
#         else:
#             slice1_idx = Domain(tuple([]))
#
#         if isinstance(slice2_var, (slice_op, var_index)) or _is_node_type_instance(slice2_var, "GroupNode"):
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