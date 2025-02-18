from typing import Tuple, Any, Union, Sequence
from polymath import DEFAULT_SHAPES
from polymath.srdfg.util import is_iterable, _is_node_type_instance, _is_node_instance
from dataclasses import dataclass, field
import numpy as np
from numbers import Integral
from itertools import product, groupby
from operator import itemgetter
import time
from collections import defaultdict
from functools import reduce

dom_fields = ("doms", "names")

@dataclass(unsafe_hash=True)
class Domain(object):
    doms: Tuple[Any] = field(hash=False)
    einstein_notation: Sequence = field(default=None, hash=False)
    dom_set: Tuple[Any] = field(default=None, hash=False)
    names: Tuple[str] = field(init=False)
    computed: dict = field(default=None, hash=False)
    computed_pairs: Sequence = field(default=None, hash=False, repr=False)
    computed_set_pairs: Sequence = field(default=None, hash=False, repr=False)

    def __post_init__(self):
        if is_iterable(self.doms):
            self.doms = tuple(self.doms)
        else:
            self.doms = tuple([self.doms])
        names = []
        for d in self.doms:
            if _is_node_instance(d):
                names.append(d.name)
            else:
                names.append(d)
        if not self.dom_set:
            dset = []

            for a in self.doms:
                if _is_node_type_instance(a, "index_op"):
                    dset += [i for i in a.domain.dom_set]
                elif _is_node_type_instance(a, "index"):
                    dset += [i for i in a.domain.dom_set]
                elif _is_node_instance(a):
                    dset += [i for i in a.domain.dom_set]
                elif isinstance(a, Domain):
                    dset += a.doms
                elif not isinstance(a, Integral):
                    raise RuntimeError(f"Invalid domain type for domain:\n"
                                       f"Dom: {a}\n"
                                       f"Names: {names}")
                else:
                    dset.append(a)
            self.dom_set = tuple(dset)
        self.names = tuple(names)

    def set_computed(self, in_shape, indices):
        nindices = tuple([np.arange(in_shape[i]) for i in range(len(in_shape))])
        nindices = np.array(list(product(*nindices)))
        assert len(nindices) == len(indices)
        self.computed = {}
        for i, idx in enumerate(indices):
            self.computed[idx] = nindices[i]

    def __len__(self):
        return len(self.dom_set)

    def __iter__(self):
        for i in self.dom_set:
            yield i

    def index(self, o):
        return self.dom_set.index(o)

    @property
    def is_scalar(self):
        return len(self.dom_set) == 0 or self.doms == DEFAULT_SHAPES[0]

    def compute_dom_set(self):
        dset = []
        for a in self.doms:
            if _is_node_type_instance(a, "index_op"):
                dset += [i for i in a.domain.dom_set]
            elif _is_node_type_instance(a, "index"):
                dset.append(a)
            elif _is_node_instance(a):
                dset += [i for i in a.domain.dom_set]
            else:
                assert isinstance(a, Integral)
                dset.append(a)
        return tuple(dset)

    def reduction_domain(self, r_dom):
        res = tuple(sorted(set(self.dom_set).difference(r_dom), key=self.dom_set.index))
        return Domain(res)

    def combine_domains(self, dom):
        assert isinstance(dom, Domain)
        all_keys = self.doms + dom.doms
        unique_keys = list(sorted(set(all_keys), key=all_keys.index))
        return Domain(unique_keys)

    def set_einstein_repr(self, dom):

        cnt = 0
        op1 = []
        op2 = []
        all_ops = []

        for i in self.set_names:
            if isinstance(i, Integral):
                op1.append("1")
                all_ops.append("1")
            else:
                op1.append(f"d{cnt}")
                all_ops.append(f"d{cnt}")
                cnt += 1

        for i in dom.set_names:
            if i in self.set_names:
                idx = self.set_names.index(i)
                op2.append(op1[idx])
            elif isinstance(i, Integral):
                op2.append("1")
                all_ops.append("1")
            else:
                op2.append(f"d{cnt}")
                all_ops.append(f"d{cnt}")

                cnt += 1
        self.einstein_notation = {"in": [op1, op2], "out": [all_ops]}


    def combine_set_domains(self, dom):
        assert isinstance(dom, Domain)
        self.set_einstein_repr(dom)
        all_keys = self.dom_set + dom.dom_set
        unique_keys = list(sorted(set(all_keys), key=all_keys.index))
        return Domain(unique_keys)

    @property
    def set_names(self):
        names = []
        for d in self.dom_set:
            if _is_node_instance(d):
                names.append(d.name)
            else:
                names.append(d)
        return tuple(names)

    @property
    def ndims(self):
        return len(self.doms)

    def compute_set_pairs_from_idx(self, indices, tuples=True):

        dom_pairs = []
        for i in indices:
            if _is_node_instance(i):
                if i.value is not None:
                    dom_pairs.append(i.value)

                elif _is_node_type_instance(i, "index"):
                    assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                    dom_pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                else:
                    raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
            elif isinstance(i, np.ndarray):
                dom_pairs.append(i.tolist())
            else:
                assert isinstance(i, list)
                dom_pairs.append(i)

        dom_pairs = tuple(dom_pairs)
        dom_pairs = np.array(list(product(*dom_pairs)))

        if tuples:
            dom_pairs = [tuple(i) for i in dom_pairs]
        return dom_pairs

    def compute_set_pairs(self, dom=None, tuples=True):
        if self.computed_set_pairs is not None:
            dom_pairs = self.computed_set_pairs
        else:
            dom_pairs = []
            if self.dom_set == DEFAULT_SHAPES[0]:
                dom_pairs.append([0])
            else:
                for n, i in enumerate(self.dom_set):
                    if _is_node_instance(i):
                        if i.value is not None:
                            dom_pairs.append(i.value)

                        elif _is_node_type_instance(i, "index"):
                            assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                            dom_pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                        else:
                            raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
                    elif isinstance(i, np.ndarray):
                        dom_pairs.append(i.tolist())
                    elif isinstance(i, Integral):
                        dom_pairs.append([i])
                    else:
                        assert isinstance(i, list)
                        dom_pairs.append(i)

            dom_pairs = tuple(dom_pairs)
            dom_pairs = np.array(list(product(*dom_pairs)))
            self.computed_set_pairs = dom_pairs

        if tuples:
            dom_pairs = [tuple(i) for i in dom_pairs]
        return dom_pairs


    def compute_shape_domain(self, indices=None):
        if indices:
            dom_pairs = []
            for i in indices:
                if _is_node_instance(i):
                    dom_pairs.append(i.value)
                elif isinstance(i, Integral):
                    dom_pairs.append((i,))
                else:
                    dom_pairs.append(i)

            # dom_pairs = np.asarray([i.value if _is_node_instance(i) else i for i in indices])
            dom_pairs = np.asarray(dom_pairs)
        else:
            dom_pairs = []
            for i in self.doms:
                if _is_node_instance(i):
                    if i.value is not None:
                        dom_pairs.append(np.arange(0, i.value, dtype=np.int))
                    elif _is_node_type_instance(i, "index"):
                        assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                        dom_pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                    elif i.shape in DEFAULT_SHAPES:
                        continue
                    else:
                        raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
                elif isinstance(i, np.ndarray):
                    dom_pairs.append(i.tolist())
                elif isinstance(i, Integral):
                    dom_pairs.append(np.arange(0, i, dtype=np.int))
                else:
                    assert isinstance(i, list)
                    dom_pairs.append(i)
        dom_pairs = tuple(dom_pairs)
        dom_pairs = np.array(list(product(*dom_pairs)))
        dom_pairs = [tuple(i) for i in dom_pairs]
        return dom_pairs

    def compute_pairs(self, tuples=True, squeeze=False):
        if self.computed_pairs is not None:
            pairs = self.computed_pairs
        else:
            pairs = []
            if self.doms == DEFAULT_SHAPES[0]:
                pairs.append([0])
            else:
                for i in self.doms:
                    if _is_node_instance(i):

                        if i.value is not None and is_iterable(i.value):
                            pairs.append(i.value)
                        elif _is_node_type_instance(i, "index"):
                            assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                            pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                        elif i.shape in DEFAULT_SHAPES:
                            continue
                        else:
                            raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
                    elif isinstance(i, np.ndarray):
                        pairs.append(i.tolist())
                    elif isinstance(i, Integral):
                        pairs.append([i])
                    else:
                        assert isinstance(i, list)
                        pairs.append(i)
            pairs = tuple(pairs)
            pairs = np.array(list(product(*pairs)))
            self.computed_pairs = pairs

        if squeeze:
            pairs = pairs[:, ~np.all(pairs == 0, axis=0)]

        if tuples:
            pairs = list(map(lambda x: tuple(x), pairs))
        return pairs

    def map_index_op_domains(self, indices, tuples=True):
        if self.computed_pairs is not None:
            pairs = self.computed_pairs
        else:
            pairs = []
            if self.doms == DEFAULT_SHAPES[0]:
                pairs.append([0])
            else:
                for i in self.doms:
                    if _is_node_instance(i):

                        if i.value is not None and is_iterable(i.value):
                            pairs.append(i.value)
                        elif _is_node_type_instance(i, "index"):
                            assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                            pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                        elif i.shape in DEFAULT_SHAPES:
                            continue
                        else:
                            raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
                    elif isinstance(i, np.ndarray):
                        pairs.append(i.tolist())
                    elif isinstance(i, Integral):
                        pairs.append([i])
                    else:
                        assert isinstance(i, list)
                        pairs.append(i)
            pairs = tuple(pairs)
            pairs = np.array(list(product(*pairs)))
            self.computed_pairs = pairs

        if tuples:
            pairs = list(map(lambda x: tuple(x), pairs))
        return pairs

    def compute_index_pairs(self, tuples=True):
        pairs = []
        dom_set_pairs = self.compute_set_pairs(tuples=False)
        unraveled_set_pairs = np.ravel_multi_index(dom_set_pairs.T, self.computed_set_shape)
        self.computed_pairs = unraveled_set_pairs.reshape(np.prod(self.computed_set_shape), 1)
        return self.computed_pairs

    def map_sub_domain(self, dom, is_index_dom=False, is_write=False, tuples=True, do_print=False):

        dom_set_pairs = self.compute_set_pairs(tuples=False)
        target_set_pairs = dom.compute_set_pairs(dom=dom, tuples=False)
        target_pairs = dom.compute_pairs(tuples=False)

        if dom.computed:
            target_pairs = np.asarray([dom.computed[tuple(x)] for x in target_pairs])

        if dom_set_pairs.shape[-1] < target_set_pairs.shape[-1]:
            idx = np.argwhere(np.all(target_set_pairs[..., :] == 0, axis=0))
            target_set_pairs = np.delete(target_set_pairs, idx, axis=1)

        if dom_set_pairs.shape[-1] < target_pairs.shape[-1]:
            idx = np.argwhere(np.all(target_pairs[..., :] == 0, axis=0))
            target_pairs = np.delete(target_pairs, idx, axis=1)


        idx_map = np.asarray([self.set_names.index(n) for n in dom.set_names if n in self.set_names], dtype=np.int)
        pair_mappings = np.apply_along_axis(lambda x: x[idx_map], 1, dom_set_pairs)
        dims = target_set_pairs.max(0) + 1

        X1D = np.ravel_multi_index(target_set_pairs.T, dims)
        searched_valuesID = np.ravel_multi_index(pair_mappings.T, dims)
        sidx = X1D.argsort()
        out = sidx[np.searchsorted(X1D, searched_valuesID, sorter=sidx)]
        out = np.apply_along_axis(lambda x: target_pairs[x], 0, out)

        if tuples:
            out = list(map(lambda x: tuple(x), out))

        return out

    # ref: https://stackoverflow.com/questions/49964765/for-each-row-of-a-2d-numpy-array-get-the-index-of-an-equal-row-in-a-second-2d-ar
    def map_reduction_dom(self, input_dom, axis_idx):

        dom_set_pairs = input_dom.compute_set_pairs(tuples=False)

        target_set_pairs = self.compute_set_pairs(tuples=False)
        target_pairs = self.compute_pairs(tuples=False)
        if self.computed:
            target_pairs = np.asarray([self.computed[tuple(x)] for x in target_pairs])

        pair_mappings = np.apply_along_axis(lambda x: x[axis_idx], 1, dom_set_pairs)
        dims = target_set_pairs.max(0) + 1
        X1D = np.ravel_multi_index(target_set_pairs.T, dims)
        searched_valuesID = np.ravel_multi_index(pair_mappings.T, dims)
        sidx = X1D.argsort()
        out = sidx[np.searchsorted(X1D, searched_valuesID, sorter=sidx)]
        out = np.apply_along_axis(lambda x: target_pairs[x], 0, out)
        out = list(map(lambda x: tuple(x), out))
        if input_dom.computed_set_shape != input_dom.computed_shape:
            dom_set_pairs = np.asarray(np.unravel_index(np.ravel_multi_index(dom_set_pairs.T, input_dom.computed_set_shape), input_dom.computed_shape)).T

        dom_set_pairs = list(map(lambda x: tuple(x), dom_set_pairs))
        out = sorted(list(zip(out, dom_set_pairs)), key=lambda x: (x[0],x[1]))
        mr_out = dict((k, [v[1] for v in itr]) for k, itr in groupby(
                                out, itemgetter(0)))
        return mr_out

    def compute_axes_index(self, dom, group_ops=False):
        names = [i.name if _is_node_instance(i) else i for i in dom.doms]
        axes = []
        for n in names:
            if n in self.names:
                axes.append(self.names.index(n))
            elif group_ops:
                op_dom = dom.doms[dom.names.index(n)]
                axes.append(tuple([self.names.index(d.name) for d in op_dom.domain]))
            else:
                op_dom = dom.doms[dom.names.index(n)]
                if _is_node_instance(op_dom):
                    axes += [self.names.index(d.name) for d in op_dom.domain]
                else:
                    axes.append(op_dom)

        return tuple(axes)

    def compute_reduction_index(self, dom):
        diff = list(set(self.names).difference(dom.names))
        out = tuple([self.names.index(n) for n in diff])
        return out

    def compute_set_reduction_index(self, dom):
        diff = list(set(self.set_names).difference(dom.set_names))
        out = tuple([self.set_names.index(n) for n in diff])
        return out

    @property
    def computed_set_shape(self):
        return tuple(np.max(self.compute_set_pairs(), axis=0) + 1)

    @property
    def computed_shape(self):
        return tuple(np.max(self.compute_pairs(), axis=0) + 1)

    def shape_from_indices(self, idx_vals):
        shape = []
        for i in idx_vals:
            if _is_node_instance(i):
                shape.append(len(i.value))
            elif is_iterable(i):
                shape.append(len(i))
            else:
                shape.append(1)
        return tuple(shape)

