from typing import Tuple, Any, Union, Sequence
from polymath.mgdfg.util import is_iterable, _is_node_type_instance, _is_node_instance
from dataclasses import dataclass, field
import numpy as np
from numbers import Integral
from itertools import product
from functools import reduce

dom_fields = ("doms", "names")

@dataclass(unsafe_hash=True)
class Domain(object):
    doms: Tuple[Any] = field(hash=False)
    names: Tuple[str] = field(init=False)
    computed: dict = field(default=None, hash=False)
    computed_pairs: Sequence = field(default=None, hash=False)
    computed_set_pairs: Sequence = field(default=None, hash=False)

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
    def dom_set(self):
        dset = []
        for a in self.doms:
            if _is_node_type_instance(a, "index"):
                dset += [i for i in a.domain]
            elif _is_node_instance(a):
                dset += a.domain.dom_set
        return tuple(dset)

    def reduction_domain(self, r_dom):
        res = tuple(sorted(set(self.dom_set).difference(r_dom), key=self.dom_set.index))
        return Domain(res)

    def combine_domains(self, dom):
        assert isinstance(dom, Domain)
        all_keys = self.doms + dom.doms
        unique_keys = list(sorted(set(all_keys), key=all_keys.index))
        return Domain(unique_keys)

    def combine_set_domains(self, dom):
        assert isinstance(dom, Domain)
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

    def compute_set_pairs(self, tuples=True):
        if self.computed_set_pairs is not None:
            dom_pairs = self.computed_set_pairs
        else:
            dom_pairs = []
            for i in self.dom_set:
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
            self.computed_set_pairs = dom_pairs

        if tuples:
            dom_pairs = [tuple(i) for i in dom_pairs]
        return dom_pairs


    def compute_shape_domain(self, indices=None):
        if indices:
            dom_pairs = np.asarray([i.value if _is_node_instance(i) else i for i in indices])
        else:
            dom_pairs = []
            for i in self.doms:
                if _is_node_instance(i):
                    if i.value is not None:
                        dom_pairs.append(np.arange(0, i.value, dtype=np.int))
                    elif _is_node_type_instance(i, "index"):
                        assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                        dom_pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                    elif i.shape == (0,) or i.shape == (1,):
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

    def compute_pairs(self, tuples=True):
        if self.computed_pairs is not None:
            pairs = self.computed_pairs
        else:
            pairs = []

            for i in self.doms:
                if _is_node_instance(i):

                    if i.value is not None and is_iterable(i.value):
                        pairs.append(i.value)
                    elif _is_node_type_instance(i, "index"):
                        assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                        pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                    elif i.shape == (0,) or i.shape == (1,):
                        continue
                    else:
                        raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
                elif isinstance(i, np.ndarray):
                    pairs.append(i.tolist())
                elif isinstance(i, Integral):
                    continue
                else:
                    assert isinstance(i, list)
                    pairs.append(i)
            pairs = tuple(pairs)
            pairs = np.array(list(product(*pairs)))
            self.computed_pairs = pairs

        if tuples:
            pairs = list(map(lambda x: tuple(x), pairs))
        return pairs

    def compute_index_pairs(self, indices, tuples=True):
        pairs = []
        for i in indices:

            if _is_node_instance(i):
                if i.value is not None and is_iterable(i.value):
                    pairs.append(i.value)
                elif _is_node_type_instance(i, "index"):
                    assert isinstance(i.lbound, Integral) and isinstance(i.ubound, Integral)
                    pairs.append([x for x in range(i.lbound, i.ubound + 1)])
                elif i.shape == (0,) or i.shape == (1,):
                    continue
                else:
                    raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
            elif isinstance(i, np.ndarray):
                pairs.append(i.tolist())
            elif isinstance(i, Integral):
                continue
            else:
                assert isinstance(i, list)
                pairs.append(i)
        pairs = tuple(pairs)
        pairs = np.array(list(product(*pairs)))

        if tuples:
            pairs = [tuple(i) for i in pairs]
        return pairs


    def map_sub_domain(self, dom):

        dom_set_pairs = self.compute_set_pairs(tuples=False)
        target_set_pairs = dom.compute_set_pairs(tuples=False)
        target_pairs = dom.compute_pairs(tuples=False)
        if dom.computed:
            target_pairs = np.asarray([dom.computed[tuple(x)] for x in target_pairs])
        idx_map = np.asarray([self.set_names.index(n) for n in dom.set_names if n in self.set_names], dtype=np.int)
        pair_mappings = np.apply_along_axis(lambda x: x[idx_map], 1, dom_set_pairs)
        dims = target_set_pairs.max(0) + 1
        X1D = np.ravel_multi_index(target_set_pairs.T, dims)
        searched_valuesID = np.ravel_multi_index(pair_mappings.T, dims)
        sidx = X1D.argsort()
        out = sidx[np.searchsorted(X1D, searched_valuesID, sorter=sidx)]
        out = np.apply_along_axis(lambda x: target_pairs[x], 0, out)
        out = list(map(lambda x: tuple(x), out))

        return out

    def get_filtered_indices(self, superset, target_axes, axes):
        pass


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

    def shape_from_indices(self, indices):
        return tuple(len(i.value) if _is_node_instance(i) else len(i) for i in indices)


