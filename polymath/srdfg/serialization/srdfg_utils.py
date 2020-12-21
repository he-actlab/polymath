from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import logging
import numbers
from typing import Text, Sequence, Any, Optional, Union, TypeVar, Callable, Tuple, List, cast

import google.protobuf.message
import numpy as np  # type: ignore
from polymath.pmlang import mapping
from polymath.srdfg.serialization.srdfg_pb2 import Tensor, Attribute, ValueInfo, TensorShape, \
    Component, OperatorSetId, Type
from six import text_type, integer_types, binary_type

logger = logging.getLogger(__name__)


def make_node(
        name: str,
        doc_string: Optional[str]=None,
        domain: Optional[Text]=None,
        **kwargs: Any
) -> Component:
    """Construct a Component.

    Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for Component
        doc_string (string, default None): optional documentation string for Component
        domain (string, default None): optional domain for Component.
            If it's None, we will just use default domain (which is empty)
        **kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
    """
    if 'inputs' not in kwargs:
        logging.critical("Error! No input for {} - {}".format(name, kwargs))

    if 'outputs' not in kwargs:
        logging.critical("Error! No output for {} - {}".format(name, kwargs))

    if 'op' not in kwargs:
        logging.critical("Error! No op for {} - {}".format(name, kwargs))

    node = Component()
    if not kwargs['op']:
        print(kwargs)
        print(name)
    else:
        node.op_type = kwargs['op']
    node.op_cat = kwargs['op_cat']
    node.name = name
    node.input.extend(kwargs['inputs'])
    node.output.extend(kwargs['outputs'])


    if 'states' in kwargs:
        node.state.extend(kwargs['states'])

    if 'params' in kwargs:
        node.parameters.extend(kwargs['params'])

    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if kwargs:
        existing = ['inputs', 'outputs', 'op', 'states', 'params', 'component', 'op_cat']
        for key, value in sorted(kwargs.items()):
            if value is not None and key not in existing:
                node.attributes[key].CopyFrom(make_attribute(key, value))
    return node


def make_operatorsetid(domain: str, version: int)-> OperatorSetId:
    """Construct an OperatorSetId.

    Arguments:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    """
    operatorsetid = OperatorSetId()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetid

def make_statement_graphs(statement_graphs) -> List[Component.StatementGraph]:

    sgraph = []

    for s in statement_graphs:
        sgraph.append(Component.StatementGraph(statement_node=s))
    return sgraph


def split_complex_to_pairs(ca):  # type: (Sequence[np.complex64]) -> Sequence[int]
    return [(ca[i // 2].real if (i % 2 == 0) else ca[i // 2].imag)
            for i in range(len(ca) * 2)]


def make_tensor(
        name,  # type: Text
        data_type,  # type: int
        dims,  # type: Sequence[int]
        vals,  # type: Any
        raw=False  # type: bool
):  # type: (...) -> Tensor
    '''
    Make a Tensor with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.
    '''
    tensor = Tensor()
    tensor.data_type = data_type
    tensor.name = name

    if data_type == Tensor.STRING:
        assert not raw, "Can not use raw_data to store string type"

    if (data_type == Tensor.COMPLEX64
            or data_type == Tensor.COMPLEX128):
        vals = split_complex_to_pairs(vals)
    if raw:
        tensor.raw_data = vals
    else:
        field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
            mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[data_type]]
        getattr(tensor, field).extend(vals)

    tensor.dims.extend(dims)
    return tensor


def _to_bytes_or_false(val):  # type: (Union[Text, bytes]) -> Union[bytes, bool]
    """An internal graph_name to convert the input to a bytes or to False.

    The criteria for conversion is as follows and should be python 2 and 3
    compatible:
    - If val is py2 str or py3 bytes: return bytes
    - If val is py2 unicode or py3 str: return val.decode('utf-8')
    - Otherwise, return False
    """
    if isinstance(val, bytes):
        return val
    else:
        try:
            return val.encode('utf-8')
        except AttributeError:
            return False


def make_attribute(
        key,  # type: Text
        value,  # type: Any
        doc_string=None  # type: Optional[Text]
):  # type: (...) -> Attribute
    """Makes an Attribute based on the value type."""
    attr = Attribute()
    attr.name = key

    if doc_string:
        attr.doc_string = doc_string

    is_iterable = isinstance(value, collections.Iterable)
    bytes_or_false = _to_bytes_or_false(value)


    # First, singular cases
    # float
    if isinstance(value, float):
        attr.f = value
        attr.type = Attribute.FLOAT
    # integer
    elif isinstance(value, numbers.Integral):
        attr.i = cast(int, value)
        attr.type = Attribute.INT
    # string
    elif bytes_or_false:
        assert isinstance(bytes_or_false, bytes)
        attr.s = bytes_or_false
        attr.type = Attribute.STRING
    elif isinstance(value, Tensor):
        attr.t.CopyFrom(value)
        attr.type = Attribute.TENSOR
    elif isinstance(value, Component):
        attr.c.CopyFrom(value)
        attr.type = Attribute.COMPONENT
    # third, iterable cases
    elif is_iterable:
        byte_array = [_to_bytes_or_false(v) for v in value]
        if len(byte_array) == 0:
            attr.ints.extend([])
            attr.type = Attribute.INTS
        elif all(isinstance(v, float) for v in value):
            attr.floats.extend(value)
            attr.type = Attribute.FLOATS
        elif all(isinstance(v, numbers.Integral) for v in value):
            # Turn np.int32/64 into Python built-in int.
            attr.ints.extend(int(v) for v in value)
            attr.type = Attribute.INTS
        elif all(byte_array):
            attr.strings.extend(cast(List[bytes], byte_array))
            attr.type = Attribute.STRINGS
        elif all(isinstance(v, Tensor) for v in value):
            attr.tensors.extend(value)
            attr.type = Attribute.TENSORS
        elif all(isinstance(v, Component) for v in value):
            attr.components.extend(value)
            attr.type = Attribute.COMPONENTS
        else:
            raise ValueError(
                "You passed in an iterable attribute but I cannot figure out "
                "its applicable type.")
    else:
        raise ValueError(
            'Value "{}" is not valid attribute data type.'.format(value))
    return attr


def get_attribute_value(attr):  # type: (Attribute) -> Any
    if attr.type == Attribute.FLOAT:
        return attr.f
    elif attr.type == Attribute.INT:
        return attr.i
    elif attr.type == Attribute.STRING:
        return attr.s.decode("utf-8")
    elif attr.type == Attribute.TENSOR:
        return attr.t
    elif attr.type == Attribute.COMPONENT:
        return attr.c
    elif attr.type == Attribute.FLOATS:
        return list(attr.floats)
    elif attr.type == Attribute.INTS:
        return list(attr.ints)
    elif attr.type == Attribute.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    elif attr.type == Attribute.TENSORS:
        return list(attr.tensors)
    elif attr.type == Attribute.COMPONENTS:
        return list(attr.components)
    else:
        raise ValueError("Unsupported CMStack attribute: {}".format(attr))

def infer_type(inputs, # type: List[ValueInfo]
               output # type: Text
               ):  # type: (...) -> int

    type = inputs[0].type.tensor_type.elem_type
    type_name = inputs[0].name

    for info in inputs:
        if info.type.tensor_type.elem_type != type:
            int_type = mapping.STRING_TEXT_TO_TENSOR_TYPE['int']
            float_type = mapping.STRING_TEXT_TO_TENSOR_TYPE['float']
            if (type,info.type.tensor_type.elem_type) != (int_type, float_type) and (type,info.type.tensor_type.elem_type) != (float_type, int_type):
                logging.warning("Error! Type mismatch between {} and {} for {}".format(type_name, info.name, output))
    return type

def infer_vtype(inputs, # type: List[ValueInfo]
               output # type: Text
               ):  # type: (...) -> int
    assert len(inputs) > 0, "No inputs to infer vtype from for {}".format(output)
    type_map = {
        'scalar': 0,
        'var' : 1,
        'index' : 2
    }
    type = 'scalar'

    for info in inputs:
        vtype = get_attribute_value(info.attributes['vtype'])

        if type_map[vtype] > type_map[type]:
            type = vtype

    return type
def make_edge_info(
        name,  # type: Text
        shape=None,  # type: Optional[Sequence[Union[Text, int]]]
        doc_string="",  # type: Text
        shape_denotation=None,  # type: Optional[List[Text]]
        **kwargs # type: Any
):  # type: (...) -> ValueInfo
    """Makes a ValueInfo based on the data type and shape."""

    edge_info = ValueInfo()

    edge_info.name = name

    if doc_string:
        edge_info.doc_string = doc_string

    tensor_type = edge_info.type.tensor_type

    if 'type' in kwargs:
        if kwargs['type']:
            type_info = kwargs['type']
            if type_info not in mapping.STRING_TEXT_TO_TENSOR_TYPE.keys():
                print(edge_info.name)
            else:
                tensor_type.elem_type = mapping.STRING_TEXT_TO_TENSOR_TYPE[kwargs['type']]

    tensor_shape = tensor_type.shape
    edge_info.src.extend(kwargs['src'])
    edge_info.src.extend(kwargs['dst'])
    if kwargs['vid']:
        edge_info.vid = kwargs['vid']

    if kwargs['iid']:
        edge_info.iid = kwargs['iid']

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        tensor_shape.dim.extend([])

        if shape_denotation:
            if len(shape_denotation) != len(shape):
                raise ValueError(
                    'Invalid shape_denotation. '
                    'Must be of the same length as shape.')

        for i, d in enumerate(shape):
            dim = tensor_shape.dim.add()
            if d is None:
                pass
            elif isinstance(d, integer_types):
                dim.dim_value = d
            elif isinstance(d, text_type):
                dim.dim_param = d
            else:
                raise ValueError(
                    'Invalid item in shape: {}. '
                    'Needs to of integer_types or text_type.'.format(d))

            if shape_denotation:
                dim.denotation = shape_denotation[i]
    if kwargs:
        for key, value in sorted(kwargs.items()):
            if value is not None and key not in ['iid', 'vid']:
                edge_info.attributes[key].CopyFrom(make_attribute(key, value))



    return edge_info


def _sanitize_str(s):  # type: (Union[Text, bytes]) -> Text
    if isinstance(s, text_type):
        sanitized = s
    elif isinstance(s, binary_type):
        sanitized = s.decode('utf-8', errors='ignore')
    else:
        sanitized = str(s)
    if len(sanitized) < 64:
        return sanitized
    else:
        return sanitized[:64] + '...<+len=%d>' % (len(sanitized) - 64)


def printable_attribute(attr, subgraphs=False):  # type: (Attribute, bool) -> Union[Text, Tuple[Text, List[Component]]]
    content = []
    content.append(attr.name)
    content.append("=")

    def str_float(f):  # type: (float) -> Text
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return '{:.15g}'.format(f)

    def str_int(i):  # type: (int) -> Text
        # NB: In Python 2, longs will repr() as '2L', which is ugly and
        # unnecessary.  Explicitly format it to keep it consistent.
        return '{:d}'.format(i)

    def str_str(s):  # type: (Text) -> Text
        return repr(s)

    _T = TypeVar('_T')  # noqa

    def str_list(str_elem, xs):  # type: (Callable[[_T], Text], Sequence[_T]) -> Text
        return '[' + ', '.join(map(str_elem, xs)) + ']'

    # for now, this logic should continue to work as long as we are running on a proto3
    # implementation. If/when we switch to proto3, we will need to use attr.type

    # To support printing subgraphs, if we find a graph_name attribute, print out
    # its name here and pass the graph_name itself up to the caller for later
    # printing.
    graphs = []
    if attr.f:
        content.append(str_float(attr.f))
    elif attr.i:
        content.append(str_int(attr.i))
    elif attr.s:
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(repr(_sanitize_str(attr.s)))
    elif attr.t:
        if len(attr.t.dims) > 0:
            content.append("<Tensor>")
        else:
            # special case to print scalars
            field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[attr.t.data_type]
            content.append('<Scalar Tensor {}>'.format(str(getattr(attr.t, field))))
    elif attr.c:
        content.append("<graph_name {}>".format(attr.g.name))
        graphs.append(attr.g)
    elif attr.floats:
        content.append(str_list(str_float, attr.floats))
    elif attr.ints:
        content.append(str_list(str_int, attr.ints))
    elif attr.strings:
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(str(list(map(_sanitize_str, attr.strings))))
    elif attr.tensors:
        content.append("[<Tensor>, ...]")
    elif attr.components:
        content.append('[')
        for i, g in enumerate(attr.components):
            comma = ',' if i != len(attr.components) - 1 else ''
            content.append('<graph_name {}>{}'.format(g.name, comma))
        content.append(']')
        graphs.extend(attr.components)
    else:
        content.append("<Unknown>")
    if subgraphs:
        return ' '.join(content), graphs
    else:
        return ' '.join(content)


def printable_dim(dim):  # type: (TensorShape.Dimension) -> Text
    which = dim.WhichOneof('value')
    assert which is not None
    return str(getattr(dim, which))


def printable_type(t):  # type: (Type) -> Text
    if t.WhichOneof('value') == "tensor_type":
        s = Tensor.DataType.Name(t.tensor_type.elem_type)
        if t.tensor_type.shape:
            if len(t.tensor_type.shape.dim):
                s += str(', ' + 'x'.join(map(printable_dim, t.tensor_type.shape.dim)))
            else:
                s += str(', scalar')
        return s
    if t.WhichOneof('value') is None:
        return ""
    return 'Unknown type {}'.format(t.WhichOneof('value'))


def printable_value_info(v):  # type: (ValueInfo) -> Text
    s = '%{}'.format(v.name)
    if v.type:
        s = '{}[{}]'.format(s, printable_type(v.type))
    return s


def printable_node(node, prefix='', subgraphs=False):  # type: (Component, Text, bool) -> Union[Text, Tuple[Text, List[Component]]]
    content = []
    if len(node.output):
        content.append(
            ', '.join(['%{}'.format(name) for name in node.output]))
        content.append('=')
    # To deal with nested graphs
    graphs = []  # type: List[Component]
    printed_attrs = []
    for attr in node.attribute:
        if subgraphs:
            printed_attr, gs = printable_attribute(attr, subgraphs)
            assert isinstance(gs, list)
            graphs.extend(gs)
            printed_attrs.append(printed_attr)
        else:
            printed = printable_attribute(attr)
            assert isinstance(printed, Text)
            printed_attrs.append(printed)
    printed_attributes = ', '.join(sorted(printed_attrs))
    printed_inputs = ', '.join(['%{}'.format(name) for name in node.input])
    if node.attribute:
        content.append("{}[{}]({})".format(node.op_type, printed_attributes, printed_inputs))
    else:
        content.append("{}({})".format(node.op_type, printed_inputs))
    if subgraphs:
        return prefix + ' '.join(content), graphs
    else:
        return prefix + ' '.join(content)


def printable_graph(graph, prefix=''):  # type: (Component, Text) -> Text
    content = []
    indent = prefix + '  '
    # header
    header = ['graph_name', graph.name]
    initialized = {t.name for t in graph.initializer}
    if len(graph.input):
        header.append("(")
        in_strs = []
        init_strs = []
        for inp in graph.input:
            if inp.name not in initialized:
                in_strs.append(printable_value_info(inp))
            else:
                init_strs.append(printable_value_info(inp))
        if in_strs:
            content.append(prefix + ' '.join(header))
            header = []
            for line in in_strs:
                content.append(prefix + '  ' + line)
        header.append(")")

        if init_strs:
            header.append("initializers (")
            content.append(prefix + ' '.join(header))
            header = []
            for line in init_strs:
                content.append(prefix + '  ' + line)
            header.append(")")

    header.append('{')
    content.append(prefix + ' '.join(header))
    graphs = []  # type: List[Component]
    # body
    for node in graph.node:
        pn, gs = printable_node(node, indent, subgraphs=True)
        assert isinstance(gs, list)
        content.append(pn)
        graphs.extend(gs)
    # tail
    tail = ['return']
    if len(graph.output):
        tail.append(
            ', '.join(['%{}'.format(out.name) for out in graph.output]))
    content.append(indent + ' '.join(tail))
    # closing bracket
    content.append(prefix + '}')
    for g in graphs:
        content.append('\n' + printable_graph(g))
    return '\n'.join(content)


def strip_doc_string(proto):  # type: (google.protobuf.message.Message) -> None
    """
    Empties `doc_string` field on any nested protobuf messages
    """
    assert isinstance(proto, google.protobuf.message.Message)
    for descriptor in proto.DESCRIPTOR.fields:
        if descriptor.name == 'doc_string':
            proto.ClearField(descriptor.name)
        elif descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                for x in getattr(proto, descriptor.name):
                    strip_doc_string(x)
            elif proto.descriptor.name:
                strip_doc_string(getattr(proto, descriptor.name))











