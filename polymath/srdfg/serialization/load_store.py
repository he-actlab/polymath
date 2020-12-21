import os
import uuid
from itertools import chain
from typing import Iterable, Text, Optional
from typing import Union, IO, cast, TypeVar, Any

import google.protobuf.message
from polymath.srdfg.serialization.srdfg_pb2 import *
from six import string_types


# f should be either readable or a file path
def _load_bytes(f: Union[IO[bytes], Text]) -> bytes:
    if hasattr(f, 'read') and callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(Text, f), 'rb') as readable:
            s = readable.read()
    return s
# str should be bytes,
# f should be either writable or a file path
def _save_bytes(str, f):  # type: (bytes, Union[IO[bytes], Text]) -> None
    if hasattr(f, 'write') and callable(cast(IO[bytes], f).write):
        cast(IO[bytes], f).write(str)
    else:
        with open(cast(Text, f), 'wb') as writable:
            writable.write(str)


# f should be either a readable file or a file path
def _get_file_path(f):  # type: (Union[IO[bytes], Text]) -> Optional[Text]
    if isinstance(f, string_types):
        return os.path.abspath(f)
    if hasattr(f, 'name'):
        return os.path.abspath(f.name)
    return None


def _serialize(proto):  # type: (Union[bytes, google.protobuf.message.Message]) -> bytes
    '''
    Serialize a in-memory proto to bytes
    @params
    proto is a in-memory proto, such as a Program, Tensor, etc
    @return
    Serialized proto in bytes
    '''
    if isinstance(proto, bytes):
        return proto
    elif hasattr(proto, 'SerializeToString') and callable(proto.SerializeToString):
        result = proto.SerializeToString()
        return result
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither proto is a str.\ntype is {}'.format(type(proto)))


_Proto = TypeVar('_Proto', bound=google.protobuf.message.Message)


def _deserialize(s, proto):  # type: (bytes, _Proto) -> _Proto
    '''
    Parse bytes into a in-memory proto
    @params
    s is bytes containing serialized proto
    proto is a in-memory proto object
    @return
    The proto instance filled in by s
    '''
    if not isinstance(s, bytes):
        raise ValueError('Parameter s must be bytes, but got type: {}'.format(type(s)))

    if not (hasattr(proto, 'ParseFromString') and callable(proto.ParseFromString)):
        raise ValueError('No ParseFromString method is detected. '
                         '\ntype is {}'.format(type(proto)))

    decoded = cast(Optional[int], proto.ParseFromString(s))
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return proto


def load_program(f, format=None, load_external_data=True):  # type: (Union[IO[bytes], Text], Optional[Any], bool) -> Program
    '''
    Loads a serialized Program into memory
    @params
    f can be a file-like object (has "read" function) or a string containing a file name
    format is for future use
    @return
    Loaded in-memory Program
    '''
    s = _load_bytes(f)
    program = load_program_from_string(s, format=format)

    if load_external_data:
        program_filepath = _get_file_path(f)
        if program_filepath:
            base_dir = os.path.dirname(program_filepath)
            load_external_data_for_program(program, base_dir)

    return program


def load_tensor(f, format=None):  # type: (Union[IO[bytes], Text], Optional[Any]) -> Tensor
    '''
    Loads a serialized Tensor into memory
    @params
    f can be a file-like object (has "read" function) or a string containing a file name
    format is for future use
    @return
    Loaded in-memory Tensor
    '''
    s = _load_bytes(f)
    return load_tensor_from_string(s, format=format)


def load_program_from_string(s, format=None):  # type: (bytes, Optional[Any]) -> Program
    '''
    Loads a binary string (bytes) that contains serialized Program
    @params
    s is a string, which contains serialized Program
    format is for future use
    @return
    Loaded in-memory Program
    '''
    return _deserialize(s, Program())


def load_tensor_from_string(s, format=None):  # type: (bytes, Optional[Any]) -> Tensor
    '''
    Loads a binary string (bytes) that contains serialized Tensor
    @params
    s is a string, which contains serialized Tensor
    format is for future use
    @return
    Loaded in-memory Tensor
    '''
    return _deserialize(s, Tensor())


def save_program(proto, f, format=None):  # type: (Union[Program, bytes], Union[IO[bytes], Text], Optional[Any]) -> None
    '''
    Saves the Program to the specified path.
    @params
    proto should be a in-memory Program
    f can be a file-like object (has "write" function) or a string containing a file name
    format is for future use
    '''
    if isinstance(proto, bytes):
        proto = _deserialize(proto, Program())

    program_filepath = _get_file_path(f)
    if program_filepath:
        basepath = os.path.dirname(program_filepath)
        proto = write_external_data_tensors(proto, basepath)

    s = _serialize(proto)
    _save_bytes(s, f)


def save_tensor(proto, f):  # type: (Tensor, Union[IO[bytes], Text]) -> None
    '''
    Saves the Tensor to the specified path.
    @params
    proto should be a in-memory Tensor
    f can be a file-like object (has "write" function) or a string containing a file name
    format is for future use
    '''
    s = _serialize(proto)
    _save_bytes(s, f)





class ExternalDataInfo(object):

    def __init__(self, tensor):  # type: (Tensor) -> None
        self.location = ''
        self.offset = None
        self.length = None
        self.checksum = None
        self.basepath = ''

        for entry in tensor.external_data:
            setattr(self, entry.key, entry.value)

        if self.offset:
            self.offset = int(self.offset)

        if self.length:
            self.length = int(self.length)


def load_external_data_for_tensor(tensor, base_dir):  # type: (Tensor, Text) -> None
    """
    Load data from an external file for tensor.
    @params
    tensor: a Tensor object.
    base_dir: directory that contains the external data.
    """
    if tensor.HasField("raw_data"):  # already loaded
        return
    info = ExternalDataInfo(tensor)
    file_location = _sanitize_path(info.location)
    external_data_file_path = os.path.join(base_dir, file_location)

    with open(external_data_file_path, 'rb') as data_file:

        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            tensor.raw_data = data_file.read(info.length)
        else:
            tensor.raw_data = data_file.read()


def load_external_data_for_program(program, base_dir):  # type: (Program, Text) -> None
    """
    Loads external tensors into program
    @params
    program: Program to load external data to
    base_dir: directory that contains external data
    """
    for tensor in _get_all_tensors(program):
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)


def set_external_data(tensor,  # type: Tensor
                      location,  # type: Text
                      offset=None,  # type: Optional[int]
                      length=None,  # type: Optional[int]
                      checksum=None,  # type: Optional[Text]
                      basepath=None  # type: Optional[Text]
                      ):  # type: (...) -> None
    del tensor.external_data[:]
    tensor.data_location = Tensor.EXTERNAL
    for (k, v) in {
        'location': location,
        'offset': int(offset) if offset is not None else None,
        'length': int(length) if length is not None else None,
        'checksum': checksum,
        'basepath': basepath
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def convert_program_to_external_data(program, all_tensors_to_one_file=True, location=None):
    # type: (Program, bool, Optional[Text]) -> None
    """
    call to set all tensors as external data. save_program saves all the tensors data as external data after calling this function.
    @params
    program: Program to be converted.
    all_tensors_to_one_file: If true, save all tensors to one external file specified by location.
                             If false, save each tensor to a file named with the tensor name.
    location: specify the external file that all tensors to save to.
              If not specified, will use the program name.
    """
    if all_tensors_to_one_file:
        file_name = Text(uuid.uuid1())
        if location:
            file_name = location
        for tensor in _get_all_tensors(program):
            set_external_data(tensor, file_name)
    else:
        for tensor in _get_all_tensors(program):
            set_external_data(tensor, tensor.name)


def convert_program_from_external_data(program):  # type: (Program) -> None
    """
    call to set all tensors data as embedded data. save_program saves all the tensors data as embedded data after calling this function.
    @params
    program: Program to be converted.
    """
    for tensor in _get_all_tensors(program):
        if uses_external_data(tensor):
            if not tensor.HasField("raw_data"):
                raise ValueError("raw_data field doesn't exist.")
            del tensor.external_data[:]
            tensor.data_location = Tensor.DEFAULT


def save_external_data(tensor, base_path):  # type: (Tensor, Text) -> None
    """
    Write tensor data to an external file according to information in the `external_data` field.
    @params
    tensor: Tensor object to be serialized
    base_path: System path of a folder where tensor data is to be stored
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = os.path.join(base_path, info.location)

    # Retrieve the tensor's data from raw_data or load external file
    if not tensor.HasField("raw_data"):
        raise ValueError("raw_data field doesn't exist.")

    # Create file if it doesn't exist
    if not os.path.isfile(external_data_file_path):
        open(external_data_file_path, 'ab').close()

    # Open file for reading and writing at random locations ('r+b')
    with open(external_data_file_path, 'r+b') as data_file:
        data_file.seek(0, 2)
        if info.offset is not None:
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b"\0" * (info.offset - file_size))

            data_file.seek(info.offset)
        offset = data_file.tell()
        data_file.write(tensor.raw_data)
        set_external_data(tensor, info.location, offset, data_file.tell() - offset)


def _get_all_tensors(polymath_program_proto):  # type: (Program) -> Iterable[Tensor]
    """Scan an ONNX program for all tensors and return as an iterator."""
    return chain(_get_initializer_tensors(polymath_program_proto),
                 _get_attribute_tensors(polymath_program_proto))


def _get_initializer_tensors(polymath_program_proto):  # type: (Program) -> Iterable[Tensor]
    """Create an iterator of initializer tensors from ONNX program."""
    for initializer in polymath_program_proto.graph.initializer:
        yield initializer


def _get_attribute_tensors(polymath_program_proto):  # type: (Program) -> Iterable[Tensor]
    """Create an iterator of tensors from node attributes of an ONNX program."""
    for node in polymath_program_proto.graph.sub_graph:
        for attribute in node.attributes:
            if node.attributes[attribute].t:
                yield node.attributes[attribute].t
            for tensor in node.attributes[attribute].tensors:
                yield tensor


def _sanitize_path(path):  # type: (Text) -> Text
    """Remove path components which would allow traversing up a directory tree from a base path.
    Note: This method is currently very basic and should be expanded.
    """
    return path.lstrip('/.')


def uses_external_data(tensor):  # type: (Tensor) -> bool
    """Return true if the tensor stores data in an external location."""
    if tensor.data_location:
        has_dl = True
    else:
        has_dl = False

    if tensor.data_location == Tensor.EXTERNAL:
        is_ext = True
    else:
        is_ext = False

    return has_dl and is_ext


def remove_external_data_field(tensor, field_key):  # type: (Tensor, Text) -> None
    """
    Remove a field from a Tensor's external_data key-value store.
    Modifies tensor object in place.
    @params
    tensor: Tensor object from which value will be removed
    field_key: The key of the field to be removed
    """
    for i in tensor.external_data:
        if i == field_key:
            del tensor.external_data[i]


def write_external_data_tensors(program, filepath):  # type: (Program, Text) -> Program
    """
    Write external data of all tensors to files on disk.
    Note: This function also strips basepath information from all tensors' external_data fields.
    @params
    program: Model object which is the source of tensors to serialize.
    filepath: System path to the directory which should be treated as base path for external data.
    @return
    The modified program object.
    """
    for tensor in _get_all_tensors(program):
        if uses_external_data(tensor):
            save_external_data(tensor, filepath)
            tensor.ClearField(str('raw_data'))

    return program

