# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ndarray.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ndarray.proto',
  package='numproto.protobuf',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rndarray.proto\x12\x11numproto.protobuf\"\x1a\n\x07NDArray\x12\x0f\n\x07ndarray\x18\x01 \x01(\x0c\x62\x06proto3'
)




_NDARRAY = _descriptor.Descriptor(
  name='NDArray',
  full_name='numproto.protobuf.NDArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ndarray', full_name='numproto.protobuf.NDArray.ndarray', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=36,
  serialized_end=62,
)

DESCRIPTOR.message_types_by_name['NDArray'] = _NDARRAY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NDArray = _reflection.GeneratedProtocolMessageType('NDArray', (_message.Message,), {
  'DESCRIPTOR' : _NDARRAY,
  '__module__' : 'ndarray_pb2'
  # @@protoc_insertion_point(class_scope:numproto.protobuf.NDArray)
  })
_sym_db.RegisterMessage(NDArray)


# @@protoc_insertion_point(module_scope)
