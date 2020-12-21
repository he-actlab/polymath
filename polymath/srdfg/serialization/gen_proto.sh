#!/usr/bin/env bash
protoc -I./ --python_out=./ srdfgv3.proto ndarray.proto