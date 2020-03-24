#!/usr/bin/env bash
protoc -I./ --python_out=./ mgdfgv3.proto ndarray.proto