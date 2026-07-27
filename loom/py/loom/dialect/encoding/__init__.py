# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Encoding dialect: encoding definition and query ops.

Encodings describe the physical data layout (quantization, packing, etc.)
of tile and tensor types and the address layout of view types. The encoding
dialect provides ops to create and query encoding values as first-class SSA
values, enabling encodings/layouts that compose across library boundaries.

A model author picks an encoding in one top-level file and it propagates
through hundreds of library .loom files unchanged — function signatures
reference %enc rather than #q8_0<block=32>.
"""

from loom.dialect.encoding.defs import (
    ALL_ENCODING_OPS,
    encoding_assume_spec,
    encoding_define,
    encoding_isa,
    encoding_layout_assume_dense,
    encoding_layout_assume_strided,
    encoding_layout_dense,
    encoding_layout_strided,
    encoding_ops,
)

__all__ = [
    "encoding_ops",
    "ALL_ENCODING_OPS",
    "encoding_layout_dense",
    "encoding_layout_strided",
    "encoding_layout_assume_dense",
    "encoding_layout_assume_strided",
    "encoding_define",
    "encoding_isa",
    "encoding_assume_spec",
]
