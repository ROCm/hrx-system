# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete source IR registry and input loading for C++ rendering."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Any

import loom
from loom.dialect.config import ALL_CONFIG_OPS
from loom.dialect.encoding import (
    ALL_ENCODING_FAMILIES,
    ALL_ENCODING_PARAMETERIZED_ATTRS,
)
from loom.dialect.func import ALL_FUNC_PARAMETERIZED_ATTRS
from loom.dialect.target import ALL_TARGET_PARAMETERIZED_ATTRS
from loom.dialect.template import ALL_TEMPLATE_OPS
from loom.dialect.test import ALL_TEST_PARAMETERIZED_ATTRS
from loom.dsl import Op, TypeDef
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import MAGIC
from loom.format.text.parser import Parser
from loom.ir import Module
from loom.target.arch.amd.xdna.aie2p.dialect import ALL_AIE2P_OPS
from loom.target.arch.amdgpu.dialect import ALL_AMDGPU_OPS
from loom.target.arch.spirv.dialect import ALL_SPIRV_OPS, ALL_SPIRV_TYPES
from loom.target.arch.vm.dialect import ALL_VM_OPS, ALL_VM_TYPES
from loom.target.arch.wasm.dialect import ALL_WASM_OPS
from loom.target.arch.x86.dialect import ALL_X86_OPS


def _unique_by_name[T](groups: Iterable[Iterable[T]]) -> tuple[T, ...]:
    values: dict[str, T] = {}
    for group in groups:
        for value in group:
            name = value.name
            previous = values.setdefault(name, value)
            if previous is not value and previous != value:
                raise ValueError(f"conflicting source declaration for {name!r}")
    return tuple(values.values())


@cache
def source_ops() -> tuple[Op, ...]:
    """Returns every operation accepted at the readable source boundary."""

    return _unique_by_name(
        (
            loom.default_ops(),
            ALL_CONFIG_OPS,
            ALL_TEMPLATE_OPS,
            ALL_AIE2P_OPS,
            ALL_AMDGPU_OPS,
            ALL_SPIRV_OPS,
            ALL_VM_OPS,
            ALL_WASM_OPS,
            ALL_X86_OPS,
        )
    )


@cache
def source_types() -> tuple[TypeDef, ...]:
    """Returns every type accepted at the readable source boundary."""

    return _unique_by_name((loom.default_types(), ALL_SPIRV_TYPES, ALL_VM_TYPES))


@cache
def source_parameterized_attrs() -> tuple[Any, ...]:
    """Returns parameterized attributes accepted in source modules."""

    return _unique_by_name(
        (
            ALL_ENCODING_PARAMETERIZED_ATTRS,
            ALL_FUNC_PARAMETERIZED_ATTRS,
            ALL_TARGET_PARAMETERIZED_ATTRS,
            ALL_TEST_PARAMETERIZED_ATTRS,
        )
    )


def load_module(data: bytes, *, filename: str = "<input>") -> Module:
    """Loads and verifies Loom text or bytecode from one immutable payload."""

    operations = source_ops()
    types = source_types()
    parameterized_attrs = source_parameterized_attrs()
    if data.startswith(MAGIC):
        return read_module(
            data,
            op_decls=operations,
            parameterized_attrs=parameterized_attrs,
            type_defs=types,
            verify=True,
        )

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filename}: Loom text is not valid UTF-8") from exc
    parser = Parser()
    parser.register_ops(operations)
    parser.register_types(types)
    parser.register_parameterized_attrs(parameterized_attrs)
    parser.register_encoding_families(ALL_ENCODING_FAMILIES)
    return parser.parse(source, filename=filename, verify=True)
