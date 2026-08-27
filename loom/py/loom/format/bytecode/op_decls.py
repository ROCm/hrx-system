# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Operation declaration helpers for Python bytecode support."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from loom.dsl import FuncLikeInterface


def default_op_decls() -> tuple[Any, ...]:
    """Return the built-in generated operation declarations."""
    from loom.dialect.buffer import ALL_BUFFER_OPS
    from loom.dialect.cfg import ALL_CFG_OPS
    from loom.dialect.command import ALL_COMMAND_OPS
    from loom.dialect.encoding import ALL_ENCODING_OPS
    from loom.dialect.func import ALL_FUNC_OPS
    from loom.dialect.globals import ALL_GLOBAL_OPS
    from loom.dialect.index import ALL_INDEX_OPS
    from loom.dialect.kernel import ALL_KERNEL_OPS
    from loom.dialect.llvmir import ALL_LLVMIR_OPS
    from loom.dialect.low import ALL_LOW_OPS
    from loom.dialect.pass_ import ALL_PASS_OPS
    from loom.dialect.pool import ALL_POOL_OPS
    from loom.dialect.scalar import ALL_SCALAR_OPS
    from loom.dialect.scf import ALL_SCF_OPS
    from loom.dialect.target import ALL_TARGET_OPS
    from loom.dialect.template import ALL_TEMPLATE_OPS
    from loom.dialect.test import ALL_TEST_OPS
    from loom.dialect.vector import ALL_VECTOR_OPS
    from loom.dialect.view import ALL_VIEW_OPS

    return (
        *ALL_TEST_OPS,
        *ALL_SCALAR_OPS,
        *ALL_FUNC_OPS,
        *ALL_TEMPLATE_OPS,
        *ALL_ENCODING_OPS,
        *ALL_POOL_OPS,
        *ALL_GLOBAL_OPS,
        *ALL_SCF_OPS,
        *ALL_CFG_OPS,
        *ALL_COMMAND_OPS,
        *ALL_BUFFER_OPS,
        *ALL_VIEW_OPS,
        *ALL_VECTOR_OPS,
        *ALL_INDEX_OPS,
        *ALL_KERNEL_OPS,
        *ALL_LLVMIR_OPS,
        *ALL_TARGET_OPS,
        *ALL_LOW_OPS,
        *ALL_PASS_OPS,
    )


def default_parameterized_attr_defs() -> tuple[Any, ...]:
    """Return built-in families not reachable from one exact op field."""
    from loom.dialect.target import ALL_TARGET_PARAMETERIZED_ATTRS

    return (*ALL_TARGET_PARAMETERIZED_ATTRS,)


def build_op_decl_map(op_decls: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return op declarations keyed by dotted operation name."""
    result: dict[str, Any] = {}
    for op_decl in default_op_decls() if op_decls is None else op_decls:
        result[op_decl.name] = op_decl
    return result


def build_parameterized_attr_def_map(
    op_decls_by_name: Mapping[str, Any],
    parameterized_attrs: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return reachable parameterized attribute declarations by stable name."""
    pending = list(
        default_parameterized_attr_defs()
        if parameterized_attrs is None
        else parameterized_attrs
    )
    for op_decl in op_decls_by_name.values():
        for attr_def in getattr(op_decl, "attrs", ()):
            definition = getattr(attr_def, "parameterized_attr", None)
            if definition is not None:
                pending.append(definition)

    result: dict[str, Any] = {}
    while pending:
        definition = pending.pop()
        existing = result.get(definition.name)
        if existing is not None:
            if existing != definition:
                raise ValueError(
                    "parameterized attribute family "
                    f"{definition.name!r} has conflicting declarations"
                )
            continue
        result[definition.name] = definition
        for parameter in definition.parameters:
            nested_definition = getattr(parameter, "parameterized_attr", None)
            if nested_definition is not None:
                pending.append(nested_definition)
    return result


def build_parameterized_type_def_map(
    type_defs: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return descriptor-backed type declarations by stable family name."""
    if type_defs is None:
        from loom.builders import default_types

        type_defs = default_types()
    result: dict[str, Any] = {}
    for type_def in type_defs:
        if not getattr(type_def, "uses_attribute_parameters", False):
            continue
        existing = result.get(type_def.name)
        if existing is not None and existing != type_def:
            raise ValueError(
                f"parameterized type family {type_def.name!r} has "
                "conflicting declarations"
            )
        result[type_def.name] = type_def
    return result


def symbol_def_for_op(op_decls_by_name: Mapping[str, Any], op_name: str) -> Any:
    """Return the generated symbol definition descriptor for ``op_name``."""
    op_decl = op_decls_by_name.get(op_name)
    symbol_def = getattr(op_decl, "symbol_def", None)
    if symbol_def is None:
        raise ValueError(
            f"symbol defining op {op_name!r} has no registered generated symbol_def"
        )
    return symbol_def


def func_like_interface_for_op(
    op_decls_by_name: Mapping[str, Any], op_name: str
) -> Any | None:
    """Return the generated FuncLike interface descriptor for ``op_name``."""
    op_decl = op_decls_by_name.get(op_name)
    if op_decl is None:
        return None
    for interface in getattr(op_decl, "interfaces", ()):
        if isinstance(interface, FuncLikeInterface):
            return interface
    return None


def attr_def_for_op(
    op_decls_by_name: Mapping[str, Any], op_name: str, attr_name: str
) -> Any | None:
    """Return the generated attribute descriptor for ``op_name.attr_name``."""
    op_decl = op_decls_by_name.get(op_name)
    if op_decl is None:
        return None
    for attr_def in getattr(op_decl, "attrs", ()):
        if attr_def.name == attr_name:
            return attr_def
    return None
