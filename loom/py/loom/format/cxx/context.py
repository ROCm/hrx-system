# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared analysis and emission state for readable Loom C++ projection."""

from __future__ import annotations

import collections
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from loom.dsl import Op
from loom.fields import compute_layout, resolve_fields
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.naming import clean_identifier
from loom.format.cxx.printer import CxxWriter
from loom.format.cxx.registry import source_ops
from loom.format.cxx.translation_unit import TargetInfo
from loom.format.cxx.types import CxxTypes
from loom.ir import (
    Block,
    BufferType,
    DialectType,
    Module,
    Operation,
    Predicate,
    PredicateListAttr,
    ScalarType,
    ShapedType,
    SignedEnumSetAttr,
    TypeKind,
)

_OPS: tuple[Op, ...] = source_ops()
_OP_BY_NAME = {operation.name: operation for operation in _OPS}


def walk_operations(block: Block) -> Iterable[Operation]:
    for operation in block.ops:
        yield operation
        for region in operation.regions:
            for child in region.blocks:
                yield from walk_operations(child)


def _walk_operations_with_blocks(
    block: Block,
) -> Iterable[tuple[Operation, Block]]:
    for operation in block.ops:
        yield operation, block
        for region in operation.regions:
            for child in region.blocks:
                yield from _walk_operations_with_blocks(child)


def _is_integer_scalar(value_type: Any) -> bool:
    return isinstance(value_type, ScalarType) and repr(value_type) in {
        "i8",
        "i16",
        "i32",
        "i64",
    }


def _is_integer_vector(value_type: Any) -> bool:
    return (
        isinstance(value_type, ShapedType)
        and value_type.type_kind == TypeKind.VECTOR
        and repr(value_type.element_type) in {"i8", "i16", "i32", "i64"}
    )


def _is_integer_value_type(value_type: Any) -> bool:
    return _is_integer_scalar(value_type) or _is_integer_vector(value_type)


def _memory_space(value: Any) -> str:
    if not isinstance(value, str):
        raise CxxRenderError(None, "memory space has no C++ enum spelling")
    suffix = f"{value}_" if value == "private" else value
    return f"loom::memory_space::{suffix}"


def _strip_outer_parentheses(expression: str) -> str:
    """Removes one redundant pair enclosing a complete C++ expression."""

    if len(expression) < 2 or expression[0] != "(" or expression[-1] != ")":
        return expression
    depth = 0
    for index, character in enumerate(expression):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0 and index != len(expression) - 1:
                return expression
    return expression[1:-1] if depth == 0 else expression


@dataclass(slots=True)
class BufferInfo:
    root: int
    name: str
    noalias: bool = False
    alignment: int | None = None
    mutable: bool = False
    is_parameter: bool = False
    element_types: set[str] = field(default_factory=set)


class CxxRenderContext:
    def __init__(self, module: Module):
        self.module = module
        self.fields = {
            name: compute_layout(operation) for name, operation in _OP_BY_NAME.items()
        }
        self.writer = CxxWriter()
        self.expressions: dict[int, str] = {}
        self.expression_types: dict[int, str] = {}
        self.use_counts: collections.Counter[int] = collections.Counter()
        self.definitions: dict[int, Operation] = {}
        self.definition_blocks: dict[int, Block] = {}
        self.use_blocks: dict[int, list[Block]] = collections.defaultdict(list)
        self.users: dict[int, list[Operation]] = collections.defaultdict(list)
        self.assumption_inputs: set[int] = set()
        for operation, block in _walk_operations_with_blocks(self.module.body):
            self.use_counts.update(operation.operands)
            for operand in operation.operands:
                self.use_blocks[operand].append(block)
                self.users[operand].append(operation)
            for result in operation.results:
                self.definitions[result] = operation
                self.definition_blocks[result] = block
            if operation.name in {"scalar.assume", "index.assume"}:
                self.assumption_inputs.update(operation.operands)
        self.operation_scopes, self.value_scopes = self._collect_value_scopes()
        self.scope_names: dict[int, set[str]] = collections.defaultdict(set)
        self.names = self._plan_names()
        self.temporary_names: set[str] = set()
        self.table_lookup_declarations: set[tuple[str, str, str, str]] = set()
        self.record_declarations: dict[str, tuple[tuple[str, str], ...]] = {}
        self.record_types: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
        self.uses_numeric_conversion = False
        self.signed_values = self._analyze_signed_values()
        self.types = CxxTypes(module, self.signed_values)
        self.unsigned_upper_bounds: dict[int, int | None] = {}
        self.assumption_inputs = {
            operand
            for operation in walk_operations(self.module.body)
            if operation.name in {"scalar.assume", "index.assume"}
            and not self._assumption_is_tautological(operation)
            for operand in operation.operands
        }
        self.buffer_roots: dict[int, int] = {}
        self.view_roots: dict[int, int] = {}
        self.buffer_info: dict[int, BufferInfo] = {}
        self.view_signed: set[int] = set()
        self.scoped_alignment_operations: set[int] = set()
        self.scoped_noalias_operations: set[int] = set()
        self.targets = self._collect_targets()
        self.value_origins = list(range(len(self.module.values)))
        self.rendered_assumptions: set[tuple[object, ...]] = set()
        self.assignment_targets: dict[int, str] = {}
        self.target_names: dict[str, str] = {}

    def _collect_value_scopes(self) -> tuple[dict[int, int], dict[int, int]]:
        """Maps top-level operations and their values to C++ local scopes."""
        operation_scopes: dict[int, int] = {}
        value_scopes: dict[int, int] = {}

        def assign(operation: Operation, scope: int) -> None:
            for value_id in (*operation.operands, *operation.results):
                value_scopes.setdefault(value_id, scope)
            for region in operation.regions:
                for block in region.blocks:
                    for argument_id in block.arg_ids:
                        value_scopes.setdefault(argument_id, scope)
                    for child in block.ops:
                        assign(child, scope)

        for scope, operation in enumerate(self.module.body.ops):
            operation_scopes[id(operation)] = scope
            assign(operation, scope)
        return operation_scopes, value_scopes

    def _plan_names(self) -> dict[int, str]:
        preferred: dict[int, str] = {}
        for value_id, value in enumerate(self.module.values):
            preferred[value_id] = clean_identifier(value.name, f"v{value_id}")

        def is_generated(value_id: int) -> bool:
            return re.fullmatch(r"v[0-9]+(?:_[0-9]+)?", preferred[value_id]) is not None

        # Casts from hardware IDs are implementation details introduced by the
        # importer. Give the C++-width value the source concept's name and let
        # the ID operation inline into it.
        for result, definition in self.definitions.items():
            if definition.name != "index.cast" or not is_generated(result):
                continue
            source_definition = self.definitions.get(definition.operands[0])
            if source_definition is None:
                continue
            category = {
                "kernel.workgroup.id": "workgroup",
                "kernel.workgroup.size": "workgroup_size",
                "kernel.workgroup.count": "workgroup_count",
                "kernel.workitem.id": "workitem",
                "kernel.subgroup.id": "subgroup",
                "kernel.subgroup.count": "subgroup_count",
                "kernel.subgroup.size": "subgroup_size",
                "kernel.subgroup.lane.id": "lane",
            }.get(source_definition.name)
            if category is None:
                continue
            dimension = source_definition.attributes.get("dimension")
            preferred[result] = (
                category if dimension in {None, "x"} else f"{category}_{dimension}"
            )

        # Assumptions refine an existing C++ binding instead of declaring a
        # second runtime value. Retain an authored name on the refined result
        # when the source topology operation carries an importer-generated
        # hardware name, and assign every assumption result its source name.
        assumption_sources: dict[int, int] = {}
        topology_operations = {
            "kernel.workgroup.id",
            "kernel.workgroup.size",
            "kernel.workgroup.count",
            "kernel.workitem.id",
        }
        for operation in walk_operations(self.module.body):
            if operation.name not in {"scalar.assume", "index.assume"}:
                continue
            resolved = self._resolved(operation)
            for source, result in zip(
                resolved.value_ids("values"),
                resolved.value_ids("results"),
                strict=True,
            ):
                assumption_sources[result] = source
                source_definition = self.definitions.get(source)
                if (
                    source_definition is not None
                    and source_definition.name in topology_operations
                    and not is_generated(result)
                ):
                    preferred[source] = preferred[result]

        # Views are source-level objects in C++. Name a generated view from its
        # sole load result when that result retained an authored name.
        for result, definition in self.definitions.items():
            if definition.name != "buffer.view" or not is_generated(result):
                continue
            users = self.users[result]
            if len(users) != 1:
                continue
            user = users[0]
            if user.name in {"view.load", "vector.load"} and len(user.results) == 1:
                loaded_name = preferred[user.results[0]]
                if not is_generated(user.results[0]):
                    preferred[result] = f"{loaded_name}_view"
            elif user.name in {"view.store", "vector.store"} and definition.operands:
                buffer_name = preferred[definition.operands[0]]
                if not is_generated(definition.operands[0]):
                    preferred[result] = f"{buffer_name}_view"

        for result, definition in self.definitions.items():
            if definition.name == "kernel.async.group" and definition.operands:
                source_name = preferred[definition.operands[0]]
                if not is_generated(definition.operands[0]):
                    preferred[result] = f"{source_name}_group"
            elif definition.name == "vector.from_elements" and is_generated(result):
                users = self.users[result]
                if users and all(
                    user.name == "vector.table.lookup"
                    and user.operands
                    and user.operands[0] == result
                    for user in users
                ):
                    preferred[result] = "lookup_table"

        # Back-propagate a view's semantic name through the transparent
        # assume/cast chain that carries its byte offset. This gives address
        # calculations names without interpreting their arithmetic.
        offset_labels: dict[int, list[str]] = collections.defaultdict(list)
        for result, definition in self.definitions.items():
            if definition.name != "buffer.view" or is_generated(result):
                continue
            resolved = self._resolved(definition)
            origin = resolved.value_id("byte_offset")
            while True:
                origin_definition = self.definitions.get(origin)
                if (
                    origin_definition is None
                    or origin_definition.name
                    not in {"index.cast", "scalar.assume", "index.assume"}
                    or not origin_definition.operands
                ):
                    break
                origin = origin_definition.operands[0]
            if (
                is_generated(origin)
                and self._constant_integer_from_definition(origin) is None
            ):
                offset_labels[origin].append(re.sub(r"_view$", "", preferred[result]))
        for value_id, labels in offset_labels.items():
            token_lists = [label.split("_") for label in labels]
            common_suffix: list[str] = []
            for tokens in zip(
                *(reversed(tokens) for tokens in token_lists), strict=False
            ):
                if len(set(tokens)) != 1:
                    break
                common_suffix.append(tokens[0])
            stem = "_".join(reversed(common_suffix)) or "view"
            preferred[value_id] = f"{stem}_byte_offset"

        names: dict[int, str] = {}
        scope_counts: dict[int, dict[str, int]] = collections.defaultdict(dict)
        for value_id in range(len(self.module.values)):
            assumption_source = assumption_sources.get(value_id)
            if assumption_source is not None:
                names[value_id] = names[assumption_source]
                continue
            base = preferred[value_id]
            scope = self.value_scopes.get(value_id, -1)
            counts = scope_counts[scope]
            ordinal = counts.get(base, 0)
            counts[base] = ordinal + 1
            name = base if ordinal == 0 else f"{base}_{ordinal}"
            names[value_id] = name
            self.scope_names[scope].add(name)
        return names

    def _constant_integer_from_definition(self, value_id: int) -> int | None:
        definition = self.definitions.get(value_id)
        if definition is None or definition.name not in {
            "scalar.constant",
            "index.constant",
        }:
            return None
        value = definition.attributes.get("value")
        return value if type(value) is int else None

    def _analyze_signed_values(self) -> set[int]:
        signed: set[int] = set()
        for operation in walk_operations(self.module.body):
            if operation.name == "scalar.constant":
                value = operation.attributes.get("value")
                if isinstance(value, int) and value < 0:
                    signed.update(operation.results)
            elif operation.name == "vector.from_elements":
                if any(operand in signed for operand in operation.operands):
                    signed.update(operation.results)
            elif operation.name in {
                "scalar.sitofp",
                "scalar.extsi",
                "scalar.divsi",
                "scalar.remsi",
                "scalar.minsi",
                "scalar.maxsi",
                "scalar.ashri",
                "vector.sitofp",
                "vector.extsi",
                "vector.divsi",
                "vector.remsi",
                "vector.minsi",
                "vector.maxsi",
                "vector.ashri",
            }:
                if operation.operands:
                    signed.add(operation.operands[0])
                if operation.name.endswith(
                    ("extsi", "divsi", "remsi", "minsi", "maxsi", "ashri")
                ):
                    signed.update(operation.results)
            elif operation.name in {"scalar.fptosi", "vector.fptosi"}:
                signed.update(operation.results)
            elif operation.name == "vector.table.lookup":
                if operation.operands and operation.operands[0] in signed:
                    signed.update(operation.results)

        # Signedness often becomes observable several signless operations after
        # a definition. Propagate only toward definitions, never across unsigned
        # semantic operations such as shrui/extui.
        passthrough = {
            "scalar.addi",
            "scalar.subi",
            "scalar.muli",
            "scalar.andi",
            "scalar.ori",
            "scalar.xori",
            "scalar.shli",
            "vector.addi",
            "vector.subi",
            "vector.muli",
            "vector.andi",
            "vector.ori",
            "vector.xori",
            "vector.shli",
            "vector.from_elements",
            "vector.extract",
            "vector.insert",
            "vector.splat",
        }
        changed = True
        operations = list(walk_operations(self.module.body))
        while changed:
            changed = False
            for operation in reversed(operations):
                if operation.name not in passthrough:
                    continue
                if any(result in signed for result in operation.results):
                    for operand in operation.operands:
                        if (
                            _is_integer_value_type(self.module.values[operand].type)
                            and operand not in signed
                        ):
                            signed.add(operand)
                            changed = True
                if operation.name in {"vector.from_elements", "vector.splat"} and any(
                    operand in signed for operand in operation.operands
                ):
                    for result in operation.results:
                        if result not in signed:
                            signed.add(result)
                            changed = True
        return signed

    def _collect_targets(self) -> dict[str, TargetInfo]:
        targets: dict[str, TargetInfo] = {}
        for operation in self.module.body.ops:
            if operation.name.endswith(".target"):
                symbol = operation.attributes.get("symbol")
                kind = operation.attributes.get("kind")
                if isinstance(symbol, str) and isinstance(kind, str):
                    if operation.name != "amdgpu.target":
                        targets[symbol] = TargetInfo(
                            operation, operation.name, kind, None, ()
                        )
                        continue
                    supported_attributes = {
                        "symbol",
                        "kind",
                        "subgroup_size",
                        "features",
                    }
                    unsupported = sorted(
                        set(operation.attributes) - supported_attributes
                    )
                    if unsupported:
                        raise CxxRenderError(
                            operation,
                            "target override attributes have no C++ projection: "
                            + ", ".join(unsupported),
                        )
                    subgroup_size = operation.attributes.get("subgroup_size")
                    if subgroup_size is not None and type(subgroup_size) is not int:
                        raise CxxRenderError(
                            operation, "subgroup_size is not an integer"
                        )
                    features: list[tuple[str, bool]] = []
                    feature_value = operation.attributes.get("features")
                    if feature_value is not None:
                        if not isinstance(feature_value, SignedEnumSetAttr):
                            raise CxxRenderError(
                                operation,
                                "AMDHSA features are not a signed enum set",
                            )
                        descriptor = next(
                            attribute
                            for attribute in _OP_BY_NAME[operation.name].attrs
                            if attribute.name == "features"
                        )
                        assert descriptor.enum_def is not None
                        names = {
                            case.value: case.keyword
                            for case in descriptor.enum_def.cases
                        }
                        features.extend(
                            (names[value], True)
                            for value in feature_value.positive_values
                        )
                        features.extend(
                            (names[value], False)
                            for value in feature_value.negative_values
                        )
                    targets[symbol] = TargetInfo(
                        operation,
                        operation.name,
                        kind,
                        subgroup_size,
                        tuple(features),
                    )
        return targets

    def _validate_projected_attributes(
        self, operation: Operation, projected: tuple[str, ...]
    ) -> None:
        unsupported = sorted(set(operation.attributes).difference(projected))
        if not unsupported:
            return
        if len(unsupported) == 1 and unsupported[0] in {
            "fastmath",
            "memory_flags",
            "overflow",
            "subnormal",
        }:
            raise CxxRenderError(
                operation,
                f"{unsupported[0]} flags have no exact C++ expression projection",
            )
        raise CxxRenderError(
            operation,
            "attributes have no exact C++ projection: " + ", ".join(unsupported),
        )

    def _resolved(self, operation: Operation):
        layout = self.fields.get(operation.name)
        if layout is None:
            raise CxxRenderError(operation, "operation has no registered descriptor")
        return resolve_fields(layout, operation, self.module)

    def _analyze_buffer_assumption(
        self, operation: Operation, is_parameter_contract: bool
    ) -> None:
        resolved = self._resolved(operation)
        if not is_parameter_contract:
            if operation.name == "buffer.assume.noalias":
                self.scoped_noalias_operations.add(id(operation))
            elif operation.name == "buffer.assume.alignment":
                self.scoped_alignment_operations.add(id(operation))
        if operation.name == "buffer.assume.memory_space":
            pairs = ((resolved.value_id("buffer"), resolved.value_id("result")),)
        else:
            pairs = zip(
                resolved.value_ids("buffers"),
                resolved.value_ids("results"),
                strict=True,
            )
        for source, result in pairs:
            root = self.buffer_roots.get(source)
            if root is None:
                continue
            self.buffer_roots[result] = root
            value_name = self.module.values[result].name
            if is_parameter_contract and value_name and not value_name.startswith("v_"):
                self.buffer_info[root].name = clean_identifier(
                    value_name, self.buffer_info[root].name
                )
            if operation.name == "buffer.assume.noalias" and is_parameter_contract:
                self.buffer_info[root].noalias = True
            elif operation.name == "buffer.assume.alignment" and is_parameter_contract:
                alignment = resolved.attr("minimum_alignment")
                if isinstance(alignment, int):
                    old = self.buffer_info[root].alignment or 1
                    self.buffer_info[root].alignment = max(old, alignment)

    def _analyze_buffers(self, kernel: Operation) -> None:
        fields = self._resolved(kernel)
        body = fields.region("body")
        if body is None or len(body.blocks) != 1:
            raise CxxRenderError(
                kernel, "kernel body must have one structured entry block"
            )
        entry = body.blocks[0]
        direct_operations = {id(operation) for operation in entry.ops}
        for argument in entry.arg_ids:
            if not isinstance(self.module.values[argument].type, BufferType):
                continue
            self.buffer_roots[argument] = argument
            self.buffer_info[argument] = BufferInfo(
                root=argument,
                name=self.names[argument],
                is_parameter=True,
            )

        for operation in walk_operations(entry):
            if operation.name in {
                "buffer.assume.alignment",
                "buffer.assume.memory_space",
                "buffer.assume.noalias",
            }:
                self._analyze_buffer_assumption(
                    operation,
                    operation.name != "buffer.assume.memory_space"
                    and id(operation) in direct_operations,
                )
                continue
            if operation.name == "buffer.alloca":
                for result in operation.results:
                    self.buffer_roots[result] = result
                    self.buffer_info[result] = BufferInfo(
                        root=result,
                        name=self.names[result],
                        mutable=True,
                    )
                continue
            if operation.name == "buffer.view":
                resolved = self._resolved(operation)
                source = resolved.value_id("buffer")
                result = resolved.value_id("result")
                root = self.buffer_roots.get(source)
                if root is not None:
                    self.view_roots[result] = root
                    result_type = self.module.values[result].type
                    if isinstance(result_type, ShapedType):
                        self.buffer_info[root].element_types.add(
                            repr(result_type.element_type)
                        )
                continue
            if operation.name == "view.subview":
                resolved = self._resolved(operation)
                root = self.view_roots.get(resolved.value_id("source"))
                if root is not None:
                    self.view_roots[resolved.value_id("result")] = root
                continue
            if operation.name in {"view.store", "vector.store"}:
                resolved = self._resolved(operation)
                root = self.view_roots.get(resolved.value_id("view"))
                if root is not None:
                    self.buffer_info[root].mutable = True
                continue
            if operation.name in {
                "view.atomic.store",
                "view.atomic.rmw",
                "view.atomic.reduce",
                "view.atomic.cmpxchg",
            }:
                resolved = self._resolved(operation)
                root = self.view_roots.get(resolved.value_id("view"))
                if root is not None:
                    self.buffer_info[root].mutable = True
                continue
            if operation.name == "kernel.async.gather":
                resolved = self._resolved(operation)
                root = self.view_roots.get(resolved.value_id("dest"))
                if root is not None:
                    self.buffer_info[root].mutable = True

        # Carry integer interpretation from loads/stores to the view element.
        changed = True
        while changed:
            changed = False
            for operation in walk_operations(entry):
                if operation.name == "view.subview":
                    source, result = operation.operands[0], operation.results[0]
                    if result in self.view_signed and source not in self.view_signed:
                        self.view_signed.add(source)
                        changed = True
                    if source in self.view_signed and result not in self.view_signed:
                        self.view_signed.add(result)
                        changed = True
                elif operation.name in {"view.load", "vector.load"}:
                    resolved = self._resolved(operation)
                    view_id = resolved.value_id("view")
                    result_id = operation.results[0]
                    if (
                        result_id in self.signed_values
                        and view_id not in self.view_signed
                    ):
                        self.view_signed.add(view_id)
                        changed = True
                elif operation.name in {"view.store", "vector.store"}:
                    resolved = self._resolved(operation)
                    value_field = "value"
                    view_id = resolved.value_id("view")
                    value_id = resolved.value_id(value_field)
                    if (
                        value_id in self.signed_values
                        and view_id not in self.view_signed
                    ):
                        self.view_signed.add(view_id)
                        changed = True
                elif operation.name.startswith("view.atomic."):
                    resolved = self._resolved(operation)
                    view_id = resolved.value_id("view")
                    value_fields = {
                        "view.atomic.load": ("result",),
                        "view.atomic.store": ("value",),
                        "view.atomic.rmw": ("value", "result"),
                        "view.atomic.reduce": ("value",),
                        "view.atomic.cmpxchg": (
                            "expected",
                            "replacement",
                            "old",
                        ),
                    }[operation.name]
                    signed = any(
                        value_id in self.signed_values
                        for field in value_fields
                        for value_id in resolved.value_ids(field)
                    )
                    signed = signed or (
                        operation.name in {"view.atomic.rmw", "view.atomic.reduce"}
                        and resolved.attr("kind") in {"minsi", "maxsi"}
                    )
                    if signed and view_id not in self.view_signed:
                        self.view_signed.add(view_id)
                        changed = True

    def cpp_type(self, value_id: int, *, signed: bool | None = None) -> str:
        return self.types.value(value_id, signed=signed)

    def _literal(self, value: Any, result_id: int) -> str:
        value_type = self.module.values[result_id].type
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            suffix = "f" if repr(value_type) == "f32" else ""
            text = repr(value)
            if "." not in text and "e" not in text.lower():
                text += ".0"
            return text + suffix
        if isinstance(value, int):
            if isinstance(value_type, ScalarType):
                name = repr(value_type)
                if name == "index":
                    return f"loom::type::index{{{value}}}"
                if name == "offset":
                    return f"loom::type::offset{{{value}}}"
                if name == "i64" and value >= 0:
                    return f"{value}ull"
                if (
                    name in {"i8", "i16", "i32"}
                    and value >= 0
                    and result_id not in self.signed_values
                ):
                    return f"{value}u"
            return str(value)
        raise CxxRenderError(None, f"constant {value!r} has no C++ literal")

    def _is_unsigned_scalar(self, value_id: int) -> bool:
        value_type = self.module.values[value_id].type
        if isinstance(value_type, ScalarType):
            name = repr(value_type)
            if name == "index":
                return False
            if name == "offset":
                return True
        return (
            isinstance(value_type, ScalarType)
            and repr(value_type) in {"i1", "i8", "i16", "i32", "i64"}
            and value_id not in self.signed_values
        )

    def _unsigned_type_maximum(self, value_id: int) -> int | None:
        if not self._is_unsigned_scalar(value_id):
            return None
        width = {
            "i1": 1,
            "i8": 8,
            "i16": 16,
            "i32": 32,
            "i64": 64,
            "index": 64,
            "offset": 64,
        }[repr(self.module.values[value_id].type)]
        return (1 << width) - 1

    def _constant_integer(self, value_id: int) -> int | None:
        definition = self.definitions.get(value_id)
        if definition is None:
            return None
        if definition.name in {"scalar.constant", "index.constant"}:
            value = definition.attributes.get("value")
            return value if type(value) is int else None
        if not definition.operands:
            return None
        if definition.name in {"scalar.assume", "index.assume"}:
            return self._constant_integer(definition.operands[0])
        if definition.name in {"scalar.extui", "scalar.extsi"}:
            source = definition.operands[0]
            constant = self._constant_integer(source)
            source_type = repr(self.module.values[source].type)
            width = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}.get(source_type)
            if constant is None or width is None:
                return None
            bits = constant & ((1 << width) - 1)
            if definition.name == "scalar.extui" or width == 1:
                return bits
            sign_bit = 1 << (width - 1)
            return bits - (1 << width) if bits & sign_bit else bits
        if definition.name == "index.cast":
            source = definition.operands[0]
            constant = self._constant_integer(source)
            if constant is None:
                return None
            source_type = repr(self.module.values[source].type)
            result_type = repr(self.module.values[value_id].type)
            if result_type not in {"index", "offset"}:
                return None
            width = {
                "i1": 1,
                "i8": 8,
                "i16": 16,
                "i32": 32,
                "i64": 64,
            }.get(source_type)
            if width is not None:
                bits = constant & ((1 << width) - 1)
                if result_type == "offset":
                    return bits if bits <= (1 << 63) - 1 else None
                if width == 1:
                    return bits
                sign_bit = 1 << (width - 1)
                return bits - (1 << width) if bits & sign_bit else bits
            if source_type == result_type:
                return constant
            if source_type == "index" and result_type == "offset":
                return constant if constant >= 0 else None
            if source_type == "offset" and result_type == "index":
                return constant if constant <= (1 << 63) - 1 else None
        return None

    def _known_unsigned_upper_bound(
        self, value_id: int, active: set[int] | None = None
    ) -> int | None:
        """Returns a cheap, conservative bound used only to omit tautologies."""

        if value_id in self.unsigned_upper_bounds:
            return self.unsigned_upper_bounds[value_id]
        type_maximum = self._unsigned_type_maximum(value_id)
        if type_maximum is None:
            self.unsigned_upper_bounds[value_id] = None
            return None
        if active is None:
            active = set()
        if value_id in active:
            return type_maximum
        active.add(value_id)

        upper = type_maximum
        definition = self.definitions.get(value_id)
        if definition is not None:
            operands = definition.operands
            if definition.name in {"scalar.constant", "index.constant"}:
                constant = definition.attributes.get("value")
                if type(constant) is int and constant >= 0:
                    upper = min(type_maximum, constant)
            elif (
                definition.name
                in {
                    "scalar.extui",
                    "scalar.trunci",
                    "index.cast",
                    "scalar.assume",
                    "index.assume",
                }
                and operands
            ):
                source_upper = self._known_unsigned_upper_bound(operands[0], active)
                if source_upper is not None:
                    upper = min(type_maximum, source_upper)
            elif (
                definition.name in {"scalar.addi", "scalar.muli"} and len(operands) == 2
            ):
                lhs = self._known_unsigned_upper_bound(operands[0], active)
                rhs = self._known_unsigned_upper_bound(operands[1], active)
                if lhs is not None and rhs is not None:
                    combined = (
                        lhs + rhs if definition.name == "scalar.addi" else lhs * rhs
                    )
                    upper = min(type_maximum, combined)
            elif definition.name == "scalar.shli" and len(operands) == 2:
                lhs = self._known_unsigned_upper_bound(operands[0], active)
                shift = self._constant_integer(operands[1])
                if lhs is not None and shift is not None and shift >= 0:
                    upper = min(type_maximum, lhs << shift)
            elif definition.name in {"scalar.divui", "scalar.shrui"} and operands:
                lhs = self._known_unsigned_upper_bound(operands[0], active)
                if lhs is not None:
                    upper = min(type_maximum, lhs)
            elif definition.name == "scalar.andi" and len(operands) == 2:
                lhs = self._known_unsigned_upper_bound(operands[0], active)
                rhs = self._known_unsigned_upper_bound(operands[1], active)
                known = [bound for bound in (lhs, rhs) if bound is not None]
                if known:
                    upper = min(type_maximum, *known)
            elif definition.name == "scalar.cmpi":
                upper = 1

        active.remove(value_id)
        self.unsigned_upper_bounds[value_id] = upper
        return upper

    def _predicate_constant(self, value: int, reference_id: int | None) -> str:
        if reference_id is not None:
            reference_type = self.module.values[reference_id].type
            if isinstance(reference_type, ScalarType):
                name = repr(reference_type)
                if name in {"index", "offset"}:
                    return str(value)
        if (
            value < 0
            or reference_id is None
            or not self._is_unsigned_scalar(reference_id)
        ):
            return str(value)
        type_name = repr(self.module.values[reference_id].type)
        return f"{value}{'ull' if type_name in {'i64', 'index', 'offset'} else 'u'}"

    def _predicate_is_tautological(self, predicate: Predicate) -> bool:
        if predicate.kind != "range" or len(predicate.args) != 3:
            return False
        subject, lower, upper = predicate.args
        if (
            subject.tag != "value"
            or lower.tag != "const"
            or lower.value != 0
            or upper.tag != "const"
            or upper.value < (1 << 63) - 1
            or not self._is_unsigned_scalar(subject.value)
        ):
            return False
        known_upper = self._known_unsigned_upper_bound(subject.value)
        return known_upper is not None and known_upper <= upper.value

    def _assumption_is_tautological(self, operation: Operation) -> bool:
        predicates = self._resolved(operation).attr("predicates")
        return (
            isinstance(predicates, PredicateListAttr)
            and bool(predicates)
            and all(
                self._predicate_is_tautological(predicate) for predicate in predicates
            )
        )

    def expression(self, value_id: int) -> str:
        expression = self.expressions.get(value_id)
        if expression is None:
            raise CxxRenderError(
                None, f"value %{self.module.values[value_id].name} is not in scope"
            )
        return expression

    def expression_type(self, value_id: int) -> str:
        return self.expression_types.get(value_id, self.cpp_type(value_id))

    def _declare(
        self,
        result_id: int,
        expression: str,
        *,
        declaration_type: str | None = None,
        expression_type: str | None = None,
        inlineable: bool = False,
        statement_attribute: str | None = None,
    ) -> None:
        name = self.names[result_id]
        result_type = self.module.values[result_id].type
        if declaration_type is not None:
            type_name = declaration_type
        elif (
            isinstance(result_type, ShapedType)
            and result_type.type_kind == TypeKind.VIEW
        ) or isinstance(result_type, (BufferType, DialectType)):
            type_name = "auto"
        else:
            type_name = self.cpp_type(result_id)
        self.expression_types[result_id] = (
            expression_type if expression_type is not None else type_name
        )
        if (
            inlineable
            and statement_attribute is None
            and result_id not in self.assignment_targets
            and self.use_counts[result_id] == 1
            and result_id not in self.assumption_inputs
            and (
                re.fullmatch(r"v[0-9]+(?:_[0-9]+)?", name)
                or (
                    self.definitions[result_id].name
                    in {
                        "kernel.workgroup.id",
                        "kernel.workgroup.size",
                        "kernel.workgroup.count",
                        "kernel.workitem.id",
                        "kernel.subgroup.id",
                        "kernel.subgroup.count",
                        "kernel.subgroup.size",
                        "kernel.subgroup.lane.id",
                    }
                    and self.users[result_id][0].name == "index.cast"
                )
            )
            and len(self.use_blocks[result_id]) == 1
            and self.use_blocks[result_id][0] is self.definition_blocks[result_id]
        ):
            self.expressions[result_id] = expression
            return
        statement_expression = _strip_outer_parentheses(expression)
        prefix = f"[[{statement_attribute}]] " if statement_attribute else ""
        assignment_target = self.assignment_targets.get(result_id)
        if assignment_target is not None:
            self.writer.line(f"{prefix}{assignment_target} = {statement_expression};")
            self.expressions[result_id] = assignment_target
        else:
            self.writer.line(f"{prefix}{type_name} {name} = {statement_expression};")
            self.expressions[result_id] = name

    def _fastmath_contract(self, operation: Operation) -> str | None:
        value = operation.attributes.get("fastmath")
        if value is None:
            return None
        if not isinstance(value, str):
            raise CxxRenderError(
                operation, "fastmath flags do not have a symbolic spelling"
            )
        descriptor = next(
            attribute
            for attribute in _OP_BY_NAME[operation.name].attrs
            if attribute.name == "fastmath"
        )
        assert descriptor.enum_def is not None
        supported = {case.keyword for case in descriptor.enum_def.cases}
        flags = value.split("|")
        unknown = [flag for flag in flags if flag not in supported]
        if unknown:
            raise CxxRenderError(
                operation, f"fastmath flag {unknown[0]!r} has no C++ spelling"
            )
        if flags == ["fast"]:
            return "loom::fastmath"
        return "loom::fastmath(" + ", ".join(f'"{flag}"' for flag in flags) + ")"

    def _cast(self, result_id: int, input_expression: str) -> str:
        result_type = self.module.values[result_id].type
        if isinstance(result_type, ScalarType):
            domain = repr(result_type)
            if domain in {"index", "offset"}:
                return f"loom::{domain}::cast({input_expression})"
        return f"static_cast<{self.cpp_type(result_id)}>({input_expression})"
