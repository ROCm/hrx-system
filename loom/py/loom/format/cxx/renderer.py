# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Readable Loom High IR projection into the Loom C++ authoring surface."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from loom.format.cxx.context import (
    _memory_space,
    _strip_outer_parentheses,
    walk_operations,
)
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.kernel_configurations import CxxKernelConfigurationRenderer
from loom.format.cxx.naming import clean_identifier, loop_carrier_stem
from loom.format.cxx.translation_unit import render_preamble
from loom.format.cxx.types import KERNEL_COORDINATE_CPP_TYPE
from loom.ir import (
    Block,
    BufferType,
    DynamicDim,
    DynamicEncoding,
    Module,
    Operation,
    PredicateListAttr,
    ScalarType,
    ShapedType,
    StaticDim,
    TypeKind,
    Utf8Attr,
)


def _buffer_payload_literal(contents: bytes | Utf8Attr) -> str:
    """Quotes exact payload code units without greedy hexadecimal escapes."""
    text = (
        contents.value if isinstance(contents, Utf8Attr) else contents.decode("latin1")
    )
    prefix = "u8" if isinstance(contents, Utf8Attr) else ""
    escaped = []
    for character in text:
        if character in {'"', "\\"}:
            escaped.append("\\" + character)
        elif character in {"\n", "\r", "\t"}:
            escaped.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[character])
        elif (
            ord(character) < 32
            or ord(character) == 127
            or (not prefix and ord(character) >= 128)
        ):
            escaped.append(f"\\{ord(character):03o}")
        else:
            escaped.append(character)
    return prefix + '"' + "".join(escaped) + '"'


class CxxRenderer(CxxKernelConfigurationRenderer):
    def _view_shape_arguments(
        self, operation: Operation, view_type: ShapedType
    ) -> tuple[str, str]:
        """Projects a view shape into C++ template and runtime arguments."""

        extents: list[str] = []
        dimensions: list[str] = []
        for dimension in view_type.dims:
            if isinstance(dimension, StaticDim):
                extents.append(str(dimension.size))
                continue
            if not isinstance(dimension, DynamicDim) or dimension.value_id is None:
                raise CxxRenderError(
                    operation, "unbound dynamic view dimension has no C++ projection"
                )
            extents.append("loom::type::dynamic")
            dimensions.append(self._address_operand(dimension.value_id))
        runtime_dimensions = (
            "{}"
            if not dimensions
            else f"loom::type::indices{{{', '.join(dimensions)}}}"
        )
        return ", ".join(extents), runtime_dimensions

    def _compatible_cxx_carrier(self, lhs: int, rhs: int) -> bool:
        """Returns whether one mutable C++ carrier preserves both IR types."""

        lhs_type = self.module.values[lhs].type
        rhs_type = self.module.values[rhs].type
        if repr(lhs_type) == repr(rhs_type):
            return True
        if not (
            isinstance(lhs_type, ShapedType)
            and lhs_type.type_kind == TypeKind.VIEW
            and isinstance(rhs_type, ShapedType)
            and rhs_type.type_kind == TypeKind.VIEW
        ):
            return False

        def dimension_key(dimension: StaticDim | DynamicDim) -> tuple[str, int]:
            if isinstance(dimension, StaticDim):
                return ("static", dimension.size)
            return ("dynamic", 0)

        def encoding_key(encoding: object) -> tuple[str, str]:
            if isinstance(encoding, DynamicEncoding):
                return (
                    "dynamic",
                    repr(self.module.values[encoding.value_id].type),
                )
            return ("value", repr(encoding))

        return (
            lhs_type.element_type == rhs_type.element_type
            and tuple(dimension_key(dimension) for dimension in lhs_type.dims)
            == tuple(dimension_key(dimension) for dimension in rhs_type.dims)
            and encoding_key(lhs_type.encoding) == encoding_key(rhs_type.encoding)
            and lhs_type.alignment == rhs_type.alignment
        )

    def _buffer_pointer_expression(
        self, buffer_id: int, byte_offset_id: int, element_type: str, is_const: bool
    ) -> str:
        base = self.expression(buffer_id)
        offset = self.expression(byte_offset_id)
        zero_offset = self._constant_integer(byte_offset_id) == 0
        root = self.buffer_roots.get(buffer_id)
        if (
            root is not None
            and self.buffer_info[root].is_parameter
            and zero_offset
            and self._parameter_element(root) == element_type
        ):
            return base
        byte_type = "const unsigned char*" if is_const else "unsigned char*"
        target_type = f"const {element_type}*" if is_const else f"{element_type}*"
        if zero_offset and self.expression_type(buffer_id) == target_type:
            return base
        byte_base = f"reinterpret_cast<{byte_type}>({base})"
        if self.expression_type(byte_offset_id) == "loom::type::offset":
            offset = f"static_cast<unsigned long long>({offset})"
        address = byte_base if zero_offset else f"{byte_base} + {offset}"
        if byte_type == target_type:
            return address
        return f"reinterpret_cast<{target_type}>({address})"

    def _atomic_access_arguments(self, operation: Operation) -> str:
        resolved = self._resolved(operation)
        indices = self._mixed_indices(
            resolved, dynamic_field="indices", static_field="static_indices"
        )
        arguments = [self.expression(resolved.value_id("view")), *indices]
        return ", ".join(arguments)

    def _emit_atomic_result(self, result: int, expression: str) -> None:
        if self.use_counts[result] == 0 and result not in self.assignment_targets:
            self.writer.line(f"(void){expression};")
            return
        self._declare(result, expression)

    def _render_atomic(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if not name.startswith("view.atomic."):
            return None
        resolved = self._resolved(operation)
        access = self._atomic_access_arguments(operation)
        scope = resolved.attr("scope")
        if name == "view.atomic.cmpxchg":
            success = resolved.attr("success_ordering")
            failure = resolved.attr("failure_ordering")
            expression = (
                "loom::view::atomic::cmpxchg<"
                f"loom::atomic::ordering::{success}, "
                f"loom::atomic::ordering::{failure}, "
                f"loom::atomic::scope::{scope}>("
                f"{self.expression(resolved.value_id('expected'))}, "
                f"{self.expression(resolved.value_id('replacement'))}, "
                f"{access})"
            )
            self._emit_atomic_result(resolved.value_id("old"), expression)
            return (
                "success_ordering",
                "failure_ordering",
                "scope",
                "static_indices",
            )

        ordering = resolved.attr("ordering")
        if name == "view.atomic.load":
            expression = (
                "loom::view::atomic::load<"
                f"loom::atomic::ordering::{ordering}, "
                f"loom::atomic::scope::{scope}>({access})"
            )
            self._emit_atomic_result(resolved.value_id("result"), expression)
            return ("ordering", "scope", "static_indices")
        if name == "view.atomic.store":
            self.writer.line(
                "loom::view::atomic::store<"
                f"loom::atomic::ordering::{ordering}, "
                f"loom::atomic::scope::{scope}>("
                f"{self.expression(resolved.value_id('value'))}, {access});"
            )
            return ("ordering", "scope", "static_indices")

        kind = resolved.attr("kind")
        member = "rmw" if name == "view.atomic.rmw" else "reduce"
        expression = (
            f"loom::view::atomic::{member}<loom::atomic::kind::{kind}, "
            f"loom::atomic::ordering::{ordering}, "
            f"loom::atomic::scope::{scope}>("
            f"{self.expression(resolved.value_id('value'))}, {access})"
        )
        if name == "view.atomic.rmw":
            self._emit_atomic_result(resolved.value_id("result"), expression)
        else:
            self.writer.line(f"{expression};")
        return ("kind", "ordering", "scope", "static_indices")

    def _render_memory(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        projected = self._render_atomic(operation)
        if projected is not None:
            return projected
        if name == "buffer.fence":
            resolved = self._resolved(operation)
            self.writer.line(
                "loom::buffer::fence<"
                f"loom::atomic::ordering::{resolved.attr('ordering')}, "
                f"loom::atomic::scope::{resolved.attr('scope')}>();"
            )
            return ("ordering", "scope")
        if name == "encoding.layout.dense":
            # Loom's dense layout value is rankless. Its C++ carrier acquires
            # rank from each shaped use, so buffer.view projects it in context.
            return ()
        if name == "encoding.layout.strided":
            resolved = self._resolved(operation)
            strides = ", ".join(
                self._mixed_indices(
                    resolved,
                    dynamic_field="strides",
                    static_field="static_strides",
                )
            )
            self._declare(
                resolved.value_id("result"),
                f"loom::encoding::layout::strided({strides})",
                declaration_type="auto",
            )
            return ("static_strides",)
        if name in {"buffer.constant", "buffer.alloca.init"}:
            resolved = self._resolved(operation)
            immutable = name == "buffer.constant"
            if not immutable and resolved.value_ids("pool"):
                raise CxxRenderError(
                    operation, "explicit allocation pools have no C++ projection"
                )
            if immutable:
                alignment = resolved.attr("alignment") or 1
                callee = f"loom::buffer::constant<{alignment}>"
                attributes = ("contents", "alignment")
            else:
                alignment = resolved.attr("base_alignment")
                space = _memory_space(resolved.attr("memory_space"))
                callee = f"loom::buffer::alloca_init<{space}, {alignment}>"
                attributes = ("contents", "memory_space", "base_alignment")
            result = resolved.value_id("result")
            literal = _buffer_payload_literal(resolved.attr("contents"))
            self._declare(result, f"{callee}({literal})", declaration_type="auto*")
            self.expression_types[result] = (
                "const unsigned char*" if immutable else "unsigned char*"
            )
            return attributes
        if name == "buffer.alloca":
            resolved = self._resolved(operation)
            if resolved.value_ids("pool"):
                raise CxxRenderError(
                    operation, "explicit allocation pools have no C++ projection"
                )
            space = resolved.attr("memory_space")
            alignment = resolved.attr("base_alignment")
            length_id = resolved.value_id("byte_length")
            length = self.expression(length_id)
            length_definition = self.definitions.get(length_id)
            allocates_bytes = True
            if (
                length_definition is not None
                and length_definition.name == "index.scale"
                and self._constant_integer(length_definition.operands[1]) == 1
            ):
                count = length_definition.operands[0]
                length = self.expression(count)
                allocates_bytes = False
            result = resolved.value_id("result")
            template = f"{_memory_space(space)}, {alignment}"
            if not allocates_bytes:
                template = f"unsigned char, {template}"
            expression = f"loom::buffer::alloca<{template}>({length})"
            self._declare(
                result,
                expression,
                declaration_type="auto*",
            )
            self.expression_types[result] = "unsigned char*"
            return ("memory_space", "base_alignment")
        if name == "buffer.view":
            resolved = self._resolved(operation)
            buffer_id = resolved.value_id("buffer")
            offset_id = resolved.value_id("byte_offset")
            result = resolved.value_id("result")
            result_type = self.module.values[result].type
            if (
                not isinstance(result_type, ShapedType)
                or result_type.type_kind != TypeKind.VIEW
            ):
                raise CxxRenderError(
                    operation, "buffer.view result is not a shaped view"
                )
            extents, dimensions = self._view_shape_arguments(operation, result_type)
            if isinstance(result_type.encoding, DynamicEncoding):
                encoding = self.dynamic_view_encoding_expression(operation, result_type)
            elif result_type.encoding is None:
                encoding = None
            else:
                raise CxxRenderError(
                    operation,
                    "static view encoding has no explicit C++ value projection",
                )
            if result_type.alignment is not None:
                raise CxxRenderError(
                    operation,
                    "view access alignment has no C++ type projection",
                )
            root = self.buffer_roots.get(buffer_id)
            is_const = root is not None and not self.buffer_info[root].mutable
            element = self.types.scalar(
                result_type.element_type, result in self.view_signed
            )
            source_type = self.expression_type(buffer_id)
            arguments: list[str]
            if source_type.startswith("loom::type::buffer<"):
                target_type = f"loom::type::buffer<{element}>"
                source = self.expression(buffer_id)
                if source_type != target_type:
                    source = f"loom::buffer::cast<{element}>({source})"
                arguments = [source]
                if self._constant_integer(offset_id) != 0:
                    arguments.append(self._address_operand(offset_id))
            else:
                arguments = [
                    self._buffer_pointer_expression(
                        buffer_id, offset_id, element, is_const
                    )
                ]
            arguments.append(dimensions)
            if encoding is not None:
                arguments.append(encoding)
            expression = f"loom::buffer::view<{extents}>(" + ", ".join(arguments) + ")"
            self._declare(result, expression)
            return ()
        if name == "view.subview":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            result_type = self.module.values[result].type
            if (
                not isinstance(result_type, ShapedType)
                or result_type.type_kind != TypeKind.VIEW
            ):
                raise CxxRenderError(
                    operation, "view.subview result is not a shaped view"
                )
            extents, dimensions = self._view_shape_arguments(operation, result_type)
            origin = self._mixed_index_record(
                resolved,
                dynamic_field="offsets",
                static_field="static_offsets",
            )
            source = self.expression(resolved.value_id("source"))
            expression = (
                f"loom::view::subview<{extents}>({source}, {origin}, {dimensions})"
            )
            self._declare(result, expression)
            return ("static_offsets",)
        if name == "view.bitcast":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            source = self.expression(resolved.value_id("source"))
            self._declare(
                result,
                f"loom::view::bitcast<{self.cpp_type(result)}>({source})",
                inlineable=True,
            )
            return ()
        if name == "view.load":
            resolved = self._resolved(operation)
            view_id = resolved.value_id("view")
            indices = ", ".join(
                self._mixed_indices(
                    resolved,
                    dynamic_field="indices",
                    static_field="static_indices",
                )
            )
            suffix = f", {indices}" if indices else ""
            self._declare(
                operation.results[0],
                f"loom::view::load({self.expression(view_id)}{suffix})",
            )
            return ("static_indices",)
        if name == "view.store":
            resolved = self._resolved(operation)
            indices = ", ".join(
                self._mixed_indices(
                    resolved,
                    dynamic_field="indices",
                    static_field="static_indices",
                )
            )
            suffix = f", {indices}" if indices else ""
            value = self.expression(resolved.value_id("value"))
            view = self.expression(resolved.value_id("view"))
            self.writer.line(f"loom::view::store({value}, {view}{suffix});")
            return ("static_indices",)
        if name == "vector.load":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            origin = self._mixed_index_record(
                resolved,
                dynamic_field="indices",
                static_field="static_indices",
            )
            view = self.expression(resolved.value_id("view"))
            self._declare(
                result,
                f"loom::vector::load<{self.cpp_type(result)}>({view}, {origin})",
            )
            return ("static_indices",)
        if name == "vector.store":
            resolved = self._resolved(operation)
            origin = self._mixed_index_record(
                resolved,
                dynamic_field="indices",
                static_field="static_indices",
            )
            value = self.expression(resolved.value_id("value"))
            view = self.expression(resolved.value_id("view"))
            self.writer.line(f"loom::vector::store({value}, {view}, {origin});")
            return ("static_indices",)
        if name == "vector.fragment.store":
            resolved = self._resolved(operation)
            role = resolved.attr("role")
            shape = self._fragment_shape(operation, resolved)
            template, origin = self._fragment_origin(resolved)
            value = self.expression(resolved.value_id("value"))
            view = self.expression(resolved.value_id("view"))
            arguments = ", ".join([value, view, *origin, shape])
            self.writer.line(
                "loom::vector::fragment::store<"
                f"loom::vector::fragment::role::{role}{template}>("
                f"{arguments});"
            )
            return ("role", "static_indices")
        return None

    def _render_kernel_op(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name in {
            "kernel.workgroup.id",
            "kernel.workgroup.size",
            "kernel.workgroup.count",
            "kernel.workitem.id",
        }:
            dimension = operation.attributes.get("dimension")
            category, member = {
                "kernel.workgroup.id": ("workgroup", "id"),
                "kernel.workgroup.size": ("workgroup", "size"),
                "kernel.workgroup.count": ("workgroup", "count"),
                "kernel.workitem.id": ("workitem", "id"),
            }[name]
            expression = f"loom::kernel::{category}::{member}.{dimension}"
            self._declare(
                operation.results[0],
                expression,
                declaration_type="auto",
                expression_type=KERNEL_COORDINATE_CPP_TYPE,
                inlineable=True,
            )
            return ("dimension",)
        if name in {
            "kernel.subgroup.id",
            "kernel.subgroup.count",
            "kernel.subgroup.size",
            "kernel.subgroup.lane.id",
        }:
            member = {
                "kernel.subgroup.id": "id",
                "kernel.subgroup.count": "count",
                "kernel.subgroup.size": "size",
                "kernel.subgroup.lane.id": "lane_id",
            }[name]
            expression = f"loom::kernel::subgroup::{member}()"
            self._declare(
                operation.results[0],
                expression,
                declaration_type="auto",
                expression_type=KERNEL_COORDINATE_CPP_TYPE,
                inlineable=True,
            )
            return ()
        if name in {
            "kernel.subgroup.vote.any",
            "kernel.subgroup.vote.all",
        }:
            member = "any" if name.endswith(".any") else "all"
            expression = (
                f"loom::kernel::subgroup::vote::{member}("
                f"{self.expression(operation.operands[0])})"
            )
            self._declare(operation.results[0], expression, inlineable=True)
            return ()
        if name == "kernel.subgroup.vote.ballot":
            result = operation.results[0]
            expression = (
                "loom::kernel::subgroup::vote::ballot<"
                f"{self.cpp_type(result)}>("
                f"{self.expression(operation.operands[0])})"
            )
            self._declare(result, expression, inlineable=True)
            return ()
        if name == "kernel.subgroup.active.mask":
            result = operation.results[0]
            expression = (
                f"loom::kernel::subgroup::active_mask<{self.cpp_type(result)}>()"
            )
            self._declare(result, expression, inlineable=True)
            return ()
        if name == "kernel.subgroup.broadcast":
            resolved = self._resolved(operation)
            lane = resolved.value_id("lane")
            lane_expression = self.expression(lane)
            if self.expression_type(lane) != "unsigned":
                lane_expression = f"static_cast<unsigned>({lane_expression})"
            expression = (
                "loom::kernel::subgroup::broadcast("
                f"{self.expression(resolved.value_id('value'))}, "
                f"{lane_expression})"
            )
            self._declare(resolved.value_id("result"), expression)
            return ()
        if name == "kernel.subgroup.broadcast.first":
            resolved = self._resolved(operation)
            expression = (
                "loom::kernel::subgroup::broadcast_first("
                f"{self.expression(resolved.value_id('value'))})"
            )
            self._declare(resolved.value_id("result"), expression)
            return ()
        if name == "kernel.barrier":
            space = operation.attributes.get("memory_space")
            scope = operation.attributes.get("scope")
            ordering = operation.attributes.get("ordering")
            self.writer.template_call(
                "loom::kernel::barrier",
                [
                    _memory_space(space),
                    f"loom::atomic::scope::{scope}",
                    f"loom::atomic::ordering::{ordering}",
                ],
                [],
            )
            return ("memory_space", "scope", "ordering")
        if name == "kernel.async.gather":
            resolved = self._resolved(operation)
            scope = resolved.attr("cache_scope")
            temporal = resolved.attr("cache_temporal")
            expression = (
                f"loom::kernel::async::gather<loom::cache::scope::{scope}, "
                f"loom::cache::temporal::{temporal}>("
                f"{self.expression(resolved.value_id('source'))}, "
                f"{self.expression(resolved.value_id('dest'))})"
            )
            self._declare(resolved.value_id("token"), expression)
            return ("cache_scope", "cache_temporal")
        if name == "kernel.async.group":
            arguments = ", ".join(
                self.expression(operand) for operand in operation.operands
            )
            self._declare(
                operation.results[0],
                f"loom::kernel::async::group({arguments})",
            )
            return ()
        if name == "kernel.async.wait":
            resolved = self._resolved(operation)
            newer = resolved.attr("newer_groups")
            self.writer.line(
                f"loom::kernel::async::wait<{newer}>({self.expression(resolved.value_id('group'))});"
            )
            return ("newer_groups",)
        if name == "kernel.return":
            if operation.operands:
                values = ", ".join(
                    self.expression(value) for value in operation.operands
                )
                self.writer.line(f"return {{{values}}};")
            return ()
        return None

    def _render_for(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        initial = resolved.value_ids("iter_args")
        results = resolved.value_ids("results")
        if len(initial) != len(results):
            raise CxxRenderError(operation, "loop-carried result arity is inconsistent")
        for result, source in zip(results, initial, strict=True):
            source_expression = self.expression(source)
            source_users = self.users[source]
            preferred_name = loop_carrier_stem(self.names[result])
            if (
                preferred_name != self.names[result]
                and preferred_name not in self.expressions.values()
            ):
                self.names[result] = preferred_name
            can_reuse_source = (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source_expression) is not None
                and loop_carrier_stem(self.names[result]) == source_expression
                and source_users
                and all(user is operation for user in source_users)
            )
            if can_reuse_source:
                self.expressions[result] = source_expression
                self.expression_types[result] = self.expression_type(source)
            else:
                self._declare(result, self._loop_initializer(source))

        attributes: list[str] = []
        unroll = resolved.value_ids("unroll_factor")
        if unroll:
            attributes.append(f"loom::unroll({self.expression(unroll[0])})")
        elif resolved.attr("unroll_policy") is not None:
            attributes.append("loom::unroll")
        pipeline = resolved.value_ids("pipeline_depth")
        if pipeline:
            attributes.append(f"loom::pipeline({self.expression(pipeline[0])})")
        schedule = resolved.attr("unroll_schedule")
        if schedule is not None:
            attributes.append(f'loom::schedule("{schedule}")')
        if attributes:
            self.writer.attributes(attributes)

        body = resolved.region("body")
        if body is None or len(body.blocks) != 1:
            raise CxxRenderError(
                operation, "scf.for requires one structured body block"
            )
        block = body.blocks[0]
        if len(block.arg_ids) != 1 + len(results):
            raise CxxRenderError(
                operation, "scf.for block argument arity is inconsistent"
            )
        induction = block.arg_ids[0]
        preferred_induction = loop_carrier_stem(self.names[induction])
        if (
            preferred_induction != self.names[induction]
            and preferred_induction not in self.expressions.values()
        ):
            self.names[induction] = preferred_induction
        induction_name = self.names[induction]
        lower = self.expression(resolved.value_id("lower_bound"))
        upper = self.expression(resolved.value_id("upper_bound"))
        step = self.expression(resolved.value_id("step"))
        loop_type = "auto"
        lower_id = resolved.value_id("lower_bound")
        upper_id = resolved.value_id("upper_bound")
        step_id = resolved.value_id("step")
        step_constant = self._constant_integer(step_id)
        upper_constant = self._constant_integer(upper_id)
        counted_bounds: list[str | None] = [None, None, None]
        induction_domain = repr(self.module.values[induction].type)
        if induction_domain == "index" and (
            step_constant is not None
            and 0 < step_constant <= (1 << 31) - 1
            and (
                (
                    upper_constant is not None
                    and upper_constant <= (1 << 31) - step_constant
                )
                or (upper_constant is None and step_constant == 1)
            )
        ):
            counted_bounds = [
                self._signed_counted_bound(value_id)
                for value_id in (lower_id, upper_id, step_id)
            ]
            if all(bound is not None for bound in counted_bounds):
                loop_type = "int"
        elif induction_domain == "offset" and (
            step_constant is not None
            and 0 < step_constant <= (1 << 32) - 1
            and (
                (
                    upper_constant is not None
                    and upper_constant <= (1 << 32) - step_constant
                )
                or (upper_constant is None and step_constant == 1)
            )
        ):
            counted_bounds = [
                self._unsigned_counted_bound(value_id)
                for value_id in (lower_id, upper_id, step_id)
            ]
            if all(bound is not None for bound in counted_bounds):
                loop_type = "unsigned"
        if loop_type in {"int", "unsigned"}:
            lower, upper, step = (str(bound) for bound in counted_bounds)
        else:
            loop_type = self.cpp_type(induction)
            lower = self._address_operand(lower_id)
            upper = self._address_operand(upper_id)
            step = self._address_operand(step_id)
        if attributes and loop_type not in {"int", "unsigned"}:
            raise CxxRenderError(
                operation,
                "scheduled loop has no recoverable nonwrapping integer C++ "
                "induction contract",
            )
        increment = (
            f"++{induction_name}"
            if loop_type in {"int", "unsigned"} and step in {"1", "1u", "1ull"}
            else (
                f"{induction_name} += {step}"
                if loop_type in {"int", "unsigned"}
                else f"{induction_name} = {induction_name} + {step}"
            )
        )
        self.expressions[induction] = induction_name
        self.expression_types[induction] = loop_type
        for block_arg, result in zip(block.arg_ids[1:], results, strict=True):
            self.expressions[block_arg] = self.expression(result)
            self.expression_types[block_arg] = self.expression_type(result)
        self.writer.open(
            f"for ({loop_type} {induction_name} = {lower}; "
            f"{induction_name} < {upper}; {increment})"
        )
        self._render_block(block, yield_targets=results)
        self.writer.close()

    def _while_result_initializer(self, result: int, initial: list[int]) -> str:
        result_type = repr(self.module.values[result].type)
        for source in initial:
            if self._compatible_cxx_carrier(result, source):
                return self._loop_initializer(source)
        value_type = self.module.values[result].type
        if isinstance(value_type, ScalarType):
            if result_type in {"index", "offset"}:
                return self._literal(0, result)
            return f"{self.cpp_type(result)}{{}}"
        if (
            isinstance(value_type, ShapedType)
            and value_type.type_kind == TypeKind.VECTOR
        ):
            return f"{self.cpp_type(result)}{{}}"
        raise CxxRenderError(
            None,
            f"while result type {value_type!r} has no initializable C++ carrier",
        )

    def _assign_values(
        self, targets: Sequence[int | None], sources: Sequence[int]
    ) -> None:
        assignments = [
            (self.expression(target), self.expression(source))
            for target, source in zip(targets, sources, strict=True)
            if target is not None and self.expression(target) != self.expression(source)
        ]
        if len(assignments) == 1:
            target, source = assignments[0]
            self.writer.line(f"{target} = {_strip_outer_parentheses(source)};")
            return
        temporaries: list[tuple[str, str]] = []
        for target, source in assignments:
            temporary = self._temporary_name(
                f"next_{clean_identifier(target, 'value')}"
            )
            self.writer.line(f"auto {temporary} = {_strip_outer_parentheses(source)};")
            temporaries.append((target, temporary))
        for target, temporary in temporaries:
            self.writer.line(f"{target} = {temporary};")

    def _while_forward_only_carriers(
        self, before: Block, after: Block, results: Sequence[int]
    ) -> set[int]:
        """Finds loop columns represented only by structured forwarding."""

        if not before.ops or not after.ops:
            return set()
        condition = before.ops[-1]
        yield_operation = after.ops[-1]
        if condition.name != "scf.condition" or yield_operation.name != "scf.yield":
            return set()
        if len(self._resolved(condition).value_ids("forwarded")) != len(results) or len(
            yield_operation.operands
        ) != len(results):
            return set()
        forward_only: set[int] = set()
        carriers = zip(before.arg_ids, after.arg_ids, results, strict=True)
        for index, (before_argument, after_argument, result) in enumerate(carriers):
            if (
                self.use_counts[result] != 0
                or self.view_construction_type_users[result]
            ):
                continue
            allowed_uses = {
                (id(condition), index + 1),
                (id(yield_operation), index),
            }
            before_closure = self._forwarding_closure(
                before_argument, allowed_uses, set()
            )
            after_closure = self._forwarding_closure(
                after_argument, allowed_uses, set()
            )
            if before_closure is None or after_closure is None:
                continue
            self.suppressed_forwarding_values.update(before_closure)
            self.suppressed_forwarding_values.update(after_closure)
            self.suppressed_forwarding_values.add(result)
            forward_only.add(index)
        return forward_only

    def _while_identity_condition(self, before: Block) -> int | None:
        """Returns an identity-header loop's condition block argument."""

        if len(before.ops) != 1 or before.ops[0].name != "scf.condition":
            return None
        condition = before.ops[0]
        resolved = self._resolved(condition)
        if resolved.value_ids("forwarded") != before.arg_ids:
            return None
        condition_id = resolved.value_id("condition")
        return condition_id if condition_id in before.arg_ids else None

    def _forwarding_closure(
        self,
        value_id: int,
        allowed_uses: set[tuple[int, int]],
        active: set[int],
    ) -> set[int] | None:
        """Returns values that only forward into the selected loop edge."""

        if self.view_construction_type_users[value_id] or value_id in active:
            return None
        nested_active = {*active, value_id}
        closure = {value_id}
        for user, operand_index in self.uses[value_id]:
            if (id(user), operand_index) in allowed_uses:
                continue
            parent = self.parent_operations.get(id(user))
            if (
                user.name != "scf.yield"
                or parent is None
                or parent.name != "scf.if"
                or operand_index >= len(parent.results)
            ):
                return None
            forwarded = self._forwarding_closure(
                parent.results[operand_index], allowed_uses, nested_active
            )
            if forwarded is None:
                return None
            closure.update(forwarded)
        return closure

    def _render_while(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        initial = resolved.value_ids("iter_args")
        results = resolved.value_ids("results")
        before_region = resolved.region("before")
        after_region = resolved.region("after")
        if before_region is None or len(before_region.blocks) != 1:
            raise CxxRenderError(
                operation, "scf.while before region must have one block"
            )
        if after_region is None or len(after_region.blocks) != 1:
            raise CxxRenderError(
                operation, "scf.while after region must have one block"
            )
        before = before_region.blocks[0]
        after = after_region.blocks[0]
        if len(before.arg_ids) != len(initial):
            raise CxxRenderError(
                operation, "scf.while before argument arity is inconsistent"
            )
        if len(after.arg_ids) != len(results):
            raise CxxRenderError(
                operation, "scf.while after argument arity is inconsistent"
            )

        claimed_names = set(self.expressions.values())
        for result in results:
            preferred_name = loop_carrier_stem(self.names[result])
            if (
                preferred_name != self.names[result]
                and preferred_name not in claimed_names
            ):
                self.names[result] = preferred_name
            claimed_names.add(self.names[result])

        forward_only = self._while_forward_only_carriers(before, after, results)
        shared_carriers = len(initial) == len(results) and all(
            index in forward_only or self._compatible_cxx_carrier(argument, result)
            for index, (argument, result) in enumerate(
                zip(before.arg_ids, results, strict=True)
            )
        )
        if shared_carriers:
            for index, (argument, result, source) in enumerate(
                zip(before.arg_ids, results, initial, strict=True)
            ):
                if index in forward_only:
                    continue
                self._declare(result, self._loop_initializer(source))
                self.expressions[argument] = self.expression(result)
                self.expression_types[argument] = self.expression_type(result)
            header_targets: list[int | None] = [
                None if index in forward_only else result
                for index, result in enumerate(results)
            ]
        else:
            for index, (argument, source) in enumerate(
                zip(before.arg_ids, initial, strict=True)
            ):
                if index in forward_only:
                    continue
                self._declare(argument, self._loop_initializer(source))
            for index, result in enumerate(results):
                if index in forward_only:
                    continue
                self._declare(result, self._while_result_initializer(result, initial))
            header_targets = [
                None if index in forward_only else argument
                for index, argument in enumerate(before.arg_ids)
            ]

        for index, (argument, result) in enumerate(
            zip(after.arg_ids, results, strict=True)
        ):
            if index in forward_only:
                continue
            self.expressions[argument] = self.expression(result)
            self.expression_types[argument] = self.expression_type(result)

        identity_condition = self._while_identity_condition(before)
        if shared_carriers and identity_condition is not None:
            self._validate_projected_attributes(before.ops[0], ())
            condition = _strip_outer_parentheses(self.expression(identity_condition))
            self.writer.open(f"while ({condition})")
            self._render_block(after, yield_targets=header_targets)
            self.writer.close()
            return

        continuation = self._temporary_name("loop_continues")
        self.writer.line(f"bool {continuation} = true;")
        self.writer.open(f"while ({continuation})")
        self._render_block(
            before,
            condition_target=continuation,
            condition_forward_targets=[
                None if index in forward_only else result
                for index, result in enumerate(results)
            ],
        )
        self.writer.open(f"if ({continuation})")
        self._render_block(after, yield_targets=header_targets)
        self.writer.close()
        self.writer.close()

    def _render_if(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        results = resolved.value_ids("results")
        for result in results:
            if result in self.suppressed_forwarding_values:
                continue
            assignment_target_id = self.assignment_targets.get(result)
            if assignment_target_id is not None:
                # A surrounding structured operation already owns the mutable
                # carrier. Each branch assigns or preserves that value; a
                # default assignment here would destroy its incoming state.
                self.expressions[result] = self.expression(assignment_target_id)
                self.expression_types[result] = self.expression_type(
                    assignment_target_id
                )
            else:
                self._declare(result, f"{self.cpp_type(result)}{{}}")
        then_region = resolved.region("then_region")
        else_region = resolved.region("else_region")
        if then_region is None or len(then_region.blocks) != 1:
            raise CxxRenderError(operation, "scf.if then region must have one block")
        condition = _strip_outer_parentheses(
            self.expression(resolved.value_id("condition"))
        )
        yield_targets: list[int | None] = [
            None if result in self.suppressed_forwarding_values else result
            for result in results
        ]
        self.writer.open(f"if ({condition})")
        self._render_block(then_region.blocks[0], yield_targets=yield_targets)
        if else_region is None:
            self.writer.close()
            return
        if len(else_region.blocks) != 1:
            raise CxxRenderError(operation, "scf.if else region must have one block")
        else_block = else_region.blocks[0]
        if len(else_block.ops) == 1 and else_block.ops[0].name == "scf.yield":
            yield_operation = else_block.ops[0]
            self._validate_projected_attributes(yield_operation, ())
            if len(yield_operation.operands) == len(results) and all(
                target is None or self.expression(source) == self.expression(target)
                for source, target in zip(
                    yield_operation.operands, yield_targets, strict=True
                )
            ):
                self.writer.close()
                return
        self.writer.continue_block("else")
        self._render_block(else_block, yield_targets=yield_targets)
        self.writer.close()

    def _direct_yield_assignments(
        self, block: Block, yield_targets: Sequence[int | None] | None
    ) -> dict[int, int]:
        if yield_targets is None or not block.ops:
            return {}
        yield_operation = block.ops[-1]
        if yield_operation.name != "scf.yield" or len(yield_operation.operands) != len(
            yield_targets
        ):
            return {}
        positions = {id(operation): index for index, operation in enumerate(block.ops)}
        assignments: dict[int, int] = {}
        for source, target in zip(yield_operation.operands, yield_targets, strict=True):
            if target is None:
                continue
            definition = self.definitions.get(source)
            target_expression = self.expression(target)
            if (
                definition is None
                or self.definition_blocks.get(source) is not block
                or self.use_counts[source] != 1
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target_expression) is None
            ):
                continue
            definition_position = positions[id(definition)]
            carrier_values = {
                value_id
                for value_id, expression in self.expressions.items()
                if expression == target_expression
            }
            if any(
                carrier_values.intersection(operation.operands)
                for operation in block.ops[definition_position + 1 : -1]
            ):
                continue
            assignments[source] = target
        return assignments

    def _render_block(
        self,
        block: Block,
        *,
        yield_targets: Sequence[int | None] | None = None,
        condition_target: str | None = None,
        condition_forward_targets: Sequence[int | None] | None = None,
        launch_configuration: bool = False,
    ) -> None:
        direct_assignments = self._direct_yield_assignments(block, yield_targets)
        previous_assignments = self.assignment_targets
        self.assignment_targets = {**previous_assignments, **direct_assignments}
        try:
            for operation in block.ops:
                if operation.name == "kernel.launch.config":
                    self._validate_projected_attributes(operation, ())
                    if not launch_configuration:
                        raise CxxRenderError(
                            operation,
                            "launch configuration terminator appears outside its "
                            "kernel configuration",
                        )
                    self._render_launch_configuration_return(operation)
                    continue
                if operation.name == "func.return":
                    self._validate_projected_attributes(operation, ())
                    if not operation.operands:
                        self.writer.line("return;")
                    elif len(operation.operands) == 1:
                        self.writer.line(
                            f"return {self.expression(operation.operands[0])};"
                        )
                    else:
                        values = ", ".join(
                            self.expression(value) for value in operation.operands
                        )
                        self.writer.line(f"return {{{values}}};")
                    continue
                if operation.name == "func.call":
                    projected = self._render_function_call(operation)
                    self._validate_projected_attributes(operation, projected)
                    continue
                if operation.name == "scf.yield":
                    self._validate_projected_attributes(operation, ())
                    if yield_targets is None:
                        raise CxxRenderError(
                            operation,
                            "yield appears outside a projected structured operation",
                        )
                    if len(operation.operands) != len(yield_targets):
                        raise CxxRenderError(
                            operation, "yield/result arity is inconsistent"
                        )
                    self._assign_values(yield_targets, operation.operands)
                    continue
                if operation.name == "scf.condition":
                    self._validate_projected_attributes(operation, ())
                    if condition_target is None or condition_forward_targets is None:
                        raise CxxRenderError(
                            operation,
                            "condition appears outside a projected while operation",
                        )
                    resolved = self._resolved(operation)
                    forwarded = resolved.value_ids("forwarded")
                    if len(forwarded) != len(condition_forward_targets):
                        raise CxxRenderError(
                            operation, "condition/result arity is inconsistent"
                        )
                    condition = _strip_outer_parentheses(
                        self.expression(resolved.value_id("condition"))
                    )
                    self.writer.line(f"{condition_target} = {condition};")
                    self._assign_values(condition_forward_targets, forwarded)
                    continue
                if operation.name == "scf.for":
                    self._validate_projected_attributes(
                        operation, ("unroll_policy", "unroll_schedule")
                    )
                    self._render_for(operation)
                    continue
                if operation.name == "scf.if":
                    self._validate_projected_attributes(operation, ())
                    self._render_if(operation)
                    continue
                if operation.name == "scf.while":
                    self._validate_projected_attributes(operation, ())
                    self._render_while(operation)
                    continue
                projected = self._render_command_op(operation)
                if projected is not None:
                    self._validate_projected_attributes(operation, projected)
                    continue
                projected = self._render_simple(operation)
                if projected is not None:
                    self._validate_projected_attributes(operation, projected)
                    continue
                projected = self._render_memory(operation)
                if projected is not None:
                    self._validate_projected_attributes(operation, projected)
                    continue
                projected = self._render_kernel_op(operation)
                if projected is not None:
                    self._validate_projected_attributes(operation, projected)
                    continue
                raise CxxRenderError(
                    operation, "operation has no validated C++ projection"
                )
        finally:
            self.assignment_targets = previous_assignments

    def _parameter_element(self, root: int) -> str:
        elements = self.buffer_info[root].element_types
        if not elements or len(elements) != 1:
            return "unsigned char"
        name = next(iter(elements))
        scalar = next(
            (
                value.type.element_type
                for value in self.module.values
                if isinstance(value.type, ShapedType)
                and repr(value.type.element_type) == name
            ),
            None,
        )
        if scalar is None:
            return "unsigned char"
        signed = any(
            self.view_roots.get(view_id) == root for view_id in self.view_signed
        )
        return self.types.scalar(scalar, signed)

    def _render_kernel(self, kernel: Operation) -> None:
        scope = self.operation_scopes[id(kernel)]
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        self._validate_projected_attributes(kernel, ("callee", "target"))
        self._analyze_buffers(kernel)
        resolved = self._resolved(kernel)
        body = resolved.region("body")
        assert body is not None
        entry = body.blocks[0]
        attributes = self._render_kernel_configuration(kernel)
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        target_symbol = resolved.attr("target")
        target_info = self.targets.get(str(target_symbol)) if target_symbol else None
        if target_symbol is not None and target_info is None:
            raise CxxRenderError(
                kernel, f"target record {target_symbol!s} is not present in the module"
            )
        if target_info is not None and target_info.operation_name != "amdgpu.target":
            raise CxxRenderError(
                target_info.operation,
                f"target record {target_info.operation_name} has no C++ "
                "declaration projection",
            )
        if target_info is not None:
            attributes.append(f"loom::target({self.target_names[str(target_symbol)]})")
        self.writer.attributes(attributes)

        parameters: list[str] = []
        buffer_elements: list[str | None] = []
        for argument in entry.arg_ids:
            root = self.buffer_roots.get(argument)
            if root is not None:
                if root != argument:
                    raise CxxRenderError(
                        kernel, "kernel buffer argument aliases another storage root"
                    )
                info = self.buffer_info[root]
                qualifiers: list[str] = []
                if info.noalias:
                    qualifiers.append("loom::noalias")
                if info.alignment is not None:
                    qualifiers.append(f"loom::assume_aligned({info.alignment})")
                attribute = f"[[{', '.join(qualifiers)}]] " if qualifiers else ""
                const = "" if info.mutable else "const "
                element = self._parameter_element(root)
                parameters.append(f"{attribute}{const}{element}* {info.name}")
                buffer_elements.append(f"{const}{element}")
                self.expressions[argument] = info.name
                continue
            value_type = self.module.values[argument].type
            if not isinstance(value_type, ScalarType):
                raise CxxRenderError(
                    kernel,
                    f"kernel argument type {value_type!r} has no C++ ABI projection",
                )
            name = self.names[argument]
            type_name = self.cpp_type(argument)
            parameters.append(f"{type_name} {name}")
            buffer_elements.append(None)
            self.expressions[argument] = name
            self.expression_types[argument] = type_name
        symbol = str(resolved.attr("callee"))
        self.kernel_buffer_elements[symbol] = tuple(buffer_elements)
        if parameters:
            self.writer.open(f"void {symbol}({', '.join(parameters)})")
        else:
            self.writer.open(f"void {symbol}()")
        self._render_block(entry)
        self.writer.close()

    def _function_result_type(self, function: Operation) -> str:
        results = function.results
        if not results:
            return "void"
        if len(results) == 1:
            return self.cpp_type(results[0])
        symbol = str(self._resolved(function).attr("callee"))
        fields = tuple(
            (f"value{index}", self.cpp_type(result))
            for index, result in enumerate(results)
        )
        return self._record_type(
            clean_identifier(symbol, "function") + "_result", fields
        )

    def _render_function_call(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        callee = self.function_names.get(symbol)
        if callee is None:
            raise CxxRenderError(
                operation, f"called function {symbol!r} is not present in the module"
            )
        expected_buffers = self.function_buffer_elements.get(symbol)
        arguments_list: list[str] = []
        for index, operand in enumerate(operation.operands):
            expression = self.expression(operand)
            expected_element = (
                expected_buffers[index]
                if expected_buffers is not None and index < len(expected_buffers)
                else None
            )
            if expected_element is not None:
                expected_type = f"loom::type::buffer<{expected_element}>"
                if self.expression_type(operand) != expected_type:
                    expression = f"loom::buffer::cast<{expected_element}>({expression})"
            arguments_list.append(expression)
        arguments = ", ".join(arguments_list)
        call = f"{callee}({arguments})"
        if not operation.results:
            self.writer.line(f"{call};")
        elif len(operation.results) == 1:
            self._declare(operation.results[0], call)
        else:
            fields = tuple(
                (f"value{index}", self.cpp_type(result))
                for index, result in enumerate(operation.results)
            )
            record_type = self._record_type(
                clean_identifier(symbol, "function") + "_result", fields
            )
            record_name = self._temporary_name("call_result")
            self.writer.line(f"{record_type} {record_name} = {call};")
            for index, result in enumerate(operation.results):
                self._declare(result, f"{record_name}.value{index}")
        projected_attributes = ["callee"]
        effect_contract = resolved.attr("effect_contract")
        if effect_contract == "pure":
            # A C++ call carries purity from the callable definition. The
            # importer reconstructs the explicit func.call contract after the
            # emitted definition is parsed and analyzed.
            projected_attributes.append("effect_contract")
        elif effect_contract is not None:
            raise CxxRenderError(
                operation,
                "function call effect contract "
                f"{effect_contract!r} has no C++ projection",
            )
        return tuple(projected_attributes)

    def _function_attributes(self, function: Operation) -> list[str]:
        resolved = self._resolved(function)
        symbol = str(resolved.attr("callee"))
        attributes = [f'loom::symbol("{symbol}")']
        inline_policy = resolved.attr("inline_policy")
        if inline_policy == "inline":
            attributes.append("loom::force_inline")
        elif inline_policy is not None:
            raise CxxRenderError(
                function,
                f"function inline policy {inline_policy!r} has no C++ projection",
            )
        return attributes

    def _function_signature(self, function: Operation, *, bind_parameters: bool) -> str:
        resolved = self._resolved(function)
        body = resolved.region("body")
        if body is None or len(body.blocks) != 1:
            raise CxxRenderError(
                function, "function body must have one structured entry block"
            )
        entry = body.blocks[0]
        parameters: list[str] = []
        buffer_elements: list[str | None] = []
        for argument in entry.arg_ids:
            value_type = self.module.values[argument].type
            if isinstance(value_type, BufferType):
                element = self._parameter_element(argument)
                type_name = f"loom::type::buffer<{element}>"
                buffer_elements.append(element)
            else:
                type_name = self.cpp_type(argument)
                buffer_elements.append(None)
            name = self.names[argument]
            parameters.append(f"{type_name} {name}")
            if bind_parameters:
                self.expressions[argument] = name
                self.expression_types[argument] = type_name

        symbol = str(resolved.attr("callee"))
        self.function_buffer_elements[symbol] = tuple(buffer_elements)
        visibility = resolved.attr("visibility")
        storage = "" if visibility == "public" else "static "
        return (
            f"{storage}{self._function_result_type(function)} "
            f"{self.function_names[symbol]}(" + ", ".join(parameters) + ")"
        )

    def _render_function_declaration(self, function: Operation) -> None:
        self._analyze_buffers(function)
        self.writer.attributes(self._function_attributes(function))
        self.writer.line(
            self._function_signature(function, bind_parameters=False) + ";"
        )

    def _render_function(self, function: Operation) -> None:
        scope = self.operation_scopes[id(function)]
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        self._validate_projected_attributes(
            function, ("callee", "visibility", "inline_policy")
        )
        self._analyze_buffers(function)
        body = self._resolved(function).region("body")
        assert body is not None
        self.writer.attributes(self._function_attributes(function))
        self.writer.open(self._function_signature(function, bind_parameters=True))
        entry = body.blocks[0]
        self._render_block(entry)
        self.writer.close()

    def _plan_callable_names(
        self,
        kernels: list[Operation],
        functions: list[Operation],
        commands: list[Operation],
        pipelines: list[Operation],
    ) -> None:
        used_names = {str(self._resolved(kernel).attr("callee")) for kernel in kernels}
        for function in functions:
            symbol = str(self._resolved(function).attr("callee"))
            base = clean_identifier(symbol, "function")
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.function_names[symbol] = name

        for command in commands:
            symbol = str(self._resolved(command).attr("callee"))
            if symbol in self.command_names:
                continue
            base = clean_identifier(symbol, "command_program")
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.command_names[symbol] = name

        for pipeline in pipelines:
            symbol = str(self._resolved(pipeline).attr("callee"))
            base = clean_identifier(symbol, "pipeline")
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.pipeline_names[symbol] = name
            self.pipeline_declarations[symbol] = pipeline

        for kernel in kernels:
            symbol = str(self._resolved(kernel).attr("callee"))
            base = clean_identifier(symbol, "kernel") + "_configuration"
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.kernel_configuration_names[symbol] = name

        target_symbols: list[str] = []
        for callable_operation in (*kernels, *commands, *pipelines):
            resolved = self._resolved(callable_operation)
            target_symbol = resolved.attr("target")
            if target_symbol is not None:
                symbol = str(target_symbol)
                if symbol and symbol not in target_symbols:
                    target_symbols.append(symbol)

        for symbol in target_symbols:
            base = (
                "rendered_target"
                if len(target_symbols) == 1
                else f"rendered_target_{clean_identifier(symbol, 'target')}"
            )
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.target_names[symbol] = name

    def render(self) -> str:
        kernels = [
            operation
            for operation in self.module.body.ops
            if operation.name == "kernel.def"
        ]
        functions = [
            operation
            for operation in self.module.body.ops
            if operation.name == "func.def"
        ]
        commands = [
            operation
            for operation in self.module.body.ops
            if operation.name in {"command.def", "command.decl"}
        ]
        pipelines = [
            operation
            for operation in self.module.body.ops
            if operation.name == "pipeline.decl"
        ]
        command_definitions = [
            operation for operation in commands if operation.name == "command.def"
        ]
        if not kernels and not functions and not commands and not pipelines:
            raise CxxRenderError(
                None,
                "module contains no callable definitions",
            )
        for operation in self.module.body.ops:
            if operation.name in {
                "kernel.def",
                "func.def",
                "command.def",
                "command.decl",
                "pipeline.decl",
                "config.decl",
            } or operation.name.endswith(".target"):
                continue
            raise CxxRenderError(
                operation,
                "a top-level definition would be omitted from the C++ projection",
            )
        self._plan_callable_names(kernels, functions, commands, pipelines)
        for index, function in enumerate(functions):
            if index:
                self.writer.line()
            self._render_function_declaration(function)
        if functions:
            self.writer.line()
        for index, pipeline in enumerate(pipelines):
            if index:
                self.writer.line()
            self._render_pipeline_declaration(pipeline)
        if pipelines:
            self.writer.line()
        for index, command in enumerate(commands):
            if index:
                self.writer.line()
            if command.name == "command.def":
                body = self._resolved(command).region("body")
                assert body is not None
                self._validate_projected_attributes(
                    command,
                    (
                        "callee",
                        "visibility",
                        "target",
                        "predicates",
                        "specialization_count",
                    ),
                )
                self.writer.attributes(self._command_attributes(command))
                self.writer.line(
                    self._command_signature(command, list(body.blocks[0].arg_ids)) + ";"
                )
            else:
                self._render_command_declaration(command)
        if commands:
            self.writer.line()
        callables = [*functions, *kernels, *command_definitions]
        for index, callable_operation in enumerate(callables):
            if index:
                self.writer.line()
            if callable_operation.name == "func.def":
                self._render_function(callable_operation)
            elif callable_operation.name == "command.def":
                self._render_command_program(callable_operation)
            else:
                self._render_kernel(callable_operation)

        for operation in self.module.body.ops:
            if (
                operation.name == "config.decl"
                and id(operation) not in self.projected_config_declarations
            ):
                raise CxxRenderError(
                    operation,
                    "configuration declaration is not owned by a projected "
                    "kernel launch contract",
                )

        operation_names = {
            operation.name for operation in walk_operations(self.module.body)
        }
        scalar_names: set[str] = set()
        for value in self.module.values:
            value_type = value.type
            if isinstance(value_type, ScalarType):
                scalar_names.add(repr(value_type))
            elif isinstance(value_type, ShapedType):
                scalar_names.add(repr(value_type.element_type))
        uses_predicate_functions = any(
            isinstance(
                predicates := operation.attributes.get("predicates"), PredicateListAttr
            )
            and any(
                any(
                    argument.tag == "value"
                    and isinstance(self.module.values[argument.value].type, ScalarType)
                    and repr(self.module.values[argument.value].type)
                    in {"index", "offset"}
                    for argument in predicate.args
                )
                for predicate in predicates
            )
            for operation in self.module.body.ops
            if id(operation) not in self.projected_config_declarations
        )
        preamble = render_preamble(
            operation_names=operation_names,
            scalar_names=scalar_names,
            has_kernels=bool(kernels or pipelines),
            uses_buffers=any(
                isinstance(value.type, BufferType) for value in self.module.values
            ),
            uses_views=any(
                isinstance(value.type, ShapedType)
                and value.type.type_kind == TypeKind.VIEW
                for value in self.module.values
            ),
            vector_aliases=self.types.vector_aliases,
            record_declarations=self.record_declarations,
            table_lookup_declarations=self.table_lookup_declarations,
            targets=tuple(
                (target_name, self.targets[target_symbol])
                for target_symbol, target_name in self.target_names.items()
            ),
            uses_scalar_functions=self.uses_scalar_functions,
            uses_numeric_conversion=self.uses_numeric_conversion,
            uses_predicate_functions=uses_predicate_functions,
        )
        return "\n".join((*preamble, *self.writer.lines)) + "\n"


def _select_callable_roots(module: Module, roots: Sequence[str]) -> Module:
    """Returns selected callables, their helper closure, and kernel targets."""

    normalized_roots: list[str] = []
    seen_roots: set[str] = set()
    for root in roots:
        name = root.removeprefix("@")
        if not name:
            raise CxxRenderError(None, "callable root name is empty")
        if name in seen_roots:
            raise CxxRenderError(None, f"callable root '@{name}' is repeated")
        seen_roots.add(name)
        normalized_roots.append(name)

    symbols = {symbol.name: symbol for symbol in module.symbols}
    selected_operation_ids: set[int] = set()
    target_names: set[str] = set()
    pending = [(name, True) for name in normalized_roots]
    while pending:
        name, require_definition = pending.pop()
        symbol = symbols.get(name)
        if symbol is None or symbol.op is None:
            raise CxxRenderError(None, f"callable root '@{name}' was not found")
        callable_operations = {
            "kernel.def",
            "func.def",
            "command.def",
            "command.decl",
            "pipeline.decl",
        }
        if symbol.op.name not in callable_operations or (
            require_definition and symbol.op.name in {"command.decl", "pipeline.decl"}
        ):
            raise CxxRenderError(
                symbol.op,
                f"root '@{name}' is not a callable definition",
            )
        if id(symbol.op) in selected_operation_ids:
            continue
        selected_operation_ids.add(id(symbol.op))
        target = symbol.op.attributes.get("target")
        if isinstance(target, str):
            target_names.add(target)
        for region in symbol.op.regions:
            for block in region.blocks:
                for operation in walk_operations(block):
                    if operation.name == "config.get":
                        config = operation.attributes.get("config")
                        config_definition = (
                            symbols.get(config) if isinstance(config, str) else None
                        )
                        if (
                            config_definition is not None
                            and config_definition.op is not None
                        ):
                            selected_operation_ids.add(id(config_definition.op))
                        continue
                    if operation.name not in {
                        "func.call",
                        "command.launch",
                        "kernel.launch",
                    }:
                        continue
                    callee = operation.attributes.get("callee")
                    if isinstance(callee, str):
                        pending.append((callee, False))

    selected_operations = [
        operation
        for operation in module.body.ops
        if id(operation) in selected_operation_ids
        or (
            operation.name.endswith(".target")
            and operation.attributes.get("symbol") in target_names
        )
    ]
    return replace(module, body=replace(module.body, ops=selected_operations))


def render_module(module: Module, *, roots: Sequence[str] = ()) -> str:
    """Renders supported High IR callable roots as standalone Loom C++."""
    projected_module = _select_callable_roots(module, roots) if roots else module
    return CxxRenderer(projected_module).render()
