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

from loom.format.cxx.buffer_access import BufferAccessPlan
from loom.format.cxx.context import (
    _memory_space,
    _strip_outer_parentheses,
    cxx_string_literal,
    is_pure_operation,
    is_target_operation,
    walk_operations,
)
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.functions import CxxFunctionRenderer
from loom.format.cxx.naming import clean_identifier, loop_carrier_stem
from loom.format.cxx.translation_unit import render_preamble
from loom.format.cxx.types import KERNEL_COORDINATE_CPP_TYPE
from loom.format.cxx.value_transport import plan_value_transport
from loom.format.cxx.view_access import plan_view_access
from loom.ir import (
    Block,
    BufferType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    Module,
    Operation,
    ParameterizedType,
    PoolType,
    PredicateListAttr,
    ScalarType,
    ShapedType,
    StaticDim,
    TypeKind,
)


class CxxRenderer(CxxFunctionRenderer):
    def _static_view_encoding_expression(
        self, operation: Operation, view_type: ShapedType
    ) -> str:
        encoding = view_type.encoding
        if not isinstance(encoding, EncodingInstance):
            raise CxxRenderError(
                operation, "view encoding is neither a static nor dynamic encoding"
            )
        if encoding.name == "encoding.layout.dense" and not encoding.params:
            return f"loom::encoding::layout::dense<{view_type.rank}>()"
        if encoding.name == "encoding.layout.strided":
            parameters = dict(encoding.params)
            strides = parameters.get("strides")
            if set(parameters) != {"strides"} or not isinstance(strides, tuple):
                raise CxxRenderError(
                    operation, "static strided layout has malformed stride metadata"
                )
            if len(strides) != view_type.rank or any(
                type(stride) is not int for stride in strides
            ):
                raise CxxRenderError(
                    operation,
                    "static strided layout requires one integer stride per view axis",
                )
            arguments = ", ".join(
                f"loom::type::index{{{stride}}}" for stride in strides
            )
            return f"loom::encoding::layout::strided({arguments})"
        raise CxxRenderError(
            operation,
            f"static view encoding {encoding.name!r} has no C++ value projection",
        )

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
        if isinstance(lhs_type, BufferType) and isinstance(rhs_type, BufferType):
            return self._buffer_element(lhs) == self._buffer_element(rhs)
        if repr(lhs_type) == repr(rhs_type):
            return True
        if not (
            isinstance(lhs_type, ShapedType)
            and lhs_type.type_kind in {TypeKind.VIEW, TypeKind.VECTOR, TypeKind.TILE}
            and isinstance(rhs_type, ShapedType)
            and rhs_type.type_kind == lhs_type.type_kind
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

    @staticmethod
    def _append_memory_access_contract(
        operation: Operation, selectors: list[str]
    ) -> tuple[str, ...]:
        flags = operation.attributes.get("memory_flags")
        if not flags:
            return ()
        members = {"volatile": "volatile_", "root_in_bounds": "root_in_bounds"}
        names = flags.split("|")
        if any(name not in members for name in names):
            return ()
        selectors.append(
            " | ".join(f"loom::memory_access::{members[name]}" for name in names)
        )
        return ("memory_flags",)

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
            selectors = [
                f"loom::atomic::ordering::{success}",
                f"loom::atomic::ordering::{failure}",
                f"loom::atomic::scope::{scope}",
            ]
            projected_flags = self._append_memory_access_contract(operation, selectors)
            template = ", ".join(selectors)
            expression = (
                f"loom::view::atomic::cmpxchg<{template}>("
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
                *projected_flags,
            )

        ordering = resolved.attr("ordering")
        if name == "view.atomic.load":
            selectors = [
                f"loom::atomic::ordering::{ordering}",
                f"loom::atomic::scope::{scope}",
            ]
            projected_flags = self._append_memory_access_contract(operation, selectors)
            template = ", ".join(selectors)
            expression = f"loom::view::atomic::load<{template}>({access})"
            self._emit_atomic_result(resolved.value_id("result"), expression)
            return (
                "ordering",
                "scope",
                "static_indices",
                *projected_flags,
            )
        if name == "view.atomic.store":
            selectors = [
                f"loom::atomic::ordering::{ordering}",
                f"loom::atomic::scope::{scope}",
            ]
            projected_flags = self._append_memory_access_contract(operation, selectors)
            template = ", ".join(selectors)
            self.writer.line(
                f"loom::view::atomic::store<{template}>("
                f"{self.expression(resolved.value_id('value'))}, {access});"
            )
            return (
                "ordering",
                "scope",
                "static_indices",
                *projected_flags,
            )

        kind = resolved.attr("kind")
        member = "rmw" if name == "view.atomic.rmw" else "reduce"
        selectors = [
            f"loom::atomic::kind::{kind}",
            f"loom::atomic::ordering::{ordering}",
            f"loom::atomic::scope::{scope}",
        ]
        projected_flags = self._append_memory_access_contract(operation, selectors)
        template = ", ".join(selectors)
        expression = (
            f"loom::view::atomic::{member}<{template}>("
            f"{self.expression(resolved.value_id('value'))}, {access})"
        )
        if name == "view.atomic.rmw":
            self._emit_atomic_result(resolved.value_id("result"), expression)
        else:
            self.writer.line(f"{expression};")
        return (
            "kind",
            "ordering",
            "scope",
            "static_indices",
            *projected_flags,
        )

    def _render_memory(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name == "buffer.null":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            element = self._buffer_element(result)
            self._declare(
                result,
                f"loom::buffer::null<{element}>()",
                declaration_type="auto",
                expression_type=f"loom::type::buffer<{element}>",
            )
            return ()
        if name == "buffer.compare.null":
            resolved = self._resolved(operation)
            operand = resolved.value_id("buffer")
            argument = self._buffer_argument(operand, self._buffer_element(operand))
            self._declare(
                resolved.value_id("result"),
                f"loom::buffer::compare::null({argument})",
            )
            return ()
        if name in {"buffer.compare.address", "buffer.distance"}:
            resolved = self._resolved(operation)
            arguments = []
            for side in ("lhs", "rhs"):
                root = resolved.value_id(side)
                arguments.extend(
                    (
                        self._buffer_argument(root, self._buffer_element(root)),
                        self.expression(resolved.value_id(f"{side}_offset")),
                    )
                )
            if name == "buffer.distance":
                result = resolved.value_id("result")
                self._declare(
                    result,
                    f"loom::buffer::distance<{self.cpp_type(result)}>("
                    f"{', '.join(arguments)})",
                )
                return ()
            predicate = resolved.attr("predicate")
            self._declare(
                resolved.value_id("result"),
                "loom::buffer::compare::address<"
                f"loom::buffer::compare::predicate::{predicate}>("
                f"{', '.join(arguments)})",
            )
            return ("predicate",)
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
            literal = cxx_string_literal(resolved.attr("contents"))
            self._declare(result, f"{callee}({literal})", declaration_type="auto*")
            self.expression_types[result] = (
                "const unsigned char*" if immutable else "unsigned char*"
            )
            return attributes
        if name == "buffer.alloca":
            resolved = self._resolved(operation)
            pools = resolved.value_ids("pool")
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
            arguments = [length]
            if pools:
                arguments.append(self.expression(pools[0]))
            expression = (
                f"loom::buffer::alloca<{template}>(" + ", ".join(arguments) + ")"
            )
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
                encoding = self._static_view_encoding_expression(operation, result_type)
            if result_type.alignment is not None:
                raise CxxRenderError(
                    operation,
                    "view access alignment has no C++ type projection",
                )
            root = self.buffer_roots.get(buffer_id)
            is_const = (
                not self.view_access.is_mutable(result)
                if self.view_access is not None
                else root is not None and not self.buffer_info[root].mutable
            )
            if not is_const and self.buffer_access.is_immutable(
                buffer_id, self.buffer_projection
            ):
                raise CxxRenderError(operation, "immutable buffer storage is written")
            element = self.types.scalar(
                result_type.element_type, result in self.view_signed
            )
            source_type = self.expression_type(buffer_id)
            arguments: list[str]
            if source_type.startswith("loom::type::buffer<"):
                qualified_element = f"const {element}" if is_const else element
                target_type = f"loom::type::buffer<{qualified_element}>"
                source = self.expression(buffer_id)
                if source_type != target_type:
                    source = f"loom::buffer::cast<{qualified_element}>({source})"
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
            selectors: list[str] = []
            projected_flags = self._append_memory_access_contract(operation, selectors)
            template = f"<{selectors[0]}>" if selectors else ""
            self._declare(
                operation.results[0],
                f"loom::view::load{template}({self.expression(view_id)}{suffix})",
            )
            return (
                "static_indices",
                *projected_flags,
            )
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
            selectors = []
            projected_flags = self._append_memory_access_contract(operation, selectors)
            template = f"<{selectors[0]}>" if selectors else ""
            self.writer.line(f"loom::view::store{template}({value}, {view}{suffix});")
            return (
                "static_indices",
                *projected_flags,
            )
        if name == "vector.load":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            origin = self._mixed_index_record(
                resolved,
                dynamic_field="indices",
                static_field="static_indices",
            )
            view = self.expression(resolved.value_id("view"))
            selectors = [self.cpp_type(result)]
            projected_flags = self._append_memory_access_contract(operation, selectors)
            value_type = self.module.values[result].type
            dimensions = (
                f", {self.dynamic_dimensions(value_type)}"
                if not value_type.is_all_static
                else ""
            )
            self._declare(
                result,
                "loom::vector::load<"
                + ", ".join(selectors)
                + f">({view}, {origin}{dimensions})",
            )
            return (
                "static_indices",
                *projected_flags,
            )
        if name == "vector.store":
            resolved = self._resolved(operation)
            origin = self._mixed_index_record(
                resolved,
                dynamic_field="indices",
                static_field="static_indices",
            )
            value = self.expression(resolved.value_id("value"))
            view = self.expression(resolved.value_id("view"))
            selectors = []
            projected_flags = self._append_memory_access_contract(operation, selectors)
            template = f"<{selectors[0]}>" if selectors else ""
            self.writer.line(
                f"loom::vector::store{template}({value}, {view}, {origin});"
            )
            return (
                "static_indices",
                *projected_flags,
            )
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
        if name in {
            "kernel.async.copy",
            "kernel.async.copy.mask",
            "kernel.async.gather",
            "kernel.async.gather.mask",
            "kernel.async.cluster.gather",
            "kernel.async.cluster.gather.mask",
        }:
            resolved = self._resolved(operation)
            scope = resolved.attr("cache_scope")
            temporal = resolved.attr("cache_temporal")
            is_copy = name.startswith("kernel.async.copy")
            is_cluster = name.startswith("kernel.async.cluster.")
            is_masked = name.endswith(".mask")
            if is_copy:
                function = "copy_mask" if is_masked else "copy"
                template_arguments = [
                    "loom::kernel::async::direction::"
                    + str(resolved.attr("direction")),
                    f"loom::cache::scope::{scope}",
                    f"loom::cache::temporal::{temporal}",
                ]
            else:
                function = "gather_mask" if is_masked else "gather"
                if is_cluster:
                    function = "cluster::" + function
                template_arguments = [
                    f"loom::cache::scope::{scope}",
                    f"loom::cache::temporal::{temporal}",
                ]
            arguments = [
                self.expression(resolved.value_id("source")),
                self.expression(resolved.value_id("dest")),
            ]
            if is_cluster:
                arguments.append(self.expression(resolved.value_id("cluster_mask")))
            if is_masked:
                arguments.append(self.expression(resolved.value_id("predicate")))
            expression = (
                f"loom::kernel::async::{function}<"
                + ", ".join(template_arguments)
                + ">("
                + ", ".join(arguments)
                + ")"
            )
            self._declare(resolved.value_id("token"), expression)
            projected = ["cache_scope", "cache_temporal"]
            if is_copy:
                projected.append("direction")
            return tuple(projected)
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
            elif self.parent_operations[id(operation)].name == "func.exit":
                self.writer.line("return;")
            return ()
        return None

    def _render_for(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        initial = resolved.value_ids("iter_args")
        results = resolved.value_ids("results")
        if len(initial) != len(results):
            raise CxxRenderError(operation, "loop-carried result arity is inconsistent")
        for result, source in zip(results, initial, strict=True):
            if result in self.suppressed_forwarding_values:
                continue
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
            if can_reuse_source and self._compatible_cxx_carrier(result, source):
                self.expressions[result] = source_expression
                self.expression_types[result] = self.expression_type(source)
            else:
                self._declare_loop_carrier(result, source)

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
            if result in self.suppressed_forwarding_values:
                continue
            self.expressions[block_arg] = self.expression(result)
            self.expression_types[block_arg] = self.expression_type(result)
        self.writer.open(
            f"for ({loop_type} {induction_name} = {lower}; "
            f"{induction_name} < {upper}; {increment})"
        )
        self._render_block(block, yield_targets=self._transport_targets(results))
        self.writer.close()

    def _declare_structured_result(self, result: int) -> None:
        """Declares the source value supplied by every continuing region edge."""

        value_type = self.module.values[result].type
        type_name = self._function_result_component_type(result)
        # Explicit initialization keeps ordinary numeric locals on the value
        # path. Resource handles begin their lifetime without a payload.
        initializer = ""
        if isinstance(value_type, ScalarType):
            initializer = "{0}" if repr(value_type) in {"index", "offset"} else "{}"
        elif (
            isinstance(value_type, ShapedType)
            and value_type.type_kind == TypeKind.VECTOR
            and value_type.rank == 1
            and repr(value_type.element_type) != "i1"
        ):
            initializer = "{}"
        name = self.names[result]
        self.expressions[result] = name
        self.expression_types[result] = type_name
        self.writer.line(f"{type_name} {name}{initializer};")

    def _assign_values(
        self, targets: Sequence[int | None], sources: Sequence[int]
    ) -> None:
        assignments = [
            (
                self.expression(target),
                self._buffer_argument(source, self._buffer_element(target))
                if isinstance(self.module.values[target].type, BufferType)
                else self.numeric_argument(source, self.expression_type(target)),
            )
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

    def _transport_targets(self, values: Sequence[int]) -> list[int | None]:
        return [
            None if value in self.suppressed_forwarding_values else value
            for value in values
        ]

    def _while_condition_expression(self, before: Block) -> str | None:
        """Projects a pure identity-forwarding header into one C++ expression."""

        if not before.ops or before.ops[-1].name != "scf.condition":
            return None
        condition = before.ops[-1]
        resolved = self._resolved(condition)
        if resolved.value_ids("forwarded") != before.arg_ids or any(
            operation.regions or not is_pure_operation(operation)
            for operation in before.ops[:-1]
        ):
            return None

        # Reuse the normal operation renderers so the condition accepts the
        # complete expression vocabulary without a second spelling table. A
        # header is directly projectable only when those renderers collapse it
        # to the condition assignment itself; a named intermediate or any
        # required statement retains the general guarded-loop projection.
        expressions = self.expressions.copy()
        expression_types = self.expression_types.copy()
        temporary_names = self.temporary_names.copy()
        line_count = len(self.writer.lines)
        try:
            self._render_block(
                before,
                condition_target="__loom_condition",
                condition_forward_targets=[None] * len(before.arg_ids),
            )
            condition_expression = _strip_outer_parentheses(
                self.expression(resolved.value_id("condition"))
            )
            emitted_line_count = len(self.writer.lines) - line_count
        except Exception:
            self.expressions = expressions
            self.expression_types = expression_types
            self.temporary_names = temporary_names
            raise
        finally:
            del self.writer.lines[line_count:]

        if emitted_line_count == 1:
            return condition_expression
        self.expressions = expressions
        self.expression_types = expression_types
        self.temporary_names = temporary_names
        return None

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

        shared_carriers = len(initial) == len(results) and all(
            (argument in self.suppressed_forwarding_values)
            == (result in self.suppressed_forwarding_values)
            and (
                result in self.suppressed_forwarding_values
                or self._compatible_cxx_carrier(argument, result)
            )
            for argument, result in zip(before.arg_ids, results, strict=True)
        )
        if shared_carriers:
            for argument, result, source in zip(
                before.arg_ids, results, initial, strict=True
            ):
                if result in self.suppressed_forwarding_values:
                    continue
                self._declare_loop_carrier(result, source)
                self.expressions[argument] = self.expression(result)
                self.expression_types[argument] = self.expression_type(result)
            header_targets = self._transport_targets(results)
        else:
            for argument, source in zip(before.arg_ids, initial, strict=True):
                if argument not in self.suppressed_forwarding_values:
                    self._declare_loop_carrier(argument, source)
            for result in results:
                if result not in self.suppressed_forwarding_values:
                    self._declare_structured_result(result)
            header_targets = self._transport_targets(before.arg_ids)

        for argument, result in zip(after.arg_ids, results, strict=True):
            if result in self.suppressed_forwarding_values:
                continue
            self.expressions[argument] = self.expression(result)
            self.expression_types[argument] = self.expression_type(result)

        condition = (
            self._while_condition_expression(before) if shared_carriers else None
        )
        if condition is not None:
            self.writer.open(f"while ({condition})")
            self._render_block(after, yield_targets=header_targets)
            self.writer.close()
            return

        continuation = self._temporary_name("loop_continues")
        self.writer.line(f"bool {continuation} = true;")
        self.writer.open("do")
        self._render_block(
            before,
            condition_target=continuation,
            condition_forward_targets=self._transport_targets(results),
        )
        self.writer.open(f"if ({continuation})")
        self._render_block(after, yield_targets=header_targets)
        self.writer.close()
        self.writer.close(f" while ({continuation});")

    def _declare_selection_results(self, operation: Operation) -> None:
        for result in operation.results:
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
                self._declare_structured_result(result)

    def _render_if(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        results = resolved.value_ids("results")
        self._declare_selection_results(operation)
        then_region = resolved.region("then_region")
        else_region = resolved.region("else_region")
        if then_region is None or len(then_region.blocks) != 1:
            raise CxxRenderError(operation, "scf.if then region must have one block")
        condition = _strip_outer_parentheses(
            self.expression(resolved.value_id("condition"))
        )
        yield_targets = self._transport_targets(results)
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

    def _render_switch(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        self._declare_selection_results(operation)
        selector = self.expression(resolved.value_id("selector"))
        yield_targets = self._transport_targets(operation.results)
        self.writer.open(f"switch (static_cast<long long>({selector}))")
        for key, region in zip(
            resolved.attr("case_keys"), resolved.regions("case_regions"), strict=True
        ):
            # Spell the signed minimum without a positive literal outside i64.
            literal = (
                "(-9223372036854775807ll - 1ll)" if key == -(1 << 63) else f"{key}ll"
            )
            self.writer.open(f"case {literal}:")
            self._render_block(region.blocks[0], yield_targets=yield_targets)
            self.writer.line("break;")
            self.writer.close()
        self.writer.open("default:")
        self._render_block(
            resolved.region("default_region").blocks[0], yield_targets=yield_targets
        )
        self.writer.line("break;")
        self.writer.close()
        self.writer.close()

    def _direct_terminator_assignments(
        self,
        block: Block,
        terminator_name: str,
        targets: Sequence[int | None] | None,
    ) -> dict[int, int]:
        if targets is None or not block.ops:
            return {}
        terminator = block.ops[-1]
        if terminator.name != terminator_name or len(terminator.operands) != len(
            targets
        ):
            return {}
        positions = {id(operation): index for index, operation in enumerate(block.ops)}
        assignments: dict[int, int] = {}
        for source, target in zip(terminator.operands, targets, strict=True):
            if target is None:
                continue
            definition = self.definitions.get(source)
            target_expression = self.expression(target)
            if (
                definition is None
                or isinstance(self.module.values[source].type, BufferType)
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
        direct_assignments = self._direct_terminator_assignments(
            block, "scf.yield", yield_targets
        )
        previous_assignments = self.assignment_targets
        self.assignment_targets = {**previous_assignments, **direct_assignments}
        try:
            for operation in block.ops:
                if operation.name == "config.get":
                    projected = self._render_config_get(operation)
                    self._validate_projected_attributes(operation, projected)
                    continue
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
                    self._render_function_return(operation)
                    continue
                if operation.name == "func.unreachable":
                    self._validate_projected_attributes(operation, ())
                    self.writer.line("__builtin_unreachable();")
                    continue
                if operation.name == "func.exit":
                    self._validate_projected_attributes(operation, ())
                    self._render_function_exit(operation)
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
                if operation.name == "scf.switch":
                    self._validate_projected_attributes(operation, ("case_keys",))
                    self._render_switch(operation)
                    continue
                if operation.name == "scf.while":
                    self._validate_projected_attributes(operation, ())
                    self._render_while(operation)
                    continue
                projected = self._render_pipeline_op(operation)
                if projected is not None:
                    self._validate_projected_attributes(operation, projected)
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
        self._validate_projected_attributes(kernel, ("callee", "retain", "target"))
        self._analyze_buffers(kernel)
        resolved = self._resolved(kernel)
        body = resolved.region("body")
        assert body is not None
        entry = body.blocks[0]
        symbol = str(resolved.attr("callee"))
        source_name = self.kernel_names[symbol]
        attributes = self._render_kernel_configuration(kernel)
        if source_name != symbol:
            attributes.insert(
                1,
                f"loom::symbol({cxx_string_literal(symbol.encode('utf-8'))})",
            )
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        if retain := self._retention_attribute(kernel):
            attributes.append(retain)
        target_symbol = resolved.attr("target")
        if target_symbol is not None:
            attributes.append(self._target_attribute(kernel, str(target_symbol)))
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
        self.kernel_buffer_elements[symbol] = tuple(buffer_elements)
        if parameters:
            self.writer.open(f"void {source_name}({', '.join(parameters)})")
        else:
            self.writer.open(f"void {source_name}()")
        self._render_block(entry)
        self.writer.close()

    def _plan_callable_names(
        self,
        kernels: list[Operation],
        functions: list[Operation],
        commands: list[Operation],
        pipelines: list[Operation],
    ) -> None:
        used_names: set[str] = set()
        for kernel in kernels:
            symbol = str(self._resolved(kernel).attr("callee"))
            base = clean_identifier(symbol, "kernel")
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.kernel_names[symbol] = name

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
            self.pipeline_callables[symbol] = pipeline

        for kernel in kernels:
            symbol = str(self._resolved(kernel).attr("callee"))
            base = self.kernel_names[symbol] + "_configuration"
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.kernel_configuration_names[symbol] = name

        target_symbols: list[str] = []

        def add_target(target: object) -> None:
            if isinstance(target, str) and target and target not in target_symbols:
                target_symbols.append(target)

        for callable_operation in (*kernels, *commands, *pipelines):
            resolved = self._resolved(callable_operation)
            add_target(resolved.attr("target"))
            for region in callable_operation.regions:
                for block in region.blocks:
                    for nested in walk_operations(block):
                        add_target(nested.attributes.get("target"))

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
            if self.targets[symbol].operation_name == "target.generic":
                type_base = f"{name}_type"
                type_name = type_base
                ordinal = 2
                while type_name in used_names:
                    type_name = f"{type_base}_{ordinal}"
                    ordinal += 1
                used_names.add(type_name)
                self.target_type_names[symbol] = type_name
        self.translation_unit_names = used_names

    def render(self) -> str:
        kernels = [
            operation
            for operation in self.module.body.ops
            if operation.name == "kernel.def"
        ]
        functions = [
            operation
            for operation in self.module.body.ops
            if operation.name in {"func.def", "func.decl"}
        ]
        function_definitions = [
            operation for operation in functions if operation.name == "func.def"
        ]
        commands = [
            operation
            for operation in self.module.body.ops
            if operation.name in {"command.def", "command.decl"}
        ]
        pipelines = [
            operation
            for operation in self.module.body.ops
            if operation.name in {"pipeline.def", "pipeline.decl"}
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
                "func.decl",
                "func.def",
                "command.def",
                "command.decl",
                "pipeline.def",
                "pipeline.decl",
                "config.decl",
                "config.def",
            } or is_target_operation(operation):
                continue
            raise CxxRenderError(
                operation,
                "a top-level definition would be omitted from the C++ projection",
            )
        self._plan_callable_names(kernels, functions, commands, pipelines)
        self._plan_kernel_configurations(kernels)
        self.suppressed_forwarding_values = plan_value_transport(
            self.module,
            self._resolved,
            self.parent_operations,
            self._source_value_operands,
        )
        self.buffer_access = BufferAccessPlan(self.module, self._resolved)
        self._plan_function_projections()
        self.view_access = plan_view_access(
            self.module, self._resolved, self._flat_source_parameters
        )
        self._plan_config_names()
        if self._render_configs():
            self.writer.line()
        for index, function in enumerate(functions):
            if index:
                self.writer.line()
            symbol = str(self._resolved(function).attr("callee"))
            for projection in self.buffer_access.projections[symbol]:
                self.buffer_projection = projection
                self._render_function_declaration(function)
        self.buffer_projection = None
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
                        "retain",
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
        pipeline_definitions = [
            operation for operation in pipelines if operation.name == "pipeline.def"
        ]
        callables = [
            *function_definitions,
            *kernels,
            *command_definitions,
            *pipeline_definitions,
        ]
        for index, callable_operation in enumerate(callables):
            if index:
                self.writer.line()
            if callable_operation.name == "func.def":
                symbol = str(self._resolved(callable_operation).attr("callee"))
                for projection in self.buffer_access.projections[symbol]:
                    self.buffer_projection = projection
                    self._render_function(callable_operation)
                self.buffer_projection = None
            elif callable_operation.name == "command.def":
                self._render_command_program(callable_operation)
            elif callable_operation.name == "pipeline.def":
                self._render_pipeline_definition(callable_operation)
            else:
                self._render_kernel(callable_operation)

        for operation in self.module.body.ops:
            if (
                operation.name in {"config.decl", "config.def"}
                and id(operation) not in self.projected_config_declarations
                and id(operation) not in self.rendered_config_operations
            ):
                raise CxxRenderError(
                    operation,
                    "configuration symbol was omitted from the C++ projection",
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
        uses_predicate_functions = self.uses_predicate_functions or any(
            isinstance(
                predicates := operation.attributes.get("predicates"), PredicateListAttr
            )
            and (
                operation.name in {"config.decl", "config.def"}
                or any(
                    any(
                        argument.tag == "value"
                        and isinstance(
                            self.module.values[argument.value].type, ScalarType
                        )
                        and repr(self.module.values[argument.value].type)
                        in {"index", "offset"}
                        for argument in predicate.args
                    )
                    for predicate in predicates
                )
            )
            for operation in self.module.body.ops
            if id(operation) not in self.projected_config_declarations
        )
        preamble = render_preamble(
            operation_names=operation_names,
            scalar_names=scalar_names,
            has_kernels=bool(kernels),
            uses_pipelines=bool(pipelines)
            or any(name.startswith("channel.") for name in operation_names)
            or any(
                isinstance(value.type, (ParameterizedType, PoolType))
                or (
                    isinstance(value.type, ShapedType)
                    and value.type.type_kind == TypeKind.TILE
                )
                for value in self.module.values
            ),
            uses_buffers=any(
                isinstance(value.type, BufferType) for value in self.module.values
            ),
            uses_views=any(
                isinstance(value.type, ShapedType)
                and value.type.type_kind == TypeKind.VIEW
                for value in self.module.values
            ),
            uses_encodings=any(
                isinstance(value.type, EncodingType) for value in self.module.values
            ),
            vector_aliases=self.types.vector_aliases,
            record_declarations=self.record_declarations,
            table_lookup_declarations=self.table_lookup_declarations,
            targets=tuple(
                (
                    target_name,
                    self.target_type_names.get(target_symbol),
                    self.targets[target_symbol],
                )
                for target_symbol, target_name in self.target_names.items()
            ),
            uses_scalar_functions=self.uses_scalar_functions,
            uses_numeric_conversion=self.uses_numeric_conversion,
            uses_predicate_functions=uses_predicate_functions,
        )
        return "\n".join((*preamble, *self.writer.lines)) + "\n"


def _select_callable_roots(module: Module, roots: Sequence[str]) -> Module:
    """Returns selected callables, their helper closure, and target records."""

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
            "func.decl",
            "func.def",
            "command.def",
            "command.decl",
            "pipeline.def",
            "pipeline.decl",
        }
        if symbol.op.name not in callable_operations or (
            require_definition
            and symbol.op.name in {"func.decl", "command.decl", "pipeline.decl"}
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
                    nested_target = operation.attributes.get("target")
                    if isinstance(nested_target, str):
                        target_names.add(nested_target)
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
                        "pipeline.compose",
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
            is_target_operation(operation)
            and operation.attributes.get("symbol") in target_names
        )
    ]
    return replace(module, body=replace(module.body, ops=selected_operations))


def render_module(module: Module, *, roots: Sequence[str] = ()) -> str:
    """Renders supported High IR callable roots as standalone Loom C++."""
    projected_module = _select_callable_roots(module, roots) if roots else module
    return CxxRenderer(projected_module).render()
