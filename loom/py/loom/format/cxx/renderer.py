# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Readable Loom High IR projection into the Loom C++ authoring surface."""

from __future__ import annotations

import re

from loom.format.cxx.context import (
    _memory_space,
    _strip_outer_parentheses,
    walk_operations,
)
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.expressions import CxxExpressionRenderer
from loom.format.cxx.naming import clean_identifier, loop_carrier_stem
from loom.format.cxx.translation_unit import render_preamble
from loom.format.cxx.types import KERNEL_COORDINATE_CPP_TYPE
from loom.ir import (
    Block,
    DynamicEncoding,
    Module,
    Operation,
    ScalarType,
    ShapedType,
    StaticDim,
    TypeKind,
)


class CxxRenderer(CxxExpressionRenderer):
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
            if not result_type.is_all_static:
                raise CxxRenderError(
                    operation,
                    "dynamic view requires explicit runtime-dimension projection",
                )
            if isinstance(result_type.encoding, DynamicEncoding):
                encoding_id = result_type.encoding.value_id
                definition = self.definitions.get(encoding_id)
                if definition is None:
                    raise CxxRenderError(
                        operation,
                        "view encoding value is not defined in the projected scope",
                    )
                if definition.name == "encoding.layout.dense":
                    encoding = f"loom::encoding::layout::dense<{result_type.rank}>()"
                else:
                    encoding = self.expression(encoding_id)
            elif result_type.encoding is None:
                encoding = f"loom::encoding::layout::dense<{result_type.rank}>()"
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
            pointer = self._buffer_pointer_expression(
                buffer_id, offset_id, element, is_const
            )
            extents = ", ".join(
                str(dimension.size)
                for dimension in result_type.dims
                if isinstance(dimension, StaticDim)
            )
            expression = f"loom::buffer::view<{extents}>({pointer}, {{}}, {encoding})"
            self._declare(result, expression)
            return ()
        if name == "view.subview":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            result_type = self.module.values[result].type
            if not isinstance(result_type, ShapedType) or not result_type.is_all_static:
                raise CxxRenderError(
                    operation, "dependent subview result is not yet projected"
                )
            extents = ", ".join(
                str(dimension.size)
                for dimension in result_type.dims
                if isinstance(dimension, StaticDim)
            )
            origin = self._mixed_index_record(
                resolved,
                dynamic_field="offsets",
                static_field="static_offsets",
            )
            source = self.expression(resolved.value_id("source"))
            expression = f"loom::view::subview<{extents}>({source}, {origin}, {{}})"
            self._declare(result, expression)
            return ("static_offsets",)
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
                declaration_type="unsigned",
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
        if (
            step_constant is not None
            and 0 < step_constant <= (1 << 32) - 1
            and (upper_constant is None or upper_constant <= (1 << 32) - step_constant)
        ):
            counted_bounds = [
                self._unsigned_counted_bound(value_id)
                for value_id in (lower_id, upper_id, step_id)
            ]
        if all(bound is not None for bound in counted_bounds):
            loop_type = "unsigned"
            lower, upper, step = (str(bound) for bound in counted_bounds)
        else:
            loop_type = self.cpp_type(induction)
            lower = self._address_operand(lower_id)
            upper = self._address_operand(upper_id)
            step = self._address_operand(step_id)
        if attributes and loop_type != "unsigned":
            raise CxxRenderError(
                operation,
                "scheduled loop has no recoverable nonwrapping unsigned C++ "
                "induction contract",
            )
        increment = (
            f"++{induction_name}"
            if loop_type == "unsigned" and step in {"1", "1u", "1ull"}
            else (
                f"{induction_name} += {step}"
                if loop_type == "unsigned"
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
            if repr(self.module.values[source].type) == result_type:
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

    def _assign_values(self, targets: list[int], sources: list[int]) -> None:
        assignments = [
            (self.expression(target), self.expression(source))
            for target, source in zip(targets, sources, strict=True)
            if self.expression(target) != self.expression(source)
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

        shared_carriers = len(initial) == len(results) and all(
            repr(self.module.values[argument].type)
            == repr(self.module.values[result].type)
            for argument, result in zip(before.arg_ids, results, strict=True)
        )
        if shared_carriers:
            for argument, result, source in zip(
                before.arg_ids, results, initial, strict=True
            ):
                self._declare(result, self._loop_initializer(source))
                self.expressions[argument] = self.expression(result)
                self.expression_types[argument] = self.expression_type(result)
            header_targets = results
        else:
            for argument, source in zip(before.arg_ids, initial, strict=True):
                self._declare(argument, self._loop_initializer(source))
            for result in results:
                self._declare(result, self._while_result_initializer(result, initial))
            header_targets = before.arg_ids

        for argument, result in zip(after.arg_ids, results, strict=True):
            self.expressions[argument] = self.expression(result)
            self.expression_types[argument] = self.expression_type(result)

        continuation = self._temporary_name("loop_continues")
        self.writer.line(f"bool {continuation} = true;")
        self.writer.open(f"while ({continuation})")
        self._render_block(
            before,
            condition_target=continuation,
            condition_forward_targets=results,
        )
        self.writer.open(f"if ({continuation})")
        self._render_block(after, yield_targets=header_targets)
        self.writer.close()
        self.writer.close()

    def _render_if(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        results = resolved.value_ids("results")
        for result in results:
            assignment_target = self.assignment_targets.get(result)
            if assignment_target is not None:
                # A surrounding structured operation already owns the mutable
                # carrier. Each branch assigns or preserves that value; a
                # default assignment here would destroy its incoming state.
                self.expressions[result] = assignment_target
                self.expression_types[result] = self.cpp_type(result)
            else:
                self._declare(result, f"{self.cpp_type(result)}{{}}")
        then_region = resolved.region("then_region")
        else_region = resolved.region("else_region")
        if then_region is None or len(then_region.blocks) != 1:
            raise CxxRenderError(operation, "scf.if then region must have one block")
        condition = _strip_outer_parentheses(
            self.expression(resolved.value_id("condition"))
        )
        self.writer.open(f"if ({condition})")
        self._render_block(then_region.blocks[0], yield_targets=results)
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
                self.expression(source) == self.expression(result)
                for source, result in zip(
                    yield_operation.operands, results, strict=True
                )
            ):
                self.writer.close()
                return
        self.writer.continue_block("else")
        self._render_block(else_block, yield_targets=results)
        self.writer.close()

    def _direct_yield_assignments(
        self, block: Block, yield_targets: list[int] | None
    ) -> dict[int, str]:
        if yield_targets is None or not block.ops:
            return {}
        yield_operation = block.ops[-1]
        if yield_operation.name != "scf.yield" or len(yield_operation.operands) != len(
            yield_targets
        ):
            return {}
        positions = {id(operation): index for index, operation in enumerate(block.ops)}
        assignments: dict[int, str] = {}
        for source, target in zip(yield_operation.operands, yield_targets, strict=True):
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
            assignments[source] = target_expression
        return assignments

    def _render_block(
        self,
        block: Block,
        *,
        yield_targets: list[int] | None = None,
        condition_target: str | None = None,
        condition_forward_targets: list[int] | None = None,
    ) -> None:
        direct_assignments = self._direct_yield_assignments(block, yield_targets)
        previous_assignments = self.assignment_targets
        self.assignment_targets = {**previous_assignments, **direct_assignments}
        try:
            for operation in block.ops:
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

    def _launch_values(self, kernel: Operation) -> tuple[list[str], list[str]]:
        resolved = self._resolved(kernel)
        config = resolved.region("config")
        if config is None or len(config.blocks) != 1:
            raise CxxRenderError(
                kernel, "kernel launch configuration must be one block"
            )
        local: dict[int, str] = {}
        launch: Operation | None = None
        for operation in config.blocks[0].ops:
            if operation.name == "index.constant":
                self._validate_projected_attributes(operation, ("value",))
                result = operation.results[0]
                local[result] = str(operation.attributes["value"])
            elif operation.name == "kernel.launch.config":
                self._validate_projected_attributes(operation, ())
                launch = operation
            else:
                raise CxxRenderError(
                    operation, "launch configuration uses a non-constant expression"
                )
        if launch is None:
            raise CxxRenderError(kernel, "kernel has no launch configuration")
        if len(launch.operands) != 6:
            raise CxxRenderError(
                launch,
                "clustered launch geometry requires a configuration-function "
                "projection",
            )
        fields = self._resolved(launch)
        try:
            counts = [
                local[fields.value_id(f"workgroup_count_{axis}")] for axis in "xyz"
            ]
            sizes = [local[fields.value_id(f"workgroup_size_{axis}")] for axis in "xyz"]
        except KeyError as error:
            raise CxxRenderError(
                launch,
                "dynamic launch geometry requires a configuration-function projection",
            ) from error
        return counts, sizes

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
        self.rendered_assumptions.clear()
        self._validate_projected_attributes(kernel, ("callee", "target"))
        self._analyze_buffers(kernel)
        resolved = self._resolved(kernel)
        body = resolved.region("body")
        assert body is not None
        entry = body.blocks[0]
        counts, sizes = self._launch_values(kernel)
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
        attributes = ["loom::kernel"]
        if target_info is not None:
            attributes.append(f"loom::target({self.target_names[str(target_symbol)]})")
        attributes.append(f"loom::workgroup_count({', '.join(counts)})")
        attributes.append(f"loom::workgroup_size({', '.join(sizes)})")
        self.writer.attributes(attributes)

        parameters: list[str] = []
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
            self.expressions[argument] = name
            self.expression_types[argument] = type_name
        if parameters:
            self.writer.open(f"void {resolved.attr('callee')}({', '.join(parameters)})")
        else:
            self.writer.open(f"void {resolved.attr('callee')}()")
        self._render_block(entry)
        self.writer.close()

    def _plan_target_names(self, kernels: list[Operation]) -> None:
        target_symbols: list[str] = []
        function_names: set[str] = set()
        for kernel in kernels:
            resolved = self._resolved(kernel)
            function_names.add(str(resolved.attr("callee")))
            target_symbol = resolved.attr("target")
            if target_symbol is not None:
                symbol = str(target_symbol)
                if symbol and symbol not in target_symbols:
                    target_symbols.append(symbol)

        used_names = set(function_names)
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
        if not kernels:
            raise CxxRenderError(
                None,
                "module contains no kernel definitions",
            )
        for operation in self.module.body.ops:
            if operation.name == "kernel.def" or operation.name.endswith(".target"):
                continue
            raise CxxRenderError(
                operation,
                "a top-level definition would be omitted from the C++ projection",
            )
        self._plan_target_names(kernels)
        for index, kernel in enumerate(kernels):
            if index:
                self.writer.line()
            self._render_kernel(kernel)

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
        preamble = render_preamble(
            operation_names=operation_names,
            scalar_names=scalar_names,
            vector_aliases=self.types.vector_aliases,
            record_declarations=self.record_declarations,
            table_lookup_declarations=self.table_lookup_declarations,
            targets=tuple(
                (target_name, self.targets[target_symbol])
                for target_symbol, target_name in self.target_names.items()
            ),
            uses_numeric_conversion=self.uses_numeric_conversion,
        )
        return "\n".join((*preamble, *self.writer.lines)) + "\n"


def render_module(module: Module) -> str:
    """Renders a supported High IR kernel module as standalone Loom C++."""
    return CxxRenderer(module).render()
