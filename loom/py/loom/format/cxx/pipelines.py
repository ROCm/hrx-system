# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Persistent-pipeline and channel projection into Loom C++."""

from __future__ import annotations

from loom.format.cxx.commands import CxxCommandRenderer
from loom.format.cxx.context import _memory_space
from loom.format.cxx.errors import CxxRenderError
from loom.ir import DynamicEncoding, Operation, ParameterizedType, ShapedType, TypeKind


class CxxPipelineRenderer(CxxCommandRenderer):
    """Renders pipeline definitions, strand regions, and channel ownership."""

    def _bind_expression(self, value_id: int, expression: str) -> None:
        self.expressions[value_id] = expression
        self.expression_types[value_id] = self.cpp_type(value_id)

    def _render_pipeline_definition(self, operation: Operation) -> None:
        scope = self.operation_scopes[id(operation)]
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        self._validate_projected_attributes(
            operation,
            (
                "callee",
                "scope",
                "visibility",
                "retain",
                "target",
                "predicates",
                "specialization_count",
            ),
        )
        self._analyze_buffers(operation)
        body = self._resolved(operation).region("body")
        if body is None or len(body.blocks) != 1:
            raise CxxRenderError(
                operation, "pipeline definition body must have one structured block"
            )
        self.writer.attributes(self._pipeline_attributes(operation))
        signature = self._pipeline_signature(operation, list(body.blocks[0].arg_ids))
        self.writer.open(signature)
        self._render_block(body.blocks[0])
        self.writer.close()

    def _render_pipeline_memory(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        coordinates = self._mixed_index_record(
            resolved,
            dynamic_field="coordinates",
            static_field="static_coordinates",
            force_typed=True,
        )
        space = _memory_space(resolved.attr("memory_space"))
        self._declare(
            resolved.value_id("result"),
            f"loom::pipeline::memory<{space}>({coordinates})",
        )
        return ("memory_space", "static_coordinates")

    def _render_pipeline_compose(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        callee = self.pipeline_names.get(symbol)
        pipeline = self.pipeline_callables.get(symbol)
        if callee is None or pipeline is None:
            raise CxxRenderError(
                operation, f"composed pipeline {symbol!r} is not present"
            )
        arguments = self._pipeline_source_arguments(
            operation,
            pipeline,
            resolved.value_ids("specializations"),
            resolved.value_ids("bindings"),
        )
        self.writer.line(f"loom::pipeline::compose<{callee}>({', '.join(arguments)});")
        return ("callee",)

    def _render_pipeline_strand(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        attributes = [
            "loom::strand("
            + ", ".join(
                self._mixed_index_record(
                    resolved,
                    dynamic_field=dynamic,
                    static_field=static,
                    force_typed=True,
                )
                for dynamic, static in (
                    ("origins", "static_origins"),
                    ("counts", "static_counts"),
                    ("strides", "static_strides"),
                )
            )
            + ")"
        ]
        target = resolved.attr("target")
        if target is not None:
            attributes.append(self._target_attribute(operation, str(target)))
        body = resolved.region("body")
        if body is None or len(body.blocks) != 1 or body.blocks[0].arg_ids:
            raise CxxRenderError(
                operation, "pipeline strand must have one argument-free body block"
            )
        self.writer.open("[[" + ", ".join(attributes) + "]]")
        self._render_block(body.blocks[0])
        self.writer.close()
        return (
            "target",
            "static_origins",
            "static_counts",
            "static_strides",
        )

    def _channel_payload(self, operation: Operation, value_id: int) -> ShapedType:
        value_type = self.module.values[value_id].type
        if not isinstance(value_type, ParameterizedType) or (
            value_type.family_name not in {"channel", "read", "write"}
        ):
            raise CxxRenderError(operation, "channel operation has a non-channel value")
        payload = value_type.get("payload")
        if not isinstance(payload, ShapedType) or payload.type_kind != TypeKind.TILE:
            raise CxxRenderError(operation, "channel payload is not a tile type")
        return payload

    def _channel_mode_template(self, operation: Operation, mode: object) -> str:
        if mode is None:
            return ""
        if mode != "mutable":
            raise CxxRenderError(
                operation, f"channel mode {mode!r} has no C++ spelling"
            )
        return "<loom::channel::mode::mutable_>"

    def _render_channel_bind(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        result = resolved.value_id("result")
        discipline = resolved.attr("discipline")
        if discipline != "fifo":
            raise CxxRenderError(
                operation, f"channel discipline {discipline!r} has no C++ spelling"
            )
        arguments = [
            self.expression(resolved.value_id("storage")),
            self.expression(resolved.value_id("capacity")),
        ]
        payload = self._channel_payload(operation, result)
        if isinstance(payload.encoding, DynamicEncoding):
            arguments.append(self.expression(payload.encoding.value_id))
        expression = (
            "loom::channel::bind<loom::channel::discipline::fifo>("
            + ", ".join(arguments)
            + ")"
        )
        self._declare(result, expression)
        return ("discipline",)

    def _render_channel_pair(
        self,
        operation: Operation,
        *,
        function: str,
        operand: int,
        fields: tuple[str, str],
        template: str = "",
        stem: str,
    ) -> None:
        name = self._temporary_name(stem)
        self.writer.line(
            f"auto {name} = loom::channel::{function}{template}("
            f"{self.expression(operand)});"
        )
        for result, field in zip(operation.results, fields, strict=True):
            self._bind_expression(result, f"{name}.{field}")

    def _render_channel_op(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name == "channel.bind":
            return self._render_channel_bind(operation)
        resolved = self._resolved(operation)
        if name == "channel.reserve":
            self._render_channel_pair(
                operation,
                function="reserve",
                operand=resolved.value_id("channel"),
                fields=("write", "payload"),
                stem="reservation",
            )
            return ()
        if name == "channel.accept":
            template = self._channel_mode_template(operation, resolved.attr("mode"))
            self._declare(
                resolved.value_id("read"),
                f"loom::channel::accept{template}("
                f"{self.expression(resolved.value_id('channel'))})",
            )
            return ("mode",)
        if name == "channel.wait":
            self._declare(
                resolved.value_id("view"),
                f"loom::channel::wait({self.expression(resolved.value_id('read'))})",
            )
            return ()
        if name == "channel.acquire":
            template = self._channel_mode_template(operation, resolved.attr("mode"))
            self._render_channel_pair(
                operation,
                function="acquire",
                operand=resolved.value_id("channel"),
                fields=("read", "payload"),
                template=template,
                stem="acquisition",
            )
            return ("mode",)
        if name == "channel.publish":
            self.writer.line(
                "loom::channel::publish("
                f"{self.expression(resolved.value_id('write'))});"
            )
            return ()
        if name == "channel.release":
            self.writer.line(
                f"loom::channel::release({self.expression(resolved.value_id('read'))});"
            )
            return ()
        if name == "channel.fanout":
            results = resolved.value_ids("reads")
            result_name = self._temporary_name("fanout")
            self.writer.line(
                f"auto {result_name} = loom::channel::fanout<{len(results)}>("
                f"{self.expression(resolved.value_id('read'))});"
            )
            for ordinal, result in enumerate(results):
                remaining = ".remaining" * min(ordinal, len(results) - 2)
                field = "second" if ordinal == len(results) - 1 else "first"
                self._bind_expression(result, f"{result_name}{remaining}.{field}")
            return ()
        if name == "channel.copy":
            self.writer.line(
                "loom::channel::copy("
                f"{self.expression(resolved.value_id('source'))}, "
                f"{self.expression(resolved.value_id('destination'))});"
            )
            return ()
        if name == "channel.select":
            self._render_channel_pair(
                operation,
                function="select",
                operand=resolved.value_id("channel"),
                fields=("source_rank", "read"),
                stem="selection",
            )
            return ()
        return None

    def _render_pipeline_op(self, operation: Operation) -> tuple[str, ...] | None:
        if operation.name in {"pipeline.finish", "pipeline.end", "pipeline.yield"}:
            return ()
        if operation.name in {"pipeline.serial", "pipeline.concurrent"}:
            self._render_schedule_region(operation)
            return ()
        if operation.name == "pipeline.memory":
            return self._render_pipeline_memory(operation)
        if operation.name == "pipeline.compose":
            return self._render_pipeline_compose(operation)
        if operation.name == "pipeline.strand":
            return self._render_pipeline_strand(operation)
        return self._render_channel_op(operation)
