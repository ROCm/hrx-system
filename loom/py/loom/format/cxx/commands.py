# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-program projection into the Loom C++ authoring surface."""

from __future__ import annotations

from dataclasses import dataclass

from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.expressions import CxxExpressionRenderer
from loom.ir import (
    BufferType,
    DynamicDim,
    DynamicEncoding,
    Operation,
    ParameterizedType,
    PredicateListAttr,
    ShapedType,
    TypeKind,
    Utf8Attr,
)

_SOURCE_OPERAND_PROJECTIONS = {
    "func.call": "_ordinary_source_operands",
    "func.return": "_ordinary_source_operands",
    "command.launch": "_command_source_operands",
    "pipeline.compose": "_pipeline_source_operands",
    "kernel.launch": "_kernel_source_operands",
}


@dataclass(frozen=True, slots=True)
class SourceParameter:
    """One reconstructed C++ parameter and its flattened type dependencies."""

    # Payload identity represented by one complete C++ value.
    value_id: int
    # Ordered dimensions and layout preceding this payload in the flat ABI.
    dependencies: tuple[int, ...]
    # Whether the value belongs to a staged callable's launch bindings.
    binding: bool


def _quote_string(value: str | Utf8Attr) -> str:
    text = value.value if isinstance(value, Utf8Attr) else value
    escaped: list[str] = []
    for character in text:
        if character in {'"', "\\"}:
            escaped.append("\\" + character)
        elif character in {"\n", "\r", "\t"}:
            escaped.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[character])
        elif ord(character) < 32 or ord(character) == 127:
            escaped.append(f"\\{ord(character):03o}")
        else:
            escaped.append(character)
    return '"' + "".join(escaped) + '"'


class CxxCommandRenderer(CxxExpressionRenderer):
    """Renders command definitions and operations from verified High IR."""

    def _retention_attribute(self, operation: Operation) -> str | None:
        retain = self._resolved(operation).attr("retain")
        if retain is None:
            return None
        if retain != "retain":
            raise CxxRenderError(
                operation, f"symbol retention policy {retain!r} has no C++ spelling"
            )
        return "loom::retain"

    def _source_type_dependencies(self, value_id: int) -> tuple[int, ...]:
        """Returns flattened identities carried inside one source value."""

        value_type = self.module.values[value_id].type
        while isinstance(value_type, ParameterizedType):
            if value_type.family_name not in {"channel", "read", "write"}:
                return ()
            value_type = value_type.get("payload")
        if not isinstance(value_type, ShapedType) or value_type.type_kind not in {
            TypeKind.TILE,
            TypeKind.VIEW,
        }:
            return ()
        dependencies: list[int] = []
        for dimension in value_type.dims:
            if isinstance(dimension, DynamicDim):
                if dimension.value_id is None:
                    raise CxxRenderError(
                        None,
                        "source view parameter has an unbound dynamic dimension",
                    )
                dependencies.append(dimension.value_id)
        if isinstance(value_type.encoding, DynamicEncoding):
            dependencies.append(value_type.encoding.value_id)
        elif value_type.encoding is not None:
            raise CxxRenderError(
                None,
                "statically encoded source view has no C++ signature projection",
            )
        return tuple(dependencies)

    def _source_parameters(
        self,
        operation: Operation,
        arguments: list[int],
        *,
        specialization_count: int | None = None,
        allow_nonstorage_bindings: bool = False,
    ) -> tuple[list[SourceParameter], list[SourceParameter]]:
        if specialization_count is None:
            specialization_count = operation.attributes.get("specialization_count")
        if (
            type(specialization_count) is not int
            or specialization_count < 0
            or specialization_count > len(arguments)
        ):
            raise CxxRenderError(
                operation, "staged signature has an invalid specialization count"
            )
        key = (tuple(arguments), specialization_count, allow_nonstorage_bindings)
        cached = self.staged_source_parameter_cache.get(key)
        if cached is not None:
            return cached
        specializations = arguments[:specialization_count]
        bindings = arguments[specialization_count:]
        binding_parameters: list[SourceParameter] = []
        binding_dependencies: list[int] = []
        for binding in bindings:
            value_type = self.module.values[binding].type
            is_storage = isinstance(value_type, BufferType) or (
                isinstance(value_type, ShapedType)
                and value_type.type_kind == TypeKind.VIEW
            )
            if not is_storage and not allow_nonstorage_bindings:
                raise CxxRenderError(
                    operation,
                    "launch binding is neither a buffer nor a typed view",
                )
            dependencies = self._source_type_dependencies(binding)
            binding_dependencies.extend(dependencies)
            binding_parameters.append(
                SourceParameter(binding, dependencies, binding=True)
            )

        # Binding metadata occupies an ordered suffix of the specialization
        # prefix. Reusing an actual View repeats its metadata at each position;
        # it does not make those positions one source parameter.
        boundary = len(specializations) - len(binding_dependencies)
        if boundary < 0 or specializations[boundary:] != binding_dependencies:
            raise CxxRenderError(
                operation,
                "staged signature dependencies do not form a C++ source-order "
                "projection",
            )
        specialization_parameters = self._flat_source_parameters(
            operation, specializations[:boundary], signature="staged signature"
        )
        result = specialization_parameters, binding_parameters
        self.staged_source_parameter_cache[key] = result
        return result

    def _flat_source_parameters(
        self,
        operation: Operation,
        values: list[int],
        *,
        signature: str = "flattened signature",
    ) -> list[SourceParameter]:
        """Reconstructs source values from complete positional component groups."""

        key = tuple(values)
        cached = self.flat_source_parameter_cache.get(key)
        if cached is not None:
            return cached
        parameters: list[SourceParameter] = []
        end = len(values)
        while end:
            value = values[end - 1]
            dependencies = self._source_type_dependencies(value)
            start = end - 1 - len(dependencies)
            if start < 0 or tuple(values[start : end - 1]) != dependencies:
                raise CxxRenderError(
                    operation,
                    f"{signature} dependencies do not form a C++ source-order "
                    "projection",
                )
            parameters.append(SourceParameter(value, dependencies, binding=False))
            end = start
        parameters.reverse()
        self.flat_source_parameter_cache[key] = parameters
        return parameters

    def _hide_source_dependencies(self, parameter: SourceParameter) -> None:
        """Records metadata transported inside a complete source binding."""

        for dependency in parameter.dependencies:
            self.hidden_source_dependencies[dependency] = parameter.value_id

    def _staged_parameter_type(self, operation: Operation, value_id: int) -> str:
        value_type = self.module.values[value_id].type
        if isinstance(value_type, BufferType):
            return "loom::type::buffer<unsigned char>"
        if isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VIEW:
            return self.cpp_type(value_id)
        type_name = self.cpp_type(value_id)
        if type_name == "auto":
            raise CxxRenderError(
                operation,
                f"staged parameter type {value_type!r} has no explicit C++ spelling",
            )
        return type_name

    def _view_dependency_expressions(
        self, operation: Operation, value_type: ShapedType
    ) -> list[str]:
        dependencies = [
            self.expression(dimension.value_id)
            for dimension in value_type.dims
            if isinstance(dimension, DynamicDim) and dimension.value_id is not None
        ]
        if isinstance(value_type.encoding, DynamicEncoding):
            dependencies.append(
                self.dynamic_view_encoding_expression(operation, value_type)
            )
        return dependencies

    def _bind_staged_parameters(
        self,
        operation: Operation,
        arguments: list[int],
        *,
        allow_nonstorage_bindings: bool = False,
    ) -> tuple[list[str], list[SourceParameter], list[SourceParameter]]:
        specializations, bindings = self._source_parameters(
            operation,
            arguments,
            allow_nonstorage_bindings=allow_nonstorage_bindings,
        )
        parameters: list[str] = []
        for ordinal, parameter in enumerate((*specializations, *bindings)):
            self._hide_source_dependencies(parameter)
            name = self.names[parameter.value_id]
            type_name = self._staged_parameter_type(operation, parameter.value_id)
            attribute = (
                "[[loom::launch]] "
                if bindings and ordinal == len(specializations)
                else ""
            )
            parameters.append(f"{attribute}{type_name} {name}")
            self.expressions[parameter.value_id] = name
            self.expression_types[parameter.value_id] = type_name
        return parameters, specializations, bindings

    def _command_attributes(self, operation: Operation) -> list[str]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        attributes = ["loom::command_program", f"loom::symbol({_quote_string(symbol)})"]
        if retain := self._retention_attribute(operation):
            attributes.append(retain)
        target_symbol = resolved.attr("target")
        if target_symbol is not None:
            attributes.append(self._target_attribute(operation, str(target_symbol)))
        return attributes

    def _command_signature(self, operation: Operation, arguments: list[int]) -> str:
        resolved = self._resolved(operation)
        parameters, _, _ = self._bind_staged_parameters(operation, arguments)
        symbol = str(resolved.attr("callee"))
        visibility = resolved.attr("visibility")
        storage = "" if visibility == "public" else "static "
        signature = (
            f"{storage}void {self.command_names[symbol]}(" + ", ".join(parameters) + ")"
        )
        predicates = resolved.attr("predicates")
        if predicates is not None:
            if not isinstance(predicates, PredicateListAttr):
                raise CxxRenderError(
                    operation, "command predicates are not predicate-list metadata"
                )
            conditions = [
                condition
                for predicate in predicates
                if (condition := self._predicate(predicate)) is not None
            ]
            if conditions:
                signature += " [[loom::where(" + " && ".join(conditions) + ")]]"
        return signature

    def _render_command_declaration(self, operation: Operation) -> None:
        self._validate_projected_attributes(
            operation,
            (
                "callee",
                "visibility",
                "retain",
                "target",
                "predicates",
                "specialization_count",
            ),
        )
        self.writer.attributes(self._command_attributes(operation))
        self.writer.line(
            self._command_signature(operation, list(operation.operands)) + ";"
        )

    def _pipeline_attributes(self, operation: Operation) -> list[str]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        attributes = [
            'loom::pipeline("kernel")'
            if resolved.attr("scope") == "kernel"
            else "loom::pipeline",
            f"loom::symbol({_quote_string(symbol)})",
        ]
        if retain := self._retention_attribute(operation):
            attributes.append(retain)
        target_symbol = resolved.attr("target")
        if target_symbol is not None:
            attributes.append(self._target_attribute(operation, str(target_symbol)))
        return attributes

    def _pipeline_arguments(self, operation: Operation) -> list[int]:
        if operation.name == "pipeline.decl":
            return list(operation.operands)
        body = self._resolved(operation).region("body")
        if body is None or len(body.blocks) != 1:
            raise CxxRenderError(
                operation, "pipeline definition body must have one structured block"
            )
        return list(body.blocks[0].arg_ids)

    def _pipeline_signature(self, operation: Operation, arguments: list[int]) -> str:
        parameters, _, _ = self._bind_staged_parameters(
            operation, arguments, allow_nonstorage_bindings=True
        )
        symbol = str(self._resolved(operation).attr("callee"))
        visibility = self._resolved(operation).attr("visibility")
        storage = (
            "static "
            if operation.name == "pipeline.def" and visibility != "public"
            else ""
        )
        signature = f"{storage}void {self.pipeline_names[symbol]}("
        signature += ", ".join(parameters) + ")"
        predicates = self._resolved(operation).attr("predicates")
        if predicates is not None:
            if not isinstance(predicates, PredicateListAttr):
                raise CxxRenderError(
                    operation, "pipeline predicates are not predicate-list metadata"
                )
            conditions = [
                condition
                for predicate in predicates
                if (condition := self._predicate(predicate)) is not None
            ]
            if conditions:
                signature += " [[loom::where(" + " && ".join(conditions) + ")]]"
        return signature

    def _render_pipeline_declaration(self, operation: Operation) -> None:
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
        self.writer.attributes(self._pipeline_attributes(operation))
        self.writer.line(
            self._pipeline_signature(operation, self._pipeline_arguments(operation))
            + ";"
        )

    def _pipeline_source_bindings(
        self,
        operation: Operation,
        pipeline: Operation,
        specializations: list[int],
        bindings: list[int],
    ) -> list[tuple[int, int]]:
        """Retains formal/actual payload pairs in the source parameter order."""

        cached = self.pipeline_argument_bindings.get(id(operation))
        if cached is not None:
            return cached
        source_specializations, source_bindings = self._source_parameters(
            pipeline,
            self._pipeline_arguments(pipeline),
            allow_nonstorage_bindings=True,
        )
        if len(bindings) != len(source_bindings):
            raise CxxRenderError(
                operation, "pipeline call binding arity is inconsistent"
            )
        arguments: list[tuple[int, int]] = []
        specialization_ordinal = 0
        for parameter in source_specializations:
            specialization_ordinal += len(parameter.dependencies) + 1
            if specialization_ordinal > len(specializations):
                raise CxxRenderError(
                    operation, "pipeline call specialization arity is inconsistent"
                )
            arguments.append(
                (parameter.value_id, specializations[specialization_ordinal - 1])
            )
        for parameter, binding in zip(source_bindings, bindings, strict=True):
            specialization_ordinal += len(parameter.dependencies)
            if specialization_ordinal > len(specializations):
                raise CxxRenderError(
                    operation, "pipeline call specialization arity is inconsistent"
                )
            arguments.append((parameter.value_id, binding))
        if specialization_ordinal != len(specializations):
            raise CxxRenderError(
                operation, "pipeline call specialization arity is inconsistent"
            )
        self.pipeline_argument_bindings[id(operation)] = arguments
        return arguments

    def _pipeline_source_arguments(
        self,
        operation: Operation,
        pipeline: Operation,
        specializations: list[int],
        bindings: list[int],
    ) -> list[str]:
        return [
            self._staged_argument(parameter, argument)
            for parameter, argument in self._pipeline_source_bindings(
                operation, pipeline, specializations, bindings
            )
        ]

    def _ordinary_source_operands(self, operation: Operation) -> list[int]:
        return [
            parameter.value_id
            for parameter in self._flat_source_parameters(
                operation, list(operation.operands)
            )
        ]

    def _command_source_operands(self, operation: Operation) -> list[int]:
        resolved = self._resolved(operation)
        specializations = resolved.value_ids("specializations")
        bindings = resolved.value_ids("bindings")
        parameters, binding_parameters = self._source_parameters(
            operation,
            [*specializations, *bindings],
            specialization_count=len(specializations),
        )
        return [parameter.value_id for parameter in (*parameters, *binding_parameters)]

    def _pipeline_source_operands(self, operation: Operation) -> list[int]:
        resolved = self._resolved(operation)
        pipeline = self.pipeline_callables.get(str(resolved.attr("callee")))
        if pipeline is None:
            raise CxxRenderError(operation, "composed pipeline is not present")
        return [
            argument
            for _, argument in self._pipeline_source_bindings(
                operation,
                pipeline,
                resolved.value_ids("specializations"),
                resolved.value_ids("bindings"),
            )
        ]

    def _kernel_source_operands(self, operation: Operation) -> list[int]:
        resolved = self._resolved(operation)
        pipeline = self.pipeline_callables.get(str(resolved.attr("callee")))
        workloads = resolved.value_ids("workloads")
        arguments = resolved.value_ids("arguments")
        if pipeline is not None:
            return [
                argument
                for _, argument in self._pipeline_source_bindings(
                    operation, pipeline, workloads, arguments
                )
            ]
        return self._kernel_workload_operands(operation) + arguments

    def _kernel_workload_operands(self, operation: Operation) -> list[int]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        component_counts = self.kernel_workload_component_counts.get(symbol)
        if component_counts is None:
            raise CxxRenderError(
                operation, f"launched kernel {symbol!r} is not present"
            )
        workloads = resolved.value_ids("workloads")
        if sum(component_counts) != len(workloads):
            raise CxxRenderError(
                operation, "kernel launch workload arity is inconsistent"
            )
        arguments: list[int] = []
        component_ordinal = 0
        for component_count in component_counts:
            component_ordinal += component_count
            arguments.append(workloads[component_ordinal - 1])
        return arguments

    def _source_value_operands(self, operation: Operation) -> list[int]:
        """Returns expressions used by rendering, excluding bundled metadata."""

        project = _SOURCE_OPERAND_PROJECTIONS.get(operation.name)
        return (
            getattr(self, project)(operation)
            if project is not None
            else list(operation.operands)
        )

    def _staged_argument(self, parameter: int, argument: int) -> str:
        """Preserves the byte-buffer contract of staged source parameters."""

        expression = self.expression(argument)
        if not (
            isinstance(self.module.values[parameter].type, BufferType)
            and isinstance(self.module.values[argument].type, BufferType)
        ):
            return expression
        source_type = self.expression_type(argument)
        if source_type.endswith("*"):
            expression = f"loom::buffer::root({expression})"
            source_type = f"loom::type::buffer<{source_type[:-1].strip()}>"
        if source_type != "loom::type::buffer<unsigned char>":
            expression = f"loom::buffer::cast<unsigned char>({expression})"
        return expression

    def _render_command_program(self, operation: Operation) -> None:
        scope = self.operation_scopes[id(operation)]
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        self._validate_projected_attributes(
            operation,
            (
                "callee",
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
                operation, "command program body must have one structured block"
            )
        entry = body.blocks[0]
        self.writer.attributes(self._command_attributes(operation))
        self.writer.open(self._command_signature(operation, list(entry.arg_ids)))
        self._render_block(entry)
        self.writer.close()

    def _render_command_launch(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        callee = self.command_names.get(symbol)
        if callee is None:
            raise CxxRenderError(
                operation, f"launched command program {symbol!r} is not present"
            )
        arguments = [
            self._staged_argument(value, value)
            for value in self._command_source_operands(operation)
        ]
        self.writer.line(f"loom::command::launch<{callee}>({', '.join(arguments)});")
        return ("callee",)

    def _render_kernel_launch(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        pipeline_name = self.pipeline_names.get(symbol)
        if pipeline_name is not None:
            pipeline = self.pipeline_callables[symbol]
            source_arguments = self._pipeline_source_arguments(
                operation,
                pipeline,
                resolved.value_ids("workloads"),
                resolved.value_ids("arguments"),
            )
            self.writer.line(
                f"loom::kernel::launch<{pipeline_name}>({', '.join(source_arguments)});"
            )
            return ("callee",)
        expected_elements = self.kernel_buffer_elements.get(symbol)
        if expected_elements is None:
            raise CxxRenderError(
                operation, f"launched kernel {symbol!r} is not present"
            )
        callee = self.kernel_names[symbol]
        workload_arguments = [
            self.expression(value)
            for value in self._kernel_workload_operands(operation)
        ]
        arguments = resolved.value_ids("arguments")
        if len(arguments) != len(expected_elements):
            raise CxxRenderError(
                operation, "kernel launch argument arity is inconsistent"
            )
        rendered: list[str] = []
        for argument, element in zip(arguments, expected_elements, strict=True):
            expression = self.expression(argument)
            if element is not None:
                expected_type = f"loom::type::buffer<{element}>"
                if self.expression_type(argument) != expected_type:
                    expression = f"loom::buffer::cast<{element}>({expression})"
            rendered.append(expression)
        if workload_arguments:
            rendered.insert(
                0,
                "loom::kernel::workload(" + ", ".join(workload_arguments) + ")",
            )
        self.writer.line(f"loom::kernel::launch<{callee}>({', '.join(rendered)});")
        return ("callee",)

    def _render_command_parameter(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        result = resolved.value_id("result")
        result_type = self.module.values[result].type
        if not (
            isinstance(result_type, ShapedType)
            and result_type.type_kind == TypeKind.VIEW
        ):
            raise CxxRenderError(operation, "command parameter result is not a view")
        shape, element, encoding = self.types.shaped_arguments(
            result_type, result in self.signed_values, const=True
        )
        template_arguments = [shape, element]
        if encoding is not None:
            template_arguments.append(encoding)
        dependencies = self._view_dependency_expressions(operation, result_type)
        pattern = resolved.attr("pattern")
        if not isinstance(pattern, (str, Utf8Attr)):
            raise CxxRenderError(operation, "command parameter pattern is not text")
        arguments = [
            self.expression(resolved.value_id("source")),
            *dependencies,
            _quote_string(pattern),
            *(self.expression(value) for value in resolved.value_ids("substitutions")),
        ]
        expression = (
            "loom::command::parameter<"
            + ", ".join(template_arguments)
            + ">("
            + ", ".join(arguments)
            + ")"
        )
        self._declare(
            result,
            expression,
            expression_type=self.types.view(
                result_type, result in self.signed_values, const=True
            ),
        )
        return ("pattern",)

    def _render_command_parameter_base(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        source = resolved.value_id("source")
        source_type = self.module.values[source].type
        if not (
            isinstance(source_type, ShapedType)
            and source_type.type_kind == TypeKind.VIEW
        ):
            raise CxxRenderError(
                operation, "command parameter base source is not a view"
            )
        element = self.types.scalar(
            source_type.element_type, source in self.signed_values
        )
        result = resolved.value_id("result")
        self._declare(
            result,
            f"loom::command::base({self.expression(source)})",
            expression_type=f"loom::type::buffer<const {element}>",
        )
        return ()

    def _render_schedule_region(self, operation: Operation) -> None:
        body = self._resolved(operation).region("body")
        if body is None or len(body.blocks) != 1 or body.blocks[0].arg_ids:
            raise CxxRenderError(
                operation, "schedule region must have one argument-free block"
            )
        spelling = operation.name.rsplit(".", 1)[1]
        self.writer.open(f"[[loom::{spelling}]]")
        self._render_block(body.blocks[0])
        self.writer.close()

    def _render_command_op(self, operation: Operation) -> tuple[str, ...] | None:
        if operation.name in {"command.return", "command.yield"}:
            return ()
        if operation.name == "command.launch":
            return self._render_command_launch(operation)
        if operation.name == "kernel.launch":
            return self._render_kernel_launch(operation)
        if operation.name == "command.parameter":
            return self._render_command_parameter(operation)
        if operation.name == "command.parameter.base":
            return self._render_command_parameter_base(operation)
        if operation.name in {"command.serial", "command.concurrent"}:
            self._render_schedule_region(operation)
            return ()
        return None
