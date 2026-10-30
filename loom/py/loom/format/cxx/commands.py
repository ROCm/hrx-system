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
    PredicateListAttr,
    ShapedType,
    TypeKind,
    Utf8Attr,
)


@dataclass(frozen=True, slots=True)
class _SourceParameter:
    """One reconstructed C++ parameter and its flattened type dependencies."""

    value_id: int
    dependencies: tuple[int, ...]
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

    def _source_view_dependencies(self, value_id: int) -> tuple[int, ...]:
        value_type = self.module.values[value_id].type
        if not (
            isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VIEW
        ):
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
    ) -> tuple[list[_SourceParameter], list[_SourceParameter]]:
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
        specializations = arguments[:specialization_count]
        bindings = arguments[specialization_count:]
        specialization_set = set(specializations)

        binding_parameters: list[_SourceParameter] = []
        binding_dependencies: set[int] = set()
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
            dependencies = self._source_view_dependencies(binding)
            for dependency in dependencies:
                if dependency not in specialization_set:
                    raise CxxRenderError(
                        operation,
                        "view binding references a value outside its "
                        "specialization prefix",
                    )
                if dependency in binding_dependencies:
                    raise CxxRenderError(
                        operation,
                        "view bindings share a dependent signature value",
                    )
                binding_dependencies.add(dependency)
            binding_parameters.append(
                _SourceParameter(binding, dependencies, binding=True)
            )

        remaining = [
            value for value in specializations if value not in binding_dependencies
        ]
        remaining_set = set(remaining)
        specialization_dependencies: set[int] = set()
        specialization_views: dict[int, tuple[int, ...]] = {}
        for value in remaining:
            dependencies = self._source_view_dependencies(value)
            if not dependencies:
                continue
            for dependency in dependencies:
                if dependency not in remaining_set:
                    raise CxxRenderError(
                        operation,
                        "specialization view references a value outside "
                        "its flattened source parameter",
                    )
                if dependency in specialization_dependencies:
                    raise CxxRenderError(
                        operation,
                        "specialization views share a dependent signature value",
                    )
                specialization_dependencies.add(dependency)
            specialization_views[value] = dependencies

        specialization_parameters = [
            _SourceParameter(
                value,
                specialization_views.get(value, ()),
                binding=False,
            )
            for value in remaining
            if value not in specialization_dependencies
        ]
        expected_specializations = [
            component
            for parameter in specialization_parameters
            for component in (*parameter.dependencies, parameter.value_id)
        ]
        expected_specializations.extend(
            component
            for parameter in binding_parameters
            for component in parameter.dependencies
        )
        if expected_specializations != specializations:
            raise CxxRenderError(
                operation,
                "staged signature dependencies do not form a C++ source-order "
                "projection",
            )
        return specialization_parameters, binding_parameters

    def _staged_parameter_type(self, operation: Operation, value_id: int) -> str:
        value_type = self.module.values[value_id].type
        if isinstance(value_type, BufferType):
            return "loom::type::buffer<unsigned char>"
        if isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VIEW:
            return self.types.view(value_type, value_id in self.signed_values)
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
    ) -> tuple[list[str], list[_SourceParameter], list[_SourceParameter]]:
        specializations, bindings = self._source_parameters(
            operation,
            arguments,
            allow_nonstorage_bindings=allow_nonstorage_bindings,
        )
        parameters: list[str] = []
        for ordinal, parameter in enumerate((*specializations, *bindings)):
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
        target_symbol = resolved.attr("target")
        if target_symbol is not None:
            target_info = self.targets.get(str(target_symbol))
            if target_info is None:
                raise CxxRenderError(
                    operation,
                    f"target record {target_symbol!s} is not present in the module",
                )
            if target_info.operation_name != "amdgpu.target":
                raise CxxRenderError(
                    target_info.operation,
                    f"target record {target_info.operation_name} has no C++ "
                    "declaration projection",
                )
            target_name = self.target_names[str(target_symbol)]
            attributes.append(f"loom::target({target_name})")
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
        if resolved.attr("scope") != "kernel":
            raise CxxRenderError(
                operation, "only kernel-scoped pipeline contracts are launchable"
            )
        if resolved.attr("visibility") is not None:
            raise CxxRenderError(
                operation,
                "public pipeline declarations have no C++ contract projection",
            )
        if resolved.attr("retain") is not None:
            raise CxxRenderError(
                operation, "retained pipeline declarations have no C++ projection"
            )
        symbol = str(resolved.attr("callee"))
        attributes = [
            'loom::pipeline("kernel")',
            f"loom::symbol({_quote_string(symbol)})",
        ]
        target_symbol = resolved.attr("target")
        if target_symbol is not None:
            target_info = self.targets.get(str(target_symbol))
            if target_info is None:
                raise CxxRenderError(
                    operation,
                    f"target record {target_symbol!s} is not present in the module",
                )
            if target_info.operation_name != "amdgpu.target":
                raise CxxRenderError(
                    target_info.operation,
                    f"target record {target_info.operation_name} has no C++ "
                    "declaration projection",
                )
            attributes.append(f"loom::target({self.target_names[str(target_symbol)]})")
        return attributes

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
        arguments = list(operation.operands)
        parameters, _, _ = self._bind_staged_parameters(
            operation, arguments, allow_nonstorage_bindings=True
        )
        symbol = str(self._resolved(operation).attr("callee"))
        signature = f"void {self.pipeline_names[symbol]}(" + ", ".join(parameters) + ")"
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
        self.writer.attributes(self._pipeline_attributes(operation))
        self.writer.line(signature + ";")

    def _render_command_program(self, operation: Operation) -> None:
        scope = self.operation_scopes[id(operation)]
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        self._validate_projected_attributes(
            operation,
            (
                "callee",
                "visibility",
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
        specializations = resolved.value_ids("specializations")
        bindings = resolved.value_ids("bindings")
        source_specializations, source_bindings = self._source_parameters(
            operation,
            [*specializations, *bindings],
            specialization_count=len(specializations),
        )
        arguments = [
            self.expression(parameter.value_id)
            for parameter in (*source_specializations, *source_bindings)
        ]
        self.writer.line(f"loom::command::launch<{callee}>({', '.join(arguments)});")
        return ("callee",)

    def _render_kernel_launch(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        pipeline_name = self.pipeline_names.get(symbol)
        if pipeline_name is not None:
            pipeline = self.pipeline_declarations[symbol]
            source_specializations, source_bindings = self._source_parameters(
                pipeline,
                list(pipeline.operands),
                allow_nonstorage_bindings=True,
            )
            workloads = resolved.value_ids("workloads")
            arguments = resolved.value_ids("arguments")
            if len(arguments) != len(source_bindings):
                raise CxxRenderError(
                    operation, "pipeline launch binding arity is inconsistent"
                )
            source_arguments: list[str] = []
            workload_ordinal = 0
            for parameter in source_specializations:
                workload_ordinal += len(parameter.dependencies) + 1
                if workload_ordinal > len(workloads):
                    raise CxxRenderError(
                        operation,
                        "pipeline launch specialization arity is inconsistent",
                    )
                source_arguments.append(
                    self.expression(workloads[workload_ordinal - 1])
                )
            for parameter, argument in zip(source_bindings, arguments, strict=True):
                workload_ordinal += len(parameter.dependencies)
                if workload_ordinal > len(workloads):
                    raise CxxRenderError(
                        operation,
                        "pipeline launch specialization arity is inconsistent",
                    )
                source_arguments.append(self.expression(argument))
            if workload_ordinal != len(workloads):
                raise CxxRenderError(
                    operation, "pipeline launch specialization arity is inconsistent"
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
        workloads = resolved.value_ids("workloads")
        component_counts = self.kernel_workload_component_counts.get(symbol)
        if component_counts is None:
            raise CxxRenderError(
                operation,
                f"kernel {symbol!r} has no rendered launch configuration",
            )
        if sum(component_counts) != len(workloads):
            raise CxxRenderError(
                operation, "kernel launch workload arity is inconsistent"
            )
        workload_arguments: list[str] = []
        component_ordinal = 0
        for component_count in component_counts:
            component_ordinal += component_count
            workload_arguments.append(self.expression(workloads[component_ordinal - 1]))
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
        self.writer.line(f"loom::kernel::launch<{symbol}>({', '.join(rendered)});")
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
        shape, element, encoding = self.types.view_arguments(
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

    def _render_command_schedule(self, operation: Operation) -> None:
        body = self._resolved(operation).region("body")
        if body is None or len(body.blocks) != 1 or body.blocks[0].arg_ids:
            raise CxxRenderError(
                operation, "command schedule must have one argument-free block"
            )
        spelling = operation.name.removeprefix("command.")
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
            self._render_command_schedule(operation)
            return ()
        return None
