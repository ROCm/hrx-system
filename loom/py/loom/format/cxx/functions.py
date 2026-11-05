# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ordinary function signatures, calls, and tied value projection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from loom.format.cxx.commands import SourceParameter
from loom.format.cxx.configs import CxxConfigRenderer
from loom.format.cxx.context import (
    _strip_outer_parentheses,
    cxx_string_literal,
)
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.naming import clean_identifier
from loom.ir import BufferType, Module, Operation


@dataclass(frozen=True, slots=True)
class _ReferenceProjection:
    """One complete source parameter carried through a mutable reference."""

    operand_id: int
    result_id: int


@dataclass(frozen=True, slots=True)
class _FunctionProjection:
    """One function-like signature split into returns and reference updates."""

    ordinary_result_ids: tuple[int, ...]
    ordinary_component_indices: tuple[int, ...]
    references: tuple[_ReferenceProjection, ...]
    return_targets: tuple[int | None, ...]

    @property
    def reference_operand_ids(self) -> frozenset[int]:
        return frozenset(reference.operand_id for reference in self.references)


class CxxFunctionRenderer(CxxConfigRenderer):
    """Projects Func callables without discarding linear result ownership."""

    def __init__(self, module: Module):
        super().__init__(module)
        self._function_projections: dict[int, _FunctionProjection] = {}
        self._current_function_projection: _FunctionProjection | None = None
        self._plan_mutable_call_carriers()

    @staticmethod
    def _parameter_components(parameter: SourceParameter) -> tuple[int, ...]:
        return (*parameter.dependencies, parameter.value_id)

    def _source_parameter_type(self, value_id: int) -> str:
        """Returns the complete C++ type transporting one source parameter."""

        value_type = self.module.values[value_id].type
        if isinstance(value_type, BufferType):
            return f"loom::type::buffer<{self._parameter_element(value_id)}>"
        return self.cpp_type(value_id)

    def _plan_mutable_call_carriers(self) -> None:
        """Keeps each tied source parameter available as one writable lvalue."""

        for operation in self.tied_calls:
            tied_indices = {tied.operand_index for tied in operation.tied_results}
            component_offset = 0
            for parameter in self._flat_source_parameters(
                operation, list(operation.operands)
            ):
                component_count = len(self._parameter_components(parameter))
                component_indices = range(
                    component_offset, component_offset + component_count
                )
                component_offset += component_count
                if tied_indices.intersection(component_indices):
                    self.mutable_carriers.add(parameter.value_id)

    def _project_function_ties(
        self,
        operation: Operation,
        argument_ids: list[int],
        result_ids: list[int],
    ) -> _FunctionProjection:
        """Maps representable tied results to native mutable references."""

        source_arguments = self._flat_source_parameters(operation, argument_ids)
        source_results = self._flat_source_parameters(operation, result_ids)
        result_index = {value_id: index for index, value_id in enumerate(result_ids)}
        argument_positions: dict[int, set[int]] = {}
        for index, value_id in enumerate(argument_ids):
            argument_positions.setdefault(value_id, set()).add(index)
        argument_components: list[tuple[int, ...]] = []
        component_offset = 0
        for parameter in source_arguments:
            component_count = len(self._parameter_components(parameter))
            argument_components.append(
                tuple(range(component_offset, component_offset + component_count))
            )
            component_offset += component_count
        argument_ordinals = {
            components: ordinal
            for ordinal, components in enumerate(argument_components)
        }
        ties_by_result = {tied.result_index: tied for tied in operation.tied_results}

        ordinary_result_ids: list[int] = []
        ordinary_component_indices: list[int] = []
        references: list[_ReferenceProjection] = []
        reference_argument_ordinals: list[int] = []
        seen_reference_arguments: set[int] = set()
        saw_reference_result = False
        projected_tie_count = 0
        component_offset = 0
        for result_parameter in source_results:
            component_count = len(self._parameter_components(result_parameter))
            component_result_indices = tuple(
                range(component_offset, component_offset + component_count)
            )
            component_offset += component_count
            component_ties = tuple(
                ties_by_result.get(index) for index in component_result_indices
            )
            if all(tied is None for tied in component_ties):
                if saw_reference_result:
                    raise CxxRenderError(
                        operation,
                        "ordinary results after tied results have no exact C++ "
                        "function projection",
                    )
                ordinary_result_ids.append(result_parameter.value_id)
                ordinary_component_indices.extend(component_result_indices)
                continue
            saw_reference_result = True
            if any(tied is None for tied in component_ties):
                raise CxxRenderError(
                    operation,
                    "a partial source-parameter tie has no exact C++ reference "
                    "projection",
                )
            ties = tuple(tied for tied in component_ties if tied is not None)
            tied_operand_indices = tuple(tied.operand_index for tied in ties)
            argument_ordinal = argument_ordinals.get(tied_operand_indices)
            if argument_ordinal is None:
                raise CxxRenderError(
                    operation,
                    "tied results do not cover one complete C++ source parameter",
                )
            argument_parameter = source_arguments[argument_ordinal]
            tied_operand_set = set(tied_operand_indices)
            if any(
                argument_positions[argument_ids[index]] - tied_operand_set
                for index in tied_operand_indices
            ):
                raise CxxRenderError(
                    operation,
                    "a tied source parameter aliases another C++ call argument",
                )
            if argument_parameter.value_id in self.counted_loop_inductions:
                raise CxxRenderError(
                    operation,
                    "a tied counted-loop induction would mutate C++ loop control",
                )
            if self._source_parameter_type(
                result_parameter.value_id
            ) != self._source_parameter_type(argument_parameter.value_id):
                raise CxxRenderError(
                    operation,
                    "a source-type-changing tied result has no exact C++ reference "
                    "projection",
                )
            if argument_ordinal in seen_reference_arguments:
                raise CxxRenderError(
                    operation,
                    "one C++ source parameter cannot carry multiple tied results",
                )
            seen_reference_arguments.add(argument_ordinal)
            reference_argument_ordinals.append(argument_ordinal)
            projected_tie_count += len(ties)
            references.append(
                _ReferenceProjection(
                    operand_id=argument_parameter.value_id,
                    result_id=result_parameter.value_id,
                )
            )

        if projected_tie_count != len(operation.tied_results):
            raise CxxRenderError(
                operation, "function ties do not form complete C++ source parameters"
            )
        if reference_argument_ordinals != sorted(reference_argument_ordinals):
            raise CxxRenderError(
                operation,
                "tied result order differs from C++ mutable parameter order",
            )

        return_targets: list[int | None] = [None] * len(result_ids)
        for reference in references:
            return_targets[result_index[reference.result_id]] = reference.operand_id
        return _FunctionProjection(
            ordinary_result_ids=tuple(ordinary_result_ids),
            ordinary_component_indices=tuple(ordinary_component_indices),
            references=tuple(references),
            return_targets=tuple(return_targets),
        )

    def _function_projection(self, function: Operation) -> _FunctionProjection:
        projection = self._function_projections.get(id(function))
        if projection is not None:
            return projection
        resolved = self._resolved(function)
        if function.name == "func.decl":
            if function.tied_results:
                raise CxxRenderError(
                    function,
                    "a tied external declaration has no exact C++ projection; "
                    "a declaration alone cannot select the value-reference ABI",
                )
            argument_ids = list(function.operands)
        else:
            if function.tied_results and resolved.attr("visibility") == "public":
                raise CxxRenderError(
                    function,
                    "a public tied function has no exact C++ projection; the "
                    "value-reference ABI is internal",
                )
            body = resolved.region("body")
            if body is None or len(body.blocks) != 1:
                raise CxxRenderError(
                    function, "function body must have one structured entry block"
                )
            argument_ids = list(body.blocks[0].arg_ids)
        projection = self._project_function_ties(
            function, argument_ids, list(function.results)
        )
        self._function_projections[id(function)] = projection
        return projection

    def _function_result_type(self, function: Operation) -> str:
        results = self._function_projection(function).ordinary_result_ids
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
        projection = self._project_function_ties(
            operation, list(operation.operands), list(operation.results)
        )
        references = {
            reference.operand_id: reference for reference in projection.references
        }
        expected_buffers = self.function_buffer_elements.get(symbol)
        source_operands = self._flat_source_parameters(
            operation, list(operation.operands)
        )
        arguments_list: list[str] = []
        reference_carriers: dict[int, tuple[str, str]] = {}
        for index, parameter in enumerate(source_operands):
            operand = parameter.value_id
            expression = self.expression(operand)
            expected_element = (
                expected_buffers[index]
                if expected_buffers is not None and index < len(expected_buffers)
                else None
            )
            if expected_element is not None:
                expected_type = f"loom::type::buffer<{expected_element}>"
                source_type = self.expression_type(operand)
                if source_type.endswith("*"):
                    expression = f"loom::buffer::root({expression})"
                    source_element = source_type[:-1].strip()
                    if source_element != expected_element:
                        expression = (
                            f"loom::buffer::cast<{expected_element}>({expression})"
                        )
                elif source_type != expected_type:
                    expression = f"loom::buffer::cast<{expected_element}>({expression})"
            reference = references.get(operand)
            if reference is not None:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression) is None:
                    carrier = self._temporary_name(f"{self.names[operand]}_carrier")
                    self.writer.line(
                        f"auto {carrier} = {_strip_outer_parentheses(expression)};"
                    )
                    expression = carrier
                    expression_type = self.cpp_type(reference.result_id)
                else:
                    expression_type = self.expression_type(operand)
                reference_carriers[operand] = (expression, expression_type)
            arguments_list.append(expression)
        arguments = ", ".join(arguments_list)
        call = f"{callee}({arguments})"
        ordinary_result_values = [
            operation.results[index] for index in projection.ordinary_component_indices
        ]
        source_results = self._flat_source_parameters(operation, ordinary_result_values)
        if not source_results:
            self.writer.line(f"{call};")
        elif len(source_results) == 1:
            self._declare(source_results[0].value_id, call)
        else:
            fields = tuple(
                (f"value{index}", self.cpp_type(result.value_id))
                for index, result in enumerate(source_results)
            )
            record_type = self._record_type(
                clean_identifier(symbol, "function") + "_result", fields
            )
            record_name = self._temporary_name("call_result")
            self.writer.line(f"{record_type} {record_name} = {call};")
            for index, result in enumerate(source_results):
                self._declare(result.value_id, f"{record_name}.value{index}")
        for reference in projection.references:
            carrier, expression_type = reference_carriers[reference.operand_id]
            self.expressions[reference.result_id] = carrier
            self.expression_types[reference.result_id] = expression_type
        projected_attributes = ["callee"]
        effect_contract = resolved.attr("effect_contract")
        if effect_contract in {"pure", "readonly"}:
            # A C++ call carries its effect contract from the callable
            # definition. The importer reconstructs the explicit func.call
            # contract after the emitted definition is parsed and analyzed.
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
        if (
            function.name == "func.decl"
            and (import_module := resolved.attr("import_module")) is not None
        ):
            arguments = [cxx_string_literal(import_module.encode("utf-8"))]
            import_symbol = resolved.attr("import_symbol")
            if import_symbol is not None:
                arguments.append(cxx_string_literal(import_symbol.encode("utf-8")))
            attributes.append(f"loom::import({', '.join(arguments)})")
        if retain := self._retention_attribute(function):
            attributes.append(retain)
        target = resolved.attr("target")
        if target is not None:
            attributes.append(self._target_attribute(function, str(target)))
        temperature = resolved.attr("temperature")
        if temperature in {"hot", "cold"}:
            attributes.append(f"loom::{temperature}")
        elif temperature is not None:
            raise CxxRenderError(
                function,
                f"function temperature {temperature!r} has no C++ projection",
            )
        inline_policy = resolved.attr("inline_policy")
        if inline_policy == "inline":
            attributes.append("loom::inline")
        elif inline_policy == "noinline":
            attributes.append("loom::noinline")
        elif inline_policy is not None:
            raise CxxRenderError(
                function,
                f"function inline policy {inline_policy!r} has no C++ projection",
            )
        return attributes

    def _function_signature(self, function: Operation, *, bind_parameters: bool) -> str:
        resolved = self._resolved(function)
        if function.name == "func.decl":
            argument_ids = list(function.operands)
        else:
            body = resolved.region("body")
            if body is None or len(body.blocks) != 1:
                raise CxxRenderError(
                    function, "function body must have one structured entry block"
                )
            argument_ids = list(body.blocks[0].arg_ids)
        reference_ids = self._function_projection(function).reference_operand_ids
        parameters: list[str] = []
        buffer_elements: list[str | None] = []
        source_parameters = self._flat_source_parameters(function, argument_ids)
        for parameter in source_parameters:
            argument = parameter.value_id
            value_type = self.module.values[argument].type
            type_name = self._source_parameter_type(argument)
            buffer_elements.append(
                self._parameter_element(argument)
                if isinstance(value_type, BufferType)
                else None
            )
            reference = "&" if argument in reference_ids else ""
            name = self.names[argument]
            parameters.append(f"{type_name}{reference} {name}")
            if bind_parameters:
                self.expressions[argument] = name
                self.expression_types[argument] = type_name

        symbol = str(resolved.attr("callee"))
        self.function_buffer_elements[symbol] = tuple(buffer_elements)
        visibility = resolved.attr("visibility")
        storage = (
            "" if function.name == "func.decl" or visibility == "public" else "static "
        )
        return (
            f"{storage}{self._function_result_type(function)} "
            f"{self.function_names[symbol]}(" + ", ".join(parameters) + ")"
        )

    def _render_function_declaration(self, function: Operation) -> None:
        if function.name == "func.decl":
            self._validate_projected_attributes(
                function,
                (
                    "callee",
                    "import_module",
                    "import_symbol",
                    "inline_policy",
                    "retain",
                    "target",
                    "temperature",
                ),
            )
        self._analyze_buffers(function)
        self.writer.attributes(self._function_attributes(function))
        self.writer.line(
            self._function_signature(function, bind_parameters=False) + ";"
        )

    def _render_function_return(self, operation: Operation) -> None:
        projection = self._current_function_projection
        if projection is None:
            raise CxxRenderError(
                operation, "function return appears outside a projected function"
            )
        for target, source in zip(
            projection.return_targets, operation.operands, strict=True
        ):
            if target is not None and target == source:
                expression = self.expression(target)
                self.writer.line(f"{expression} = {expression};")
        self._assign_values(projection.return_targets, operation.operands)
        ordinary_values = [
            operation.operands[index] for index in projection.ordinary_component_indices
        ]
        results = self._flat_source_parameters(operation, ordinary_values)
        if not results:
            self.writer.line("return;")
        elif len(results) == 1:
            self.writer.line(f"return {self.expression(results[0].value_id)};")
        else:
            values = ", ".join(self.expression(result.value_id) for result in results)
            self.writer.line(f"return {{{values}}};")

    def _render_function(self, function: Operation) -> None:
        scope = self.operation_scopes[id(function)]
        self.temporary_names = set(self.scope_names[scope])
        self.rendered_assumptions_by_block.clear()
        self._validate_projected_attributes(
            function,
            (
                "callee",
                "visibility",
                "inline_policy",
                "retain",
                "target",
                "temperature",
            ),
        )
        self._analyze_buffers(function)
        body = self._resolved(function).region("body")
        assert body is not None
        self.writer.attributes(self._function_attributes(function))
        self.writer.open(self._function_signature(function, bind_parameters=True))
        previous_projection = self._current_function_projection
        self._current_function_projection = self._function_projection(function)
        try:
            self._render_block(body.blocks[0])
        finally:
            self._current_function_projection = previous_projection
        self.writer.close()
