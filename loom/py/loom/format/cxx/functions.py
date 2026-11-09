# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ordinary function signatures, calls, and result records."""

from __future__ import annotations

from loom.format.cxx.configs import CxxConfigRenderer
from loom.format.cxx.context import cxx_string_literal
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.naming import clean_identifier
from loom.ir import BufferType, Operation


class CxxFunctionRenderer(CxxConfigRenderer):
    """Projects ordinary Func values through parameters and return values."""

    def _buffer_element(self, value_id: int, element: str = "unsigned char") -> str:
        if self.buffer_access.is_immutable(value_id, self.buffer_projection):
            return f"const {element}"
        return element

    def _plan_function_projections(self) -> None:
        used_symbols = set(self.function_names)
        for symbol, projections in self.buffer_access.projections.items():
            for index, projection in enumerate(projections):
                if index == 0:
                    name = self.function_names[symbol]
                    projected_symbol = symbol
                else:
                    ordinal = index
                    projected_symbol = f"{symbol}.buffer{ordinal}"
                    name = clean_identifier(projected_symbol, "function")
                    while (
                        projected_symbol in used_symbols
                        or name in self.translation_unit_names
                    ):
                        ordinal += 1
                        projected_symbol = f"{symbol}.buffer{ordinal}"
                        name = clean_identifier(projected_symbol, "function")
                used_symbols.add(projected_symbol)
                self.translation_unit_names.add(name)
                self.function_projection_names[projection] = name
                self.function_projection_symbols[projection] = projected_symbol

    def _source_parameter_type(self, value_id: int) -> str:
        """Returns the complete C++ type transporting one source parameter."""

        value_type = self.module.values[value_id].type
        if isinstance(value_type, BufferType):
            element = self._buffer_element(value_id, self._parameter_element(value_id))
            return f"loom::type::buffer<{element}>"
        return self.cpp_type(value_id)

    def _function_result_type(self, function: Operation) -> str:
        if function.tied_results:
            raise CxxRenderError(
                function,
                "destructive function ties have no exact C++ projection; "
                "source references return ordinary SSA values",
            )
        results = tuple(
            parameter.value_id
            for parameter in self._flat_source_parameters(
                function, list(function.results)
            )
        )
        if not results:
            return "void"
        if len(results) == 1:
            return self._function_result_component_type(results[0])
        symbol = self.function_projection_symbols[self.buffer_projection]
        fields = tuple(
            (f"value{index}", self._function_result_component_type(result))
            for index, result in enumerate(results)
        )
        return self._record_type(
            clean_identifier(symbol, "function") + "_result", fields
        )

    def _function_result_component_type(self, value_id: int) -> str:
        # A buffer result carries one root, independently of any adjacent byte
        # offset. A C++ pointer would introduce another offset on reimport.
        if isinstance(self.module.values[value_id].type, BufferType):
            return f"loom::type::buffer<{self._buffer_element(value_id)}>"
        return self.cpp_type(value_id)

    def _buffer_argument(self, value_id: int, element: str) -> str:
        expression = self.expression(value_id)
        source_type = self.expression_type(value_id)
        expected_type = f"loom::type::buffer<{element}>"
        if source_type.endswith("*"):
            expression = f"loom::buffer::root({expression})"
            if source_type[:-1].strip() == element:
                return expression
        elif source_type == expected_type:
            return expression
        return f"loom::buffer::cast<{element}>({expression})"

    def _declare_loop_carrier(self, result: int, source: int) -> None:
        if isinstance(self.module.values[result].type, BufferType):
            element = self._buffer_element(result)
            self._declare(
                result,
                self._buffer_argument(source, element),
                expression_type=f"loom::type::buffer<{element}>",
            )
        else:
            self._declare(result, self._loop_initializer(source))

    def _render_function_call(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        symbol = str(resolved.attr("callee"))
        if symbol not in self.function_names:
            raise CxxRenderError(
                operation, f"called function {symbol!r} is not present in the module"
            )
        bindings = (
            self.buffer_projection.calls
            if self.buffer_projection is not None
            else self.buffer_access.calls
        )
        projection = bindings[id(operation)]
        callee = self.function_projection_names[projection]
        if callee in self.temporary_names:
            callee = f"::{callee}"
        if operation.tied_results:
            raise CxxRenderError(
                operation,
                "destructive call ties have no exact C++ projection; "
                "source references return ordinary SSA values",
            )
        expected_buffers = self.function_buffer_elements[projection]
        source_operands = self._flat_source_parameters(
            operation, list(operation.operands)
        )
        function = projection.function
        formal_ids = (
            function.regions[0].blocks[0].arg_ids
            if function.name == "func.def"
            else function.operands
        )
        formals = self._flat_source_parameters(function, list(formal_ids))
        arguments_list: list[str] = []
        for index, parameter in enumerate(source_operands):
            operand = parameter.value_id
            expression = self.expression(operand)
            expected_element = expected_buffers[index]
            if expected_element is not None:
                if self.buffer_access.is_immutable(
                    operand, self.buffer_projection
                ) and not expected_element.startswith("const "):
                    raise CxxRenderError(
                        operation,
                        "immutable buffer argument has no access-qualified "
                        "external declaration",
                    )
                expression = self._buffer_argument(operand, expected_element)
            else:
                expression = self.numeric_argument(
                    operand, self.cpp_type(formals[index].value_id)
                )
            arguments_list.append(expression)
        arguments = ", ".join(arguments_list)
        call = f"{callee}({arguments})"
        source_results = self._flat_source_parameters(
            operation, list(operation.results)
        )
        for result in source_results:
            self._hide_source_dependencies(result)
        if not source_results:
            self.writer.line(f"{call};")
        elif len(source_results) == 1:
            result = source_results[0].value_id
            self._declare(
                result,
                call,
                expression_type=self._function_result_component_type(result),
            )
        else:
            fields = tuple(
                (f"value{index}", self._function_result_component_type(result.value_id))
                for index, result in enumerate(source_results)
            )
            record_type = self._record_type(
                clean_identifier(
                    self.function_projection_symbols[projection], "function"
                )
                + "_result",
                fields,
            )
            record_name = self._temporary_name("call_result")
            self.writer.line(f"{record_type} {record_name} = {call};")
            for index, result in enumerate(source_results):
                self._declare(
                    result.value_id,
                    f"{record_name}.value{index}",
                    expression_type=fields[index][1],
                )
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
        symbol = self.function_projection_symbols[self.buffer_projection]
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
        parameters: list[str] = []
        buffer_elements: list[str | None] = []
        source_parameters = self._flat_source_parameters(function, argument_ids)
        for parameter in source_parameters:
            argument = parameter.value_id
            value_type = self.module.values[argument].type
            type_name = self._source_parameter_type(argument)
            buffer_elements.append(
                self._buffer_element(argument, self._parameter_element(argument))
                if isinstance(value_type, BufferType)
                else None
            )
            name = self.names[argument]
            parameters.append(f"{type_name} {name}")
            if bind_parameters:
                self._hide_source_dependencies(parameter)
                self.expressions[argument] = name
                self.expression_types[argument] = type_name

        self.function_buffer_elements[self.buffer_projection] = tuple(buffer_elements)
        visibility = resolved.attr("visibility")
        original = str(resolved.attr("callee"))
        primary = self.buffer_access.projections[original][0] is self.buffer_projection
        storage = (
            ""
            if function.name == "func.decl" or (visibility == "public" and primary)
            else "static "
        )
        return (
            f"{storage}{self._function_result_type(function)} "
            f"{self.function_projection_names[self.buffer_projection]}("
            + ", ".join(parameters)
            + ")"
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
        results = self._flat_source_parameters(operation, list(operation.operands))
        signature = self._flat_source_parameters(
            self.buffer_projection.function,
            list(self.buffer_projection.function.results),
        )
        expressions = [
            self._buffer_argument(
                result.value_id, self._buffer_element(formal.value_id)
            )
            if isinstance(self.module.values[result.value_id].type, BufferType)
            else self.numeric_argument(result.value_id, self.cpp_type(formal.value_id))
            for result, formal in zip(results, signature, strict=True)
        ]
        if not results:
            self.writer.line("return;")
        elif len(results) == 1:
            self.writer.line(f"return {expressions[0]};")
        else:
            values = ", ".join(expressions)
            self.writer.line(f"return {{{values}}};")

    def _render_function_exit(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        body = resolved.region("body")
        if body is None or len(body.blocks) != 1:
            raise CxxRenderError(operation, "function exit must have one body block")
        condition = self.expression(resolved.value_id("condition"))
        self.writer.open(f"if ({condition})")
        self._render_block(body.blocks[0])
        self.writer.close()

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
        self._render_block(body.blocks[0])
        self.writer.close()
