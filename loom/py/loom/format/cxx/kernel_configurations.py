# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel launch-configuration projection into Loom C++."""

from __future__ import annotations

from dataclasses import dataclass

from loom.format.cxx.context import walk_operations
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.pipelines import CxxPipelineRenderer
from loom.ir import (
    Block,
    BufferType,
    Operation,
    PredicateListAttr,
    ShapedType,
    TypeKind,
)


@dataclass(frozen=True, slots=True)
class _ConfigurationParameter:
    """One source parameter reconstructed from native config-region arguments."""

    # Payload identity represented by one C++ configuration parameter.
    value_id: int
    # Number of flat dimensions, layout, and payload components consumed.
    component_count: int


@dataclass(frozen=True, slots=True)
class _DimensionProjection:
    """One source launch attribute and the declarations it subsumes."""

    attribute: str | None
    declarations: tuple[Operation, ...]


class CxxKernelConfigurationRenderer(CxxPipelineRenderer):
    """Renders fixed and workload-dependent kernel launch configurations."""

    def _launch_dimension_attribute(
        self,
        kernel: Operation,
        launch: Operation,
        group: str,
        values: list[int],
    ) -> _DimensionProjection | None:
        """Recovers one exact, ranged, or omitted source dimension group."""

        definitions = [self.definitions.get(value_id) for value_id in values]
        if all(
            definition is not None and definition.name == "index.constant"
            for definition in definitions
        ):
            dimensions: list[str] = []
            for definition in definitions:
                assert definition is not None
                value = definition.attributes.get("value")
                if type(value) is not int or value <= 0 or value > 0x7FFFFFFF:
                    raise CxxRenderError(
                        definition,
                        "kernel launch dimension is not a positive i32 source constant",
                    )
                self._validate_projected_attributes(definition, ("value",))
                dimensions.append(str(value))
            return _DimensionProjection(f"loom::{group}({', '.join(dimensions)})", ())

        if not all(
            definition is not None and definition.name == "config.get"
            for definition in definitions
        ):
            return None

        declarations: list[Operation] = []
        ranges: list[tuple[int, int] | None] = []
        for axis, definition in zip("xyz", definitions, strict=True):
            assert definition is not None
            self._validate_projected_attributes(definition, ("config",))
            expected_symbol = f"{self._resolved(kernel).attr('callee')}.{group}.{axis}"
            config_symbol = self._resolved(definition).attr("config")
            if config_symbol != expected_symbol:
                raise CxxRenderError(
                    definition,
                    f"kernel {group} configuration references noncanonical "
                    f"symbol {config_symbol!r}",
                )
            declaration = self.config_symbols.get(str(config_symbol))
            if declaration is None or declaration.name != "config.decl":
                raise CxxRenderError(
                    definition,
                    f"kernel {group} configuration symbol {config_symbol!r} "
                    "is not a declaration",
                )
            if (
                len(declaration.results) != 1
                or repr(self.module.values[declaration.results[0]].type) != "index"
            ):
                raise CxxRenderError(
                    declaration,
                    "kernel launch configuration declaration is not an index",
                )
            self._validate_projected_attributes(declaration, ("symbol", "predicates"))
            predicates = declaration.attributes.get("predicates")
            if predicates is None:
                ranges.append(None)
            else:
                if (
                    not isinstance(predicates, PredicateListAttr)
                    or len(predicates) != 1
                ):
                    raise CxxRenderError(
                        declaration,
                        "kernel launch configuration declaration does not have "
                        "one source range",
                    )
                predicate = predicates[0]
                if (
                    predicate.kind != "range"
                    or len(predicate.args) != 3
                    or predicate.args[0].tag != "value"
                    or predicate.args[0].value != declaration.results[0]
                    or predicate.args[1].tag != "const"
                    or predicate.args[2].tag != "const"
                ):
                    raise CxxRenderError(
                        declaration,
                        "kernel launch configuration predicate is not its source range",
                    )
                lower = predicate.args[1].value
                upper = predicate.args[2].value
                if lower <= 0 or upper < lower or upper > 0x7FFFFFFF:
                    raise CxxRenderError(
                        declaration,
                        "kernel launch range is not representable by positive i32 "
                        "source bounds",
                    )
                ranges.append((lower, upper))
            declarations.append(declaration)

        if all(bounds is None for bounds in ranges):
            return _DimensionProjection(None, tuple(declarations))
        if any(bounds is None for bounds in ranges):
            raise CxxRenderError(
                launch,
                f"kernel {group} configuration mixes ranged and unconstrained "
                "dimensions",
            )
        arguments = [
            str(bound) for axis_range in ranges for bound in (axis_range or ())
        ]
        return _DimensionProjection(
            f"loom::{group}_range({', '.join(arguments)})", tuple(declarations)
        )

    def _source_launch_attributes(self, kernel: Operation) -> list[str] | None:
        resolved = self._resolved(kernel)
        config = resolved.region("config")
        if config is None or len(config.blocks) != 1:
            raise CxxRenderError(
                kernel, "kernel launch configuration must be one block"
            )
        block = config.blocks[0]
        if block.arg_ids:
            return None

        launch = block.ops[-1] if block.ops else None
        if launch is None or launch.name != "kernel.launch.config":
            raise CxxRenderError(kernel, "kernel has no launch configuration")
        if len(launch.operands) != 6:
            return None
        fields = self._resolved(launch)
        groups = [
            (
                "workgroup_count",
                [fields.value_id(f"workgroup_count_{axis}") for axis in "xyz"],
            ),
            (
                "workgroup_size",
                [fields.value_id(f"workgroup_size_{axis}") for axis in "xyz"],
            ),
        ]
        attributes = ["loom::kernel"]
        consumed_operations = {id(launch)}
        consumed_declarations: set[int] = set()
        for name, values in groups:
            projected = self._launch_dimension_attribute(kernel, launch, name, values)
            if projected is None:
                return None
            if projected.attribute is not None:
                attributes.append(projected.attribute)
            consumed_declarations.update(
                id(declaration) for declaration in projected.declarations
            )
            consumed_operations.update(
                id(self.definitions[value_id]) for value_id in values
            )
        if consumed_operations != {id(operation) for operation in block.ops}:
            return None
        consumed_symbols = {
            str(declaration.attributes["symbol"])
            for declaration in self.config_symbols.values()
            if id(declaration) in consumed_declarations
        }
        if any(
            operation.name == "config.get"
            and operation.attributes.get("config") in consumed_symbols
            and id(operation) not in consumed_operations
            for operation in walk_operations(self.module.body)
        ):
            return None
        self._validate_projected_attributes(launch, ())
        self.projected_config_declarations.update(consumed_declarations)
        return attributes

    def _plan_kernel_configurations(self, kernels: list[Operation]) -> None:
        for kernel in kernels:
            self.kernel_source_launch_attributes[id(kernel)] = (
                self._source_launch_attributes(kernel)
            )

    def _configuration_parameters(
        self, kernel: Operation, arguments: list[int]
    ) -> list[_ConfigurationParameter]:
        return [
            _ConfigurationParameter(parameter.value_id, len(parameter.dependencies) + 1)
            for parameter in self._flat_source_parameters(
                kernel, arguments, signature="kernel configuration"
            )
        ]

    def _configuration_parameter_type(self, value_id: int) -> str:
        value_type = self.module.values[value_id].type
        if isinstance(value_type, BufferType):
            return "loom::type::buffer<unsigned char>"
        if isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VIEW:
            return self.types.view(value_type, value_id in self.signed_values)
        return self.cpp_type(value_id)

    def _bind_configuration_parameters(
        self, kernel: Operation, block: Block
    ) -> list[str]:
        symbol = str(self._resolved(kernel).attr("callee"))
        parameters = self._configuration_parameters(kernel, list(block.arg_ids))
        declarations: list[str] = []
        for parameter in parameters:
            name = self.names[parameter.value_id]
            type_name = self._configuration_parameter_type(parameter.value_id)
            declarations.append(f"{type_name} {name}")
            self.expressions[parameter.value_id] = name
            self.expression_types[parameter.value_id] = type_name
        self.kernel_workload_component_counts[symbol] = tuple(
            parameter.component_count for parameter in parameters
        )
        return declarations

    def _configuration_result_expression(self, value_id: int) -> str:
        cast = self.definitions.get(value_id)
        if cast is not None and cast.name == "index.cast":
            extended = self.definitions.get(cast.operands[0])
            if extended is not None and extended.name == "scalar.extui":
                source = extended.operands[0]
                if repr(self.module.values[source].type) == "i32":
                    return self.expression(source)
        return f"loom::index::cast<unsigned>({self.expression(value_id)})"

    def _render_launch_configuration_return(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        count = [
            self._configuration_result_expression(
                resolved.value_id(f"workgroup_count_{axis}")
            )
            for axis in "xyz"
        ]
        size = [
            self._configuration_result_expression(
                resolved.value_id(f"workgroup_size_{axis}")
            )
            for axis in "xyz"
        ]
        cluster_ids = [
            resolved.value_ids(f"workgroup_cluster_size_{axis}") for axis in "xyz"
        ]
        groups = [count, size]
        if any(cluster_ids):
            if not all(len(values) == 1 for values in cluster_ids):
                raise CxxRenderError(
                    operation,
                    "kernel launch configuration has an incomplete cluster size",
                )
            groups.append(
                [
                    self._configuration_result_expression(values[0])
                    for values in cluster_ids
                ]
            )
        aggregate = ", ".join("{" + ", ".join(group) + "}" for group in groups)
        self.writer.line(f"return {{{aggregate}}};")

    def _render_kernel_configuration(self, kernel: Operation) -> list[str]:
        symbol = str(self._resolved(kernel).attr("callee"))
        source_attributes = self.kernel_source_launch_attributes.get(id(kernel))
        if id(kernel) not in self.kernel_source_launch_attributes:
            source_attributes = self._source_launch_attributes(kernel)
        if source_attributes is not None:
            self.kernel_workload_component_counts[symbol] = ()
            return source_attributes

        config = self._resolved(kernel).region("config")
        assert config is not None
        block = config.blocks[0]
        launch = block.ops[-1] if block.ops else None
        if launch is None or launch.name != "kernel.launch.config":
            raise CxxRenderError(kernel, "kernel has no launch configuration")
        clustered = len(launch.operands) == 9
        if len(launch.operands) not in {6, 9}:
            raise CxxRenderError(
                launch, "kernel launch configuration has incomplete geometry"
            )
        return_type = (
            "loom::kernel::clustered_configuration"
            if clustered
            else "loom::kernel::configuration"
        )
        parameters = self._bind_configuration_parameters(kernel, block)
        configuration_name = self.kernel_configuration_names[symbol]
        self.writer.open(
            f"static {return_type} {configuration_name}({', '.join(parameters)})"
        )
        self._render_block(block, launch_configuration=True)
        self.writer.close()
        self.writer.line()
        return [f"loom::kernel({configuration_name})"]
