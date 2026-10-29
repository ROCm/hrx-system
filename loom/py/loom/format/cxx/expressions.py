# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expression projection for readable Loom C++ rendering."""

from __future__ import annotations

import re
from typing import Any

from loom.format.cxx.context import (
    CxxRenderContext,
    _memory_space,
    _strip_outer_parentheses,
)
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.types import KERNEL_COORDINATE_CPP_TYPE
from loom.ir import (
    EncodingInstance,
    Operation,
    Predicate,
    PredicateArg,
    PredicateListAttr,
    ScalarType,
    ShapedType,
)

_ENCODING_SCHEMA_FIELDS = {
    "encoding.f4e2m1": (
        "payload_elements",
        "payload_registers",
        "payload_packing",
        "scale_format",
        "scale_group_elements",
        "scale_operands",
        "scale_topology",
        "affine",
        "zero_scale_fallback",
    ),
    "encoding.f8e4m3fn": (
        "payload_elements",
        "scale_format",
        "scale_group_elements",
        "scale_operands",
        "scale_topology",
        "affine",
    ),
}

_ENCODING_SCHEMA_DEFAULTS = {
    "encoding.f4e2m1": {
        "payload_elements": 32,
        "payload_packing": "little_endian_nibbles",
        "scale_format": "e8m0",
        "scale_group_elements": 32,
        "scale_operands": 1,
        "scale_topology": "block_1d",
        "affine": "scale_only",
        "zero_scale_fallback": True,
    },
    "encoding.f8e4m3fn": {
        "payload_elements": 32,
        "scale_format": "e8m0",
        "scale_group_elements": 32,
        "scale_operands": 1,
        "scale_topology": "block_1d",
        "affine": "scale_only",
    },
}

_ENCODING_ENUM_TYPES = {
    "payload_packing": "payload_packing",
    "scale_format": "numeric_format",
    "scale_topology": "scale_topology",
    "affine": "affine_policy",
}

_SCALAR_OPERATOR_OPERATIONS = frozenset(
    {"scalar.addf", "scalar.divf", "scalar.mulf", "scalar.subf"}
)
_SCALAR_TRIGONOMETRY_OPERATIONS = frozenset(
    {"scalar.cosf", "scalar.costurnsf", "scalar.sinf", "scalar.sinturnsf"}
)
_SCALAR_NAMED_FUNCTION_OPERATIONS = frozenset(
    {
        "scalar.absf",
        "scalar.acosf",
        "scalar.acoshf",
        "scalar.asinf",
        "scalar.asinhf",
        "scalar.atan2f",
        "scalar.atanf",
        "scalar.atanhf",
        "scalar.cbrtf",
        "scalar.ceilf",
        "scalar.copysignf",
        "scalar.coshf",
        "scalar.ctlzi",
        "scalar.ctpopi",
        "scalar.cttzi",
        "scalar.erfcf",
        "scalar.erff",
        "scalar.exp2f",
        "scalar.expf",
        "scalar.expm1f",
        "scalar.floorf",
        "scalar.fmaf",
        "scalar.ldexpf",
        "scalar.log10f",
        "scalar.log1pf",
        "scalar.log2f",
        "scalar.logf",
        "scalar.logisticf",
        "scalar.maximumf",
        "scalar.maxnumf",
        "scalar.minimumf",
        "scalar.minnumf",
        "scalar.negf",
        "scalar.powf",
        "scalar.remf",
        "scalar.roundevenf",
        "scalar.roundf",
        "scalar.rsqrtf",
        "scalar.signf",
        "scalar.siluf",
        "scalar.sinhf",
        "scalar.softplusf",
        "scalar.sqrtf",
        "scalar.tanf",
        "scalar.tanhf",
        "scalar.truncf",
    }
)

# The renderer must remain the inverse of every typed scalar operation exposed
# by loomcxx/scalar.h. The generator test compares this inventory against the
# declarations selected from the dialect so either side cannot grow alone.
SCALAR_PROJECTION_OPERATIONS = frozenset(
    {
        *_SCALAR_OPERATOR_OPERATIONS,
        *_SCALAR_TRIGONOMETRY_OPERATIONS,
        *_SCALAR_NAMED_FUNCTION_OPERATIONS,
        "scalar.frexpf",
    }
)


class CxxExpressionRenderer(CxxRenderContext):
    def _encoding_parameter_expression(
        self, operation: Operation, name: str, value: Any
    ) -> str:
        enum_type = _ENCODING_ENUM_TYPES.get(name)
        if enum_type is not None:
            if not isinstance(value, str):
                raise CxxRenderError(
                    operation,
                    f"encoding parameter {name!r} has no C++ enum spelling",
                )
            return f"loom::encoding::{enum_type}::{value}"
        if isinstance(value, bool):
            return "true" if value else "false"
        if type(value) is int:
            return str(value)
        raise CxxRenderError(
            operation,
            f"encoding parameter {name!r} has no C++ constant projection",
        )

    def _render_encoding_define(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        spec = resolved.attr("spec")
        if not isinstance(spec, EncodingInstance):
            raise CxxRenderError(
                operation, "encoding specification is not a static encoding"
            )
        result = resolved.value_id("result")
        if spec.name == "encoding.storage":
            names = resolved.attr("param_names")
            if not isinstance(names, dict) and not hasattr(names, "items"):
                raise CxxRenderError(
                    operation, "physical storage parameters are not named"
                )
            operands = resolved.value_ids("params")
            values = {name: operands[ordinal] for name, ordinal in names.items()}
            if set(values) != {"layout", "schema"}:
                raise CxxRenderError(
                    operation,
                    "physical storage requires layout and schema parameters",
                )
            expression = (
                "loom::encoding::define("
                f"{self.expression(values['layout'])}, "
                f"{self.expression(values['schema'])})"
            )
            self._declare(result, expression, declaration_type="auto")
            return ("spec", "param_names")

        parameters = dict(spec.params)
        schema_name = spec.name
        if schema_name == "encoding.operand":
            element_format = parameters.pop("element_format", None)
            if isinstance(element_format, str):
                schema_name = f"encoding.{element_format}"
        if (
            schema_name == "encoding.f8e4m3fn"
            and parameters.get("payload_packing") == "dense_lanes"
        ):
            parameters.pop("payload_packing")
        fields = _ENCODING_SCHEMA_FIELDS.get(schema_name)
        if fields is None:
            raise CxxRenderError(
                operation,
                f"encoding schema {schema_name!r} has no C++ facade",
            )
        unknown = sorted(set(parameters).difference(fields))
        if unknown:
            raise CxxRenderError(
                operation,
                f"encoding parameter {unknown[0]!r} has no C++ facade field",
            )
        defaults = dict(_ENCODING_SCHEMA_DEFAULTS[schema_name])
        if schema_name == "encoding.f4e2m1":
            payload_elements = parameters.get(
                "payload_elements", defaults["payload_elements"]
            )
            if type(payload_elements) is int:
                defaults["payload_registers"] = (payload_elements + 7) // 8
        initializers = [
            f".{name} = "
            f"{self._encoding_parameter_expression(operation, name, parameters[name])}"
            for name in fields
            if name in parameters and parameters[name] != defaults.get(name)
        ]
        family = schema_name.removeprefix("encoding.")
        aggregate = "{" + ", ".join(initializers) + "}"
        expression = f"loom::encoding::define<loom::encoding::{family}{aggregate}>()"
        self._declare(result, expression, declaration_type="auto")
        return ("spec", "param_names")

    def _record_type(self, stem: str, fields: tuple[tuple[str, str], ...]) -> str:
        key = (stem, fields)
        existing = self.record_types.get(key)
        if existing is not None:
            return existing
        name = stem
        ordinal = 2
        while name in self.record_declarations:
            name = f"{stem}{ordinal}"
            ordinal += 1
        self.record_types[key] = name
        self.record_declarations[name] = fields
        return name

    def _named_record_argument(
        self,
        operation: Operation,
        values: tuple[int, ...],
        names: object,
        stem: str,
    ) -> str:
        if not isinstance(names, dict) and not hasattr(names, "items"):
            raise CxxRenderError(operation, "named operands lack key metadata")
        ordered = sorted(names.items(), key=lambda item: item[1])
        if [ordinal for _name, ordinal in ordered] != list(range(len(values))):
            raise CxxRenderError(operation, "named operand ordinals are inconsistent")
        fields = tuple(
            (name, self.cpp_type(values[ordinal])) for name, ordinal in ordered
        )
        record_type = self._record_type(stem, fields)
        arguments = ", ".join(
            self.expression(values[ordinal]) for _name, ordinal in ordered
        )
        return f"{record_type}{{{arguments}}}"

    def _render_decode(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        result = resolved.value_id("result")
        arguments = [
            self.expression(resolved.value_id("payload")),
            self.expression(resolved.value_id("schema")),
        ]
        auxiliary = resolved.value_ids("auxiliary")
        if auxiliary:
            arguments.append(
                self._named_record_argument(
                    operation,
                    auxiliary,
                    resolved.attr("auxiliary_names"),
                    "DecodeAuxiliary",
                )
            )
        expression = (
            f"loom::vector::decode<{self.cpp_type(result)}>("
            + ", ".join(arguments)
            + ")"
        )
        self._declare(result, expression)
        return ("auxiliary_names",)

    def _fragment_shape(self, operation: Operation, resolved: Any) -> str:
        role = resolved.attr("role")
        if role not in {"lhs", "rhs", "init", "result"}:
            raise CxxRenderError(
                operation, f"fragment role {role!r} has no C++ spelling"
            )
        blocks = resolved.value_ids("blocks")
        shape_values = [
            *(self.expression(value) for value in blocks),
            self.expression(resolved.value_id("rows")),
            self.expression(resolved.value_id("columns")),
        ]
        shape_name = "batched_shape" if blocks else "shape"
        return f"loom::vector::fragment::{shape_name}{{" + ", ".join(shape_values) + "}"

    def _fragment_origin(self, resolved: Any) -> tuple[str, list[str]]:
        """Projects indexed-memory origin structure into a source specialization."""

        sentinel = -(2**63)
        static = resolved.attr("static_indices")
        dynamic = iter(resolved.value_ids("indices"))
        if static is None or all(value == sentinel for value in static):
            return "", [
                self._mixed_index_record(
                    resolved,
                    dynamic_field="indices",
                    static_field="static_indices",
                )
            ]

        pattern: list[str] = []
        runtime: list[str] = []
        for value in static:
            if value == sentinel:
                pattern.append("loom::type::dynamic")
                try:
                    runtime.append(self._address_operand(next(dynamic)))
                except StopIteration as error:
                    raise CxxRenderError(
                        None, "dynamic fragment origin metadata is incomplete"
                    ) from error
            else:
                pattern.append(str(value))
        try:
            next(dynamic)
        except StopIteration:
            pass
        else:
            raise CxxRenderError(
                None, "dynamic fragment origin metadata has trailing values"
            )
        template = ", " + ", ".join(pattern)
        arguments = (
            ["loom::type::indices{" + ", ".join(runtime) + "}"] if runtime else []
        )
        return template, arguments

    def _render_fragment_load(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        result = resolved.value_id("result")
        role = resolved.attr("role")
        shape = self._fragment_shape(operation, resolved)
        template, origin = self._fragment_origin(resolved)
        arguments = [self.expression(resolved.value_id("view")), *origin, shape]
        auxiliary = resolved.value_ids("auxiliary")
        if auxiliary:
            arguments.append(
                self._named_record_argument(
                    operation,
                    auxiliary,
                    resolved.attr("auxiliary_names"),
                    "FragmentAuxiliary",
                )
            )
        expression = (
            "loom::vector::fragment::load<"
            f"loom::vector::fragment::role::{role}, {self.cpp_type(result)}"
            f"{template}>(" + ", ".join(arguments) + ")"
        )
        self._declare(result, expression)
        return ("role", "static_indices", "auxiliary_names")

    def _render_fragment(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        role = resolved.attr("role")
        shape = self._fragment_shape(operation, resolved)
        arguments = [self.expression(resolved.value_id("data")), shape]
        parameters = resolved.value_ids("params")
        if parameters:
            arguments.append(
                self._named_record_argument(
                    operation,
                    parameters,
                    resolved.attr("param_names"),
                    "FragmentParameters",
                )
            )
        result = resolved.value_id("result")
        expression = (
            "loom::vector::fragment::attach<"
            f"loom::vector::fragment::role::{role}>(" + ", ".join(arguments) + ")"
        )
        self._declare(result, expression)
        return ("role", "param_names")

    def _render_mma(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        flags = resolved.attr("flags")
        if flags is None:
            function = "loom::vector::mma"
        elif flags == "saturate":
            function = "loom::vector::mma<loom::vector::mma_flags::saturate>"
        else:
            raise CxxRenderError(operation, f"MMA flags {flags!r} have no C++ spelling")
        arguments = ", ".join(
            self.expression(value)
            for value in (
                resolved.value_id("lhs"),
                resolved.value_id("rhs"),
                resolved.value_id("init"),
            )
        )
        self._declare(resolved.value_id("result"), f"{function}({arguments})")
        return ("flags",)

    def _temporary_name(self, stem: str) -> str:
        candidate = stem
        ordinal = 2
        while candidate in self.temporary_names:
            candidate = f"{stem}_{ordinal}"
            ordinal += 1
        self.temporary_names.add(candidate)
        return candidate

    def _address_operand(self, value_id: int) -> str:
        value_type = self.module.values[value_id].type
        if not isinstance(value_type, ScalarType):
            raise CxxRenderError(None, "address operand is not a scalar value")
        domain = repr(value_type)
        if domain not in {"index", "offset"}:
            raise CxxRenderError(None, f"value type {domain} is not an address domain")
        expression = self.expression(value_id)
        expression_type = self.expression_type(value_id)
        if expression_type == self.cpp_type(value_id):
            return expression
        if domain == "index" and expression_type == KERNEL_COORDINATE_CPP_TYPE:
            return f"static_cast<loom::type::index>({expression})"
        return f"loom::{domain}::cast({expression})"

    def _fixed_integer_index_cast_operand(self, value_id: int) -> str:
        """Projects index.cast's signed fixed-integer interpretation."""

        expression = self.expression(value_id)
        value_type = self.module.values[value_id].type
        if not isinstance(value_type, ScalarType) or repr(value_type) not in {
            "i8",
            "i16",
            "i32",
            "i64",
        }:
            return expression
        signed_type = self.cpp_type(value_id, signed=True)
        unsigned_type = self.cpp_type(value_id, signed=False)
        if (
            signed_type != unsigned_type
            and self.expression_type(value_id) == unsigned_type
        ):
            return f"static_cast<{signed_type}>({expression})"
        return expression

    def _retains_address_identity(self, value_id: int) -> bool:
        expected_type = self.cpp_type(value_id)
        expression_type = self.expression_type(value_id)
        return expression_type == expected_type or (
            expected_type == "loom::type::index"
            and expression_type == KERNEL_COORDINATE_CPP_TYPE
        )

    def _address_cast_target(self, value_id: int) -> str | None:
        """Returns the sole address domain reached through transparent uses."""

        users = self.users[value_id]
        if len(users) != 1:
            return None
        user = users[0]
        if user.name == "index.cast":
            if len(user.results) != 1:
                return None
            domain = repr(self.module.values[user.results[0]].type)
            return domain if domain in {"index", "offset"} else None
        if user.name not in {"scalar.assume", "index.assume"} or not (
            self._assumption_is_tautological(user)
        ):
            return None
        resolved = self._resolved(user)
        for source, result in zip(
            resolved.value_ids("values"), resolved.value_ids("results"), strict=True
        ):
            if source == value_id:
                return self._address_cast_target(result)
        return None

    def _numeric_conversion(self, operation: Operation) -> tuple[str, ...]:
        """Projects one exact scalar or native-vector numeric conversion."""

        self.uses_numeric_conversion = True
        member = operation.name.split(".")[-1]
        source = operation.operands[0]
        result = operation.results[0]
        source_expression = self.expression(source)
        source_signedness = {
            "extsi": True,
            "sitofp": True,
            "extui": False,
            "uitofp": False,
        }.get(member)
        if source_signedness is not None:
            source_type = self.cpp_type(source, signed=source_signedness)
            if self.expression_type(source) != source_type:
                source_expression = f"loom::convert<{source_type}>({source_expression})"

        result_signedness = {
            "extsi": True,
            "fptosi": True,
            "extui": False,
            "fptoui": False,
        }.get(member)
        result_type = self.cpp_type(result, signed=result_signedness)
        subnormal = operation.attributes.get("subnormal")
        policy = ""
        if subnormal is not None:
            spelling = {"daz": "daz", "ftz": "ftz", "daz|ftz": "daz_ftz"}.get(subnormal)
            if spelling is None:
                raise CxxRenderError(
                    operation,
                    f"subnormal policy {subnormal!r} has no C++ spelling",
                )
            policy = f", loom::conversion::subnormal::{spelling}"
        self._declare(
            result,
            f"loom::convert<{result_type}{policy}>({source_expression})",
            declaration_type=result_type,
            expression_type=result_type,
            inlineable=True,
        )
        return ("subnormal",) if subnormal is not None else ()

    def _loop_initializer(self, value_id: int) -> str:
        definition = self.definitions.get(value_id)
        if definition is not None and definition.name in {
            "vector.splati",
            "vector.splatf",
        }:
            value = definition.attributes.get("value")
            value_type = self.module.values[value_id].type
            if (
                (value == 0 or value == 0.0)
                and isinstance(value_type, ShapedType)
                and value_type.rank == 1
            ):
                return f"{self.cpp_type(value_id)}{{}}"
        if definition is not None and definition.name in {
            "scalar.constant",
            "index.constant",
        }:
            return self._literal(definition.attributes.get("value"), value_id)
        return self.expression(value_id)

    def _counted_integer_bound(
        self,
        value_id: int,
        *,
        carrier_type: str,
        minimum: int,
        maximum: int,
        suffix: str = "",
    ) -> str | None:
        """Recovers one fixed-width C++ loop bound from address-domain IR."""

        constant = self._constant_integer(value_id)
        if constant is not None:
            if minimum <= constant <= maximum:
                return f"{constant}{suffix}"
            return None
        definition = self.definitions.get(value_id)
        if (
            definition is None
            or definition.name != "index.cast"
            or len(definition.operands) != 1
        ):
            return None
        source = definition.operands[0]
        expression = self.expression(source)
        if (
            self.expression_type(source) == carrier_type
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression) is not None
        ):
            return expression
        return None

    def _signed_counted_bound(self, value_id: int) -> str | None:
        """Recovers one C++ int loop-bound expression from index IR."""

        return self._counted_integer_bound(
            value_id,
            carrier_type="int",
            minimum=-(1 << 31),
            maximum=(1 << 31) - 1,
        )

    def _unsigned_counted_bound(self, value_id: int) -> str | None:
        """Recovers one C++ unsigned loop-bound expression from offset IR."""

        return self._counted_integer_bound(
            value_id,
            carrier_type="unsigned",
            minimum=0,
            maximum=(1 << 32) - 1,
            suffix="u",
        )

    def _predicate_arg(
        self, argument: PredicateArg, *, reference_id: int | None = None
    ) -> str:
        if argument.tag == "const":
            return self._predicate_constant(argument.value, reference_id)
        value_type = self.module.values[argument.value].type
        if isinstance(value_type, ScalarType) and repr(value_type) in {
            "index",
            "offset",
        }:
            if not self._retains_address_identity(argument.value):
                raise CxxRenderError(
                    None,
                    "address predicate requires a retained index or offset binding",
                )
            return self.expression(argument.value)
        return self.expression(argument.value)

    def _predicate_key(self, predicate: Predicate) -> tuple[object, ...]:
        arguments: list[tuple[str, int]] = []
        for argument in predicate.args:
            value = (
                self.value_origins[argument.value]
                if argument.tag == "value"
                else argument.value
            )
            arguments.append((argument.tag, value))
        return (predicate.kind, *arguments)

    def _predicate(self, predicate: Predicate) -> str | None:
        reference_id = next(
            (argument.value for argument in predicate.args if argument.tag == "value"),
            None,
        )
        arguments = [
            self._predicate_arg(argument, reference_id=reference_id)
            for argument in predicate.args
        ]
        if reference_id is not None:
            reference_type = self.module.values[reference_id].type
            if isinstance(reference_type, ScalarType) and repr(reference_type) in {
                "index",
                "offset",
            }:
                joined = ", ".join(arguments)
                return f"loom::predicate::{predicate.kind}({joined})"
        binary = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
            "ult": "<",
            "ule": "<=",
            "ugt": ">",
            "uge": ">=",
        }
        if predicate.kind in binary:
            return f"({arguments[0]} {binary[predicate.kind]} {arguments[1]})"
        if predicate.kind == "range":
            subject = predicate.args[0]
            lower = predicate.args[1]
            if (
                subject.tag == "value"
                and lower.tag == "const"
                and lower.value == 0
                and self._is_unsigned_scalar(subject.value)
            ):
                if self._predicate_is_tautological(predicate):
                    return None
                return f"({arguments[0]} <= {arguments[2]})"
            return (
                f"({arguments[0]} >= {arguments[1]} && "
                f"{arguments[0]} <= {arguments[2]})"
            )
        if predicate.kind == "multiple_of":
            return f"({arguments[0]} % {arguments[1]} == 0)"
        if predicate.kind == "power_of_two":
            return (
                f"({arguments[0]} > 0 && ({arguments[0]} & ({arguments[0]} - 1)) == 0)"
            )
        raise CxxRenderError(None, f"predicate {predicate.kind} has no C++ projection")

    def _render_assume(self, operation: Operation) -> None:
        resolved = self._resolved(operation)
        predicates = resolved.attr("predicates")
        if not isinstance(predicates, PredicateListAttr):
            raise CxxRenderError(operation, "assumption lacks predicate-list metadata")
        results = resolved.value_ids("results")
        block = self.definition_blocks[results[0]]
        # Equivalent assumptions share one statement only when their source
        # binding refinements inhabit the same lexical block.
        rendered_assumptions = self.rendered_assumptions_by_block.setdefault(
            id(block), set()
        )
        conditions: list[str] = []
        for predicate in predicates:
            key = self._predicate_key(predicate)
            if key in rendered_assumptions:
                continue
            condition = self._predicate(predicate)
            rendered_assumptions.add(key)
            if condition is not None:
                conditions.append(condition)
        if conditions:
            condition = " && ".join(conditions)
            self.writer.line(f"loom::assume({_strip_outer_parentheses(condition)});")
        for source, result in zip(resolved.value_ids("values"), results, strict=True):
            self.expressions[result] = self.expression(source)
            self.expression_types[result] = self.expression_type(source)
            self.value_origins[result] = self.value_origins[source]

    def _binary_expression(self, operation: Operation, token: str) -> str:
        lhs = self.expression(operation.operands[0])
        rhs = self.expression(operation.operands[1])
        return f"({lhs} {token} {rhs})"

    def _mixed_indices(
        self, resolved: Any, *, dynamic_field: str, static_field: str
    ) -> list[str]:
        dynamic = iter(resolved.value_ids(dynamic_field))
        static = resolved.attr(static_field)
        if static is None:
            return [self.expression(value) for value in dynamic]
        result: list[str] = []
        for value in static:
            if value == -(2**63):
                try:
                    result.append(self.expression(next(dynamic)))
                except StopIteration as error:
                    raise CxxRenderError(
                        None, "dynamic index metadata is incomplete"
                    ) from error
            else:
                result.append(str(value))
        try:
            next(dynamic)
        except StopIteration:
            return result
        raise CxxRenderError(None, "dynamic index metadata has trailing values")

    def _mixed_index_record(
        self, resolved: Any, *, dynamic_field: str, static_field: str
    ) -> str:
        """Renders a coordinate record without discarding index identities."""

        dynamic = iter(resolved.value_ids(dynamic_field))
        static = resolved.attr(static_field)
        entries: list[tuple[str, int | None]] = []
        for value in static or ():
            if value == -(2**63):
                try:
                    value_id = next(dynamic)
                except StopIteration as error:
                    raise CxxRenderError(
                        None, "dynamic index metadata is incomplete"
                    ) from error
                entries.append((self.expression(value_id), value_id))
            else:
                entries.append((str(value), None))
        if static is None:
            entries.extend(
                (self.expression(value_id), value_id) for value_id in dynamic
            )
            dynamic = iter(())
        try:
            next(dynamic)
        except StopIteration:
            pass
        else:
            raise CxxRenderError(None, "dynamic index metadata has trailing values")

        strong = any(
            value_id is not None and self._retains_address_identity(value_id)
            for _, value_id in entries
        )
        if not strong:
            return "{" + ", ".join(expression for expression, _ in entries) + "}"
        elements = [
            f"loom::type::index{{{expression}}}"
            if value_id is None
            else self._address_operand(value_id)
            for expression, value_id in entries
        ]
        return "loom::type::indices{" + ", ".join(elements) + "}"

    def _render_index_expression(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if not name.startswith("index."):
            return None
        if name == "index.cast":
            result = operation.results[0]
            operand = operation.operands[0]
            operand_expression = self.expression(operand)
            constant = self._constant_integer(operand)
            if constant is not None and repr(self.module.values[result].type) in {
                "index",
                "offset",
            }:
                self.expressions[result] = self._literal(constant, result)
                self.expression_types[result] = self.cpp_type(result)
                return ()
            if repr(self.module.values[result].type) == "index":
                operand_expression = self._fixed_integer_index_cast_operand(operand)
            if self.cpp_type(result) == self.expression_type(operand):
                expression = operand_expression
            elif self.expression_type(operand) == KERNEL_COORDINATE_CPP_TYPE:
                if repr(self.module.values[result].type) in {"index", "offset"}:
                    expression = self._cast(result, self._address_operand(operand))
                elif self.cpp_type(result) == "unsigned":
                    expression = f"static_cast<unsigned>({operand_expression})"
                else:
                    expression = (
                        f"loom::index::cast<{self.cpp_type(result)}>"
                        f"({self._address_operand(operand)})"
                    )
            elif repr(self.module.values[result].type) not in {
                "index",
                "offset",
            } and repr(self.module.values[operand].type) in {"index", "offset"}:
                domain = repr(self.module.values[operand].type)
                expression = (
                    f"loom::{domain}::cast<{self.cpp_type(result)}>"
                    f"({self._address_operand(operand)})"
                )
            else:
                expression = self._cast(result, operand_expression)
            self._declare(result, expression, inlineable=True)
            return ()

        binary = {
            "index.add": "+",
            "index.sub": "-",
            "index.mul": "*",
            "index.div": "/",
            "index.rem": "%",
            "index.andi": "&",
            "index.ori": "|",
            "index.xori": "^",
            "index.shli": "<<",
            "index.shrsi": ">>",
        }
        if name in binary:
            lhs, rhs = (self._address_operand(value) for value in operation.operands)
            self._declare(
                operation.results[0],
                f"({lhs} {binary[name]} {rhs})",
                inlineable=True,
            )
            return ()
        if name == "index.scale":
            index, stride = (
                self._address_operand(value) for value in operation.operands
            )
            self._declare(
                operation.results[0],
                f"loom::index::scale({index}, {stride})",
                inlineable=True,
            )
            return ()
        named_binary = {
            "index.min": "min",
            "index.max": "max",
            "index.shrui": "shrui",
            "index.rotli": "rotli",
            "index.rotri": "rotri",
        }
        if name in named_binary:
            arguments = ", ".join(
                self._address_operand(value) for value in operation.operands
            )
            self._declare(
                operation.results[0],
                f"loom::index::{named_binary[name]}({arguments})",
                inlineable=True,
            )
            return ()
        named_unary = {
            "index.ctlzi": "ctlzi",
            "index.cttzi": "cttzi",
            "index.ctpopi": "ctpopi",
        }
        if name in named_unary:
            self._declare(
                operation.results[0],
                f"loom::index::{named_unary[name]}("
                f"{self._address_operand(operation.operands[0])})",
                inlineable=True,
            )
            return ()
        if name == "index.madd":
            arguments = ", ".join(
                self._address_operand(value) for value in operation.operands
            )
            self._declare(
                operation.results[0],
                f"loom::index::madd({arguments})",
                inlineable=True,
            )
            return ()
        if name == "index.divrem":
            quotient, remainder = operation.results
            arguments = ", ".join(
                self._address_operand(value) for value in operation.operands
            )
            temporary = self._temporary_name("quotient_remainder")
            self.writer.line(f"auto {temporary} = loom::index::divrem({arguments});")
            self.expressions[quotient] = f"{temporary}.quotient"
            self.expressions[remainder] = f"{temporary}.remainder"
            self.expression_types[quotient] = self.cpp_type(quotient)
            self.expression_types[remainder] = self.cpp_type(remainder)
            return ()
        if name == "index.cmp":
            predicate = operation.attributes.get("predicate")
            lhs, rhs = (self._address_operand(value) for value in operation.operands)
            domain = repr(self.module.values[operation.operands[0]].type)
            natural = {
                "index": {
                    "eq": "==",
                    "ne": "!=",
                    "slt": "<",
                    "sle": "<=",
                    "sgt": ">",
                    "sge": ">=",
                },
                "offset": {
                    "eq": "==",
                    "ne": "!=",
                    "ult": "<",
                    "ule": "<=",
                    "ugt": ">",
                    "uge": ">=",
                },
            }[domain]
            token = natural.get(predicate)
            if token is not None:
                expression = f"({lhs} {token} {rhs})"
            else:
                namespace = "index" if domain == "index" else "offset"
                expression = f"loom::{namespace}::{predicate}({lhs}, {rhs})"
            self._declare(operation.results[0], expression, inlineable=True)
            return ("predicate",)
        if name == "index.constexpr":
            result = operation.results[0]
            result_name = self.names[result]
            result_type = self.cpp_type(result)
            self.writer.line(
                f"[[loom::constexpr]] const {result_type} {result_name} = "
                f"{self._address_operand(operation.operands[0])};"
            )
            self.expressions[result] = result_name
            self.expression_types[result] = result_type
            return ()
        raise CxxRenderError(operation, "address operation has no C++ projection")

    def _render_comparison(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name == "vector.cmpi":
            predicate = operation.attributes.get("predicate")
            if predicate not in {
                "eq",
                "ne",
                "slt",
                "sle",
                "sgt",
                "sge",
                "ult",
                "ule",
                "ugt",
                "uge",
            }:
                raise CxxRenderError(
                    operation, f"comparison predicate {predicate!r} is unsupported"
                )
            result = operation.results[0]
            lhs, rhs = (self.expression(operand) for operand in operation.operands)
            self._declare(
                result,
                f"loom::vector::cmpi::{predicate}<{self.cpp_type(result)}>"
                f"({lhs}, {rhs})",
                inlineable=True,
            )
            return ("predicate",)
        if name == "vector.cmpf":
            predicate = operation.attributes.get("predicate")
            if predicate not in {
                "oeq",
                "ogt",
                "oge",
                "olt",
                "ole",
                "one",
                "ord",
                "ueq",
                "ugt",
                "uge",
                "ult",
                "ule",
                "une",
                "uno",
            }:
                raise CxxRenderError(
                    operation,
                    f"floating comparison predicate {predicate!r} is unsupported",
                )
            result = operation.results[0]
            lhs, rhs = (self.expression(operand) for operand in operation.operands)
            self._declare(
                result,
                f"loom::vector::cmpf::{predicate}<{self.cpp_type(result)}>"
                f"({lhs}, {rhs})",
                inlineable=True,
                statement_attribute=self._fastmath_contract(operation),
            )
            return (
                ("predicate", "fastmath")
                if "fastmath" in operation.attributes
                else ("predicate",)
            )
        if name == "scalar.cmpi":
            predicate = operation.attributes.get("predicate")
            token = {
                "eq": "==",
                "ne": "!=",
                "slt": "<",
                "sle": "<=",
                "sgt": ">",
                "sge": ">=",
                "ult": "<",
                "ule": "<=",
                "ugt": ">",
                "uge": ">=",
            }.get(predicate)
            if token is None:
                raise CxxRenderError(
                    operation, f"comparison predicate {predicate!r} is unsupported"
                )
            statement_attribute = None
        elif name == "scalar.cmpf":
            predicate = operation.attributes.get("predicate")
            # These are exactly the predicates produced by native C++ binary
            # comparisons. In particular, C++ != is unordered while == and
            # the relational operators are ordered in the presence of NaNs.
            token = {
                "oeq": "==",
                "ogt": ">",
                "oge": ">=",
                "olt": "<",
                "ole": "<=",
                "une": "!=",
            }.get(predicate)
            if token is None:
                raise CxxRenderError(
                    operation,
                    f"floating comparison predicate {predicate!r} has no "
                    "native C++ binary operator",
                )
            statement_attribute = self._fastmath_contract(operation)
        else:
            return None
        self._declare(
            operation.results[0],
            self._binary_expression(operation, token),
            inlineable=True,
            statement_attribute=statement_attribute,
        )
        return (
            ("predicate", "fastmath")
            if "fastmath" in operation.attributes
            else ("predicate",)
        )

    def _render_buffer_assumption(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        if id(operation) in self.scoped_alignment_operations:
            alignment = resolved.attr("minimum_alignment")
            for source, result in zip(
                resolved.value_ids("buffers"),
                resolved.value_ids("results"),
                strict=True,
            ):
                self._declare(
                    result,
                    "loom::buffer::assume::alignment<"
                    f"{alignment}>({self.expression(source)})",
                )
                self.expression_types[result] = self.expression_type(source)
            return ("minimum_alignment",)
        if id(operation) in self.scoped_noalias_operations:
            for source, result in zip(
                resolved.value_ids("buffers"),
                resolved.value_ids("results"),
                strict=True,
            ):
                self._declare(
                    result,
                    f"loom::buffer::assume::noalias({self.expression(source)})",
                )
                self.expression_types[result] = self.expression_type(source)
            return ()
        if operation.name == "buffer.assume.memory_space":
            memory_space = resolved.attr("memory_space")
            source = resolved.value_id("buffer")
            result = resolved.value_id("result")
            self._declare(
                result,
                "loom::buffer::assume::memory_space<"
                f"{_memory_space(memory_space)}>({self.expression(source)})",
            )
            self.expression_types[result] = self.expression_type(source)
            return ("memory_space",)
        for source, result in zip(
            resolved.value_ids("buffers"),
            resolved.value_ids("results"),
            strict=True,
        ):
            self.expressions[result] = self.expression(source)
            self.expression_types[result] = self.expression_type(source)
        return (
            ("minimum_alignment",)
            if operation.name == "buffer.assume.alignment"
            else ()
        )

    def _render_vector_slice(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        offsets = self._mixed_indices(
            resolved,
            dynamic_field="offsets",
            static_field="static_offsets",
        )
        if len(offsets) != 1:
            raise CxxRenderError(
                operation,
                "native C++ vector slicing requires one lane offset",
            )
        static_offsets = resolved.attr("static_offsets")
        offset = offsets[0]
        if static_offsets and static_offsets[0] != -(2**63):
            offset = f"loom::type::index{{{offset}}}"
        result = resolved.value_id("result")
        source = self.expression(resolved.value_id("source"))
        self._declare(
            result,
            f"loom::vector::slice<{self.cpp_type(result)}>({source}, {offset})",
        )
        return ("static_offsets",)

    def _render_vector_insert(self, operation: Operation) -> tuple[str, ...]:
        resolved = self._resolved(operation)
        indices = self._mixed_indices(
            resolved,
            dynamic_field="indices",
            static_field="static_indices",
        )
        if len(indices) != 1:
            raise CxxRenderError(
                operation,
                "native vector insertion requires one lane index",
            )

        result = resolved.value_id("result")
        source = self.expression(resolved.value_id("dest"))
        assignment_target_id = self.assignment_targets.get(result)
        assignment_target = (
            None
            if assignment_target_id is None
            else self.expression(assignment_target_id)
        )
        if assignment_target is None:
            self._declare(result, source)
            mutation_target = self.expression(result)
        elif assignment_target == source:
            mutation_target = assignment_target
        else:
            mutation_target = self._temporary_name(f"next_{self.names[result]}")
            self.writer.line(
                f"auto {mutation_target} = {_strip_outer_parentheses(source)};"
            )
        self.writer.line(
            f"{mutation_target}[{indices[0]}] = "
            f"{self.expression(resolved.value_id('value'))};"
        )
        if assignment_target_id is not None:
            target_expression = self.expression(assignment_target_id)
            if mutation_target != target_expression:
                self.writer.line(f"{target_expression} = {mutation_target};")
            self.expressions[result] = target_expression
            self.expression_types[result] = self.expression_type(assignment_target_id)
        return ("static_indices",)

    def _render_scalar_conversion(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name not in {
            "scalar.extui",
            "scalar.extsi",
            "scalar.trunci",
            "scalar.extf",
            "scalar.fptrunc",
            "scalar.sitofp",
            "scalar.uitofp",
            "scalar.fptosi",
            "scalar.fptoui",
        }:
            return None
        result = operation.results[0]
        address_target = self._address_cast_target(result)
        if name == "scalar.extui" and address_target is not None:
            source = operation.operands[0]
            unsigned_source_type = self.cpp_type(source, signed=False)
            if self.expression_type(source) == unsigned_source_type:
                self.expressions[result] = self.expression(source)
                self.expression_types[result] = self.expression_type(source)
                return ()
        if name == "scalar.extsi" and address_target == "index":
            source = operation.operands[0]
            signed_source_type = self.cpp_type(source, signed=True)
            if self.expression_type(source) == signed_source_type:
                self.expressions[result] = self.expression(source)
                self.expression_types[result] = self.expression_type(source)
                return ()
        return self._numeric_conversion(operation)

    def _render_scalar_intrinsic(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name not in SCALAR_PROJECTION_OPERATIONS:
            return None
        if name in _SCALAR_OPERATOR_OPERATIONS:
            return None

        self.uses_scalar_functions = True
        member = name.removeprefix("scalar.")
        arguments = ", ".join(
            self.expression(operand) for operand in operation.operands
        )
        if name == "scalar.frexpf":
            fraction, exponent = operation.results
            temporary = self._temporary_name("decomposition")
            self.writer.line(f"auto {temporary} = loom::scalar::frexpf({arguments});")
            self.expressions[fraction] = f"{temporary}.fraction"
            self.expressions[exponent] = f"{temporary}.exponent"
            self.expression_types[fraction] = self.cpp_type(fraction)
            self.expression_types[exponent] = self.cpp_type(exponent)
            return ()

        if name in _SCALAR_TRIGONOMETRY_OPERATIONS:
            value = operation.attributes.get("math")
            flags = value.split("|") if isinstance(value, str) else []
            approximate = "afn" in flags or "fast" in flags
            preserve = "preserve" in flags
            if preserve and not approximate:
                raise CxxRenderError(
                    operation,
                    "preserved trigonometry without AFN has no public C++ spelling",
                )
            namespace = "loom::scalar"
            if preserve:
                namespace += "::approximate::preserved"
            elif approximate:
                namespace += "::approximate"
            fastmath = self._fastmath_contract(
                operation,
                attribute_name="math",
                exclude=frozenset({"afn", "preserve"}),
            )
            self._declare(
                operation.results[0],
                f"{namespace}::{member}({arguments})",
                inlineable=True,
                statement_attribute=fastmath,
            )
            return ("math",) if value is not None else ()

        value = operation.attributes.get("fastmath")
        flags = value.split("|") if isinstance(value, str) else []
        namespace = (
            "loom::scalar::approximate"
            if "afn" in flags or "fast" in flags
            else "loom::scalar"
        )
        fastmath = self._fastmath_contract(operation, exclude=frozenset({"afn"}))
        self._declare(
            operation.results[0],
            f"{namespace}::{member}({arguments})",
            inlineable=True,
            statement_attribute=fastmath,
        )
        return ("fastmath",) if "fastmath" in operation.attributes else ()

    def _render_structured_vector(self, operation: Operation) -> tuple[str, ...] | None:
        """Projects vector construction, reshaping, and lane selection."""

        name = operation.name
        if name in {"vector.splati", "vector.splatf"}:
            result = operation.results[0]
            value = operation.attributes["value"]
            type_name = self.cpp_type(result)
            value_type = self.module.values[result].type
            if isinstance(value_type, ShapedType) and (
                value_type.rank > 1 or repr(value_type.element_type) == "i1"
            ):
                element = self.types.scalar(
                    value_type.element_type, result in self.signed_values
                )
                scalar = (
                    f"{element}{{}}"
                    if value == 0 or value == 0.0
                    else self._literal(value, result)
                )
                expression = f"loom::vector::splat<{type_name}>({{}}, {scalar})"
            else:
                expression = (
                    f"{type_name}{{}}"
                    if value == 0 or value == 0.0
                    else self._literal(value, result)
                )
            self._declare(result, expression)
            return ("value",)
        if name == "vector.from_elements":
            result = operation.results[0]
            elements = ", ".join(
                self.expression(operand) for operand in operation.operands
            )
            self._declare(result, f"{self.cpp_type(result)}{{{elements}}}")
            return ()
        if name == "vector.extract":
            resolved = self._resolved(operation)
            indices = self._mixed_indices(
                resolved, dynamic_field="indices", static_field="static_indices"
            )
            source_id = resolved.value_id("source")
            source_type = self.module.values[source_id].type
            if isinstance(source_type, ShapedType) and source_type.rank > 1:
                static_indices = resolved.attr("static_indices")
                source = self.expression(source_id)
                result = operation.results[0]
                if static_indices and all(
                    index != -(2**63) for index in static_indices
                ):
                    selectors = ", ".join(str(index) for index in static_indices)
                    expression = (
                        f"loom::vector::extract<{self.cpp_type(result)}, "
                        f"{selectors}>({source})"
                    )
                elif not static_indices or all(
                    index == -(2**63) for index in static_indices
                ):
                    arguments = ", ".join((source, *indices))
                    expression = (
                        f"loom::vector::extract<{self.cpp_type(result)}>({arguments})"
                    )
                else:
                    raise CxxRenderError(
                        operation,
                        "mixed static and dynamic structured-vector extraction "
                        "has no C++ projection",
                    )
                self._declare(result, expression, inlineable=True)
                return ("static_indices",)
            if len(indices) != 1:
                raise CxxRenderError(
                    operation, "native vector extraction requires one lane index"
                )
            self._declare(
                operation.results[0],
                f"{self.expression(operation.operands[0])}[{indices[0]}]",
                inlineable=True,
            )
            return ("static_indices",)
        if name == "vector.insert":
            return self._render_vector_insert(operation)
        if name == "vector.splat":
            result = operation.results[0]
            value_type = self.module.values[result].type
            expression = self.expression(operation.operands[0])
            if isinstance(value_type, ShapedType) and (
                value_type.rank > 1 or repr(value_type.element_type) == "i1"
            ):
                expression = (
                    f"loom::vector::splat<{self.cpp_type(result)}>({{}}, {expression})"
                )
            self._declare(result, expression)
            return ()
        if name == "vector.bitcast":
            result = operation.results[0]
            source_id = operation.operands[0]
            source = self.expression(source_id)
            source_type = self.module.values[source_id].type
            result_type = self.module.values[result].type
            structured = any(
                isinstance(value_type, ShapedType)
                and (value_type.rank > 1 or repr(value_type.element_type) == "i1")
                for value_type in (source_type, result_type)
            )
            self._declare(
                result,
                f"loom::vector::bitcast<{self.cpp_type(result)}>({source})"
                if structured
                else f"__builtin_bit_cast({self.cpp_type(result)}, {source})",
                inlineable=True,
            )
            return ()
        if name == "vector.iota":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            result_type = self.module.values[result].type
            assert isinstance(result_type, ShapedType)
            element = self.types.scalar(
                result_type.element_type, result in self.signed_values
            )

            def scalar_argument(value_id: int) -> str:
                expression = self.expression(value_id)
                if self.expression_type(value_id) != element:
                    return f"static_cast<{element}>({expression})"
                return expression

            base = scalar_argument(resolved.value_id("base"))
            step = scalar_argument(resolved.value_id("step"))
            self._declare(
                result,
                f"loom::vector::iota<{self.cpp_type(result)}>({base}, {step})",
                inlineable=True,
            )
            return ()
        if name == "vector.concat":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            arguments = ", ".join(
                self.expression(value) for value in resolved.value_ids("inputs")
            )
            self._declare(
                result,
                f"loom::vector::concat<{resolved.attr('axis')}, "
                f"{self.cpp_type(result)}>({arguments})",
                inlineable=True,
            )
            return ("axis",)
        if name == "vector.transpose":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            permutation = ", ".join(str(axis) for axis in resolved.attr("permutation"))
            source = self.expression(resolved.value_id("source"))
            self._declare(
                result,
                f"loom::vector::transpose<{self.cpp_type(result)}, "
                f"{permutation}>({source})",
                inlineable=True,
            )
            return ("permutation",)
        if name == "vector.interleave":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            even = self.expression(resolved.value_id("even"))
            odd = self.expression(resolved.value_id("odd"))
            self._declare(
                result,
                f"loom::vector::interleave<{resolved.attr('axis')}, "
                f"{self.cpp_type(result)}>({even}, {odd})",
                inlineable=True,
            )
            return ("axis",)
        if name == "vector.select":
            resolved = self._resolved(operation)
            result = resolved.value_id("result")
            arguments = ", ".join(
                self.expression(resolved.value_id(field))
                for field in ("condition", "true_value", "false_value")
            )
            self._declare(
                result,
                f"loom::vector::select({arguments})",
                inlineable=True,
            )
            return ()
        return None

    def _render_simple(self, operation: Operation) -> tuple[str, ...] | None:
        name = operation.name
        if name in {"scalar.constant", "index.constant"}:
            result = operation.results[0]
            self.expressions[result] = self._literal(
                operation.attributes["value"], result
            )
            return ("value",)
        if name == "scf.select":
            resolved = self._resolved(operation)
            condition = _strip_outer_parentheses(
                self.expression(resolved.value_id("condition"))
            )
            expression = (
                f"({condition} ? "
                f"{self.expression(resolved.value_id('true_value'))} : "
                f"{self.expression(resolved.value_id('false_value'))})"
            )
            self._declare(resolved.value_id("result"), expression, inlineable=True)
            return ()
        if name in {"scalar.assume", "index.assume"}:
            self._render_assume(operation)
            return ("predicates",)
        if name == "scalar.constexpr":
            result = operation.results[0]
            result_name = self.names[result]
            result_type = self.cpp_type(result)
            self.writer.line(
                f"[[loom::constexpr]] const {result_type} {result_name} = "
                f"{self.expression(operation.operands[0])};"
            )
            self.expressions[result] = result_name
            self.expression_types[result] = result_type
            return ()
        if name in {
            "buffer.assume.alignment",
            "buffer.assume.memory_space",
            "buffer.assume.noalias",
        }:
            return self._render_buffer_assumption(operation)
        if name == "encoding.define":
            return self._render_encoding_define(operation)
        if name == "vector.decode":
            return self._render_decode(operation)
        if name == "vector.fragment":
            return self._render_fragment(operation)
        if name == "vector.fragment.load":
            return self._render_fragment_load(operation)
        if name == "vector.mma":
            return self._render_mma(operation)
        if name == "vector.slice":
            return self._render_vector_slice(operation)

        scalar_binary = {
            "scalar.addi": "+",
            "scalar.subi": "-",
            "scalar.muli": "*",
            "scalar.divsi": "/",
            "scalar.divui": "/",
            "scalar.remsi": "%",
            "scalar.remui": "%",
            "scalar.andi": "&",
            "scalar.ori": "|",
            "scalar.xori": "^",
            "scalar.shli": "<<",
            "scalar.shrui": ">>",
            "scalar.addf": "+",
            "scalar.subf": "-",
            "scalar.mulf": "*",
            "scalar.divf": "/",
        }
        if name in scalar_binary:
            fastmath = self._fastmath_contract(operation)
            self._declare(
                operation.results[0],
                self._binary_expression(operation, scalar_binary[name]),
                inlineable=True,
                statement_attribute=fastmath,
            )
            return ("fastmath",) if fastmath else ()

        vector_binary = {
            "vector.addi": "+",
            "vector.subi": "-",
            "vector.muli": "*",
            "vector.divsi": "/",
            "vector.divui": "/",
            "vector.remsi": "%",
            "vector.remui": "%",
            "vector.andi": "&",
            "vector.ori": "|",
            "vector.xori": "^",
            "vector.shli": "<<",
            "vector.shrui": ">>",
            "vector.addf": "+",
            "vector.subf": "-",
            "vector.mulf": "*",
            "vector.divf": "/",
        }
        if name in vector_binary:
            fastmath = self._fastmath_contract(operation)
            self._declare(
                operation.results[0],
                self._binary_expression(operation, vector_binary[name]),
                inlineable=True,
                statement_attribute=fastmath,
            )
            return ("fastmath",) if fastmath else ()

        if name == "scalar.divremui":
            lhs, rhs = map(self.expression, operation.operands)
            self._declare(operation.results[0], f"({lhs} / {rhs})", inlineable=True)
            self._declare(operation.results[1], f"({lhs} % {rhs})", inlineable=True)
            return ()
        projected = self._render_scalar_conversion(operation)
        if projected is not None:
            return projected
        if name == "scalar.bitcast":
            result = operation.results[0]
            source = self.expression(operation.operands[0])
            self._declare(
                result,
                f"__builtin_bit_cast({self.cpp_type(result)}, {source})",
                inlineable=True,
            )
            return ()
        projected = self._render_index_expression(operation)
        if projected is not None:
            return projected
        projected = self._render_comparison(operation)
        if projected is not None:
            return projected
        projected = self._render_scalar_intrinsic(operation)
        if projected is not None:
            return projected

        projected = self._render_structured_vector(operation)
        if projected is not None:
            return projected
        if name in {"vector.absf", "vector.absi"}:
            member = name.removeprefix("vector.")
            fastmath = self._fastmath_contract(operation)
            self._declare(
                operation.results[0],
                f"loom::vector::{member}({self.expression(operation.operands[0])})",
                inlineable=True,
                statement_attribute=fastmath,
            )
            return ("fastmath",) if fastmath else ()
        if name in {
            "vector.minimumf",
            "vector.maximumf",
            "vector.minnumf",
            "vector.maxnumf",
        }:
            member = name.removeprefix("vector.")
            fastmath = self._fastmath_contract(operation)
            arguments = ", ".join(
                self.expression(operand) for operand in operation.operands
            )
            self._declare(
                operation.results[0],
                f"loom::vector::{member}({arguments})",
                inlineable=True,
                statement_attribute=fastmath,
            )
            return ("fastmath",) if fastmath else ()
        if name in {
            "vector.minsi",
            "vector.maxsi",
            "vector.minui",
            "vector.maxui",
        }:
            member = name.removeprefix("vector.")
            arguments = ", ".join(
                self.expression(operand) for operand in operation.operands
            )
            self._declare(
                operation.results[0],
                f"loom::vector::{member}({arguments})",
                inlineable=True,
            )
            return ()
        if name in {
            "vector.extf",
            "vector.fptrunc",
            "vector.sitofp",
            "vector.uitofp",
            "vector.extui",
            "vector.extsi",
            "vector.trunci",
            "vector.fptosi",
            "vector.fptoui",
        }:
            return self._numeric_conversion(operation)
        if name == "vector.table.lookup":
            result = operation.results[0]
            table, indices = operation.operands
            result_type = self.cpp_type(result)
            table_type = self.cpp_type(table)
            indices_type = self.cpp_type(indices)
            function = f"table_lookup_{result_type.lower()}"
            self.table_lookup_declarations.add(
                (function, result_type, table_type, indices_type)
            )
            self._declare(
                result,
                f"{function}({self.expression(table)}, {self.expression(indices)})",
                inlineable=True,
            )
            return ()
        if name in {"vector.dot2f", "vector.dotf"}:
            arguments = ", ".join(
                self.expression(operand) for operand in operation.operands
            )
            fastmath = self._fastmath_contract(operation)
            member = name.removeprefix("vector.")
            self._declare(
                operation.results[0],
                f"loom::vector::{member}({arguments})",
                statement_attribute=fastmath,
            )
            return ("fastmath",) if fastmath else ()
        if name == "vector.reduce":
            kind = operation.attributes.get("kind")
            arguments = ", ".join(
                self.expression(operand) for operand in operation.operands
            )
            fastmath = self._fastmath_contract(operation)
            self._declare(
                operation.results[0],
                f"loom::vector::reduce::{kind}({arguments})",
                statement_attribute=fastmath,
            )
            return (
                ("kind", "fastmath")
                if "fastmath" in operation.attributes
                else ("kind",)
            )
        if name == "kernel.subgroup.reduce":
            kind = operation.attributes.get("kind")
            cluster_size = operation.attributes.get("cluster_size")
            cluster_stride = operation.attributes.get("cluster_stride")
            if cluster_stride is not None and cluster_size is None:
                raise CxxRenderError(
                    operation, "cluster_stride requires an explicit cluster_size"
                )
            template_arguments = [
                str(value)
                for value in (cluster_size, cluster_stride)
                if value is not None
            ]
            template = (
                f"<{', '.join(template_arguments)}>" if template_arguments else ""
            )
            self._declare(
                operation.results[0],
                f"loom::kernel::subgroup::reduce::{kind}{template}({self.expression(operation.operands[0])})",
            )
            return ("kind", "cluster_size", "cluster_stride")
        return None
