# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for test dialect generator coverage invariants."""

from loom.assembly import (
    Attr,
    AttrDict,
    AttrTable,
    BindingList,
    BlockArgs,
    FormatElement,
    FuncArgs,
    IndexList,
    Keyword,
    OperandDict,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    RegionTable,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypedRefs,
    TypeOf,
    TypesOf,
)
from loom.dialect.test import (
    ALL_TEST_OPS,
)
from loom.dsl import Op


class TestFormatFieldsMatchDeclarations:
    """Every field referenced in a format spec must exist on the op."""

    def _collect_fields(self, elements: tuple[FormatElement, ...]) -> set[str]:
        """Recursively collect all field names from format elements."""
        fields: set[str] = set()
        for elem in elements:
            match elem:
                case Ref(field=f) | Refs(field=f) | TypedRefs(field=f):
                    fields.add(f)
                case Attr(field=f):
                    fields.add(f)
                case SymbolRef(field=f):
                    fields.add(f)
                case TypeOf(field=f) | TypesOf(field=f):
                    fields.add(f)
                case ResultType(field=f) | ResultTypeList(field=f):
                    fields.add(f)
                case Region(field=f):
                    fields.add(f)
                case BindingList(field=f) | BlockArgs(region=f) | FuncArgs(field=f):
                    fields.add(f)
                case OperandDict(operands=operands, names=names):
                    fields.add(operands)
                    fields.add(names)
                case AttrTable(keys=keys, values=values):
                    fields.add(keys)
                    fields.add(values)
                case RegionTable(
                    keys=keys,
                    case_regions=case_regions,
                    default_region=default_region,
                ):
                    fields.add(keys)
                    fields.add(case_regions)
                    fields.add(default_region)
                case PredicateList(field=f):
                    fields.add(f)
                case IndexList(dynamic=d, static=s):
                    fields.add(d)
                    fields.add(s)
                case OptionalGroup(elements=inner):
                    fields |= self._collect_fields(inner)
                case Scope(elements=inner):
                    fields |= self._collect_fields(inner)
                case Keyword() | AttrDict():
                    pass
        return fields

    def _declared_fields(self, op: Op) -> set[str]:
        """All field names declared on an op."""
        fields: set[str] = set()
        for o in op.operands:
            fields.add(o.name)
        for result in op.results:
            fields.add(result.name)
        for attr in op.attrs:
            fields.add(attr.name)
        for region in op.regions:
            fields.add(region.name)
        return fields

    def test_all_ops_format_fields_valid(self) -> None:
        """Every field in every format spec must be a declared field,
        or a well-known implicit field (iv, args)."""
        # Fields that are implicit (not in operands/results/attrs/regions
        # but valid in format specs).
        implicit_fields = {"iv", "args"}

        for op in ALL_TEST_OPS:
            format_fields = self._collect_fields(op.format)
            declared = self._declared_fields(op) | implicit_fields
            unknown = format_fields - declared
            assert not unknown, f"{op.name}: format references undeclared fields: {unknown}. Declared: {declared}"


class TestConstraintFieldsMatchDeclarations:
    """Every field referenced in constraints must exist on the op."""

    def test_all_ops_constraint_fields_valid(self) -> None:
        implicit_fields = {"iv", "args"}
        for op in ALL_TEST_OPS:
            declared: set[str] = set()
            for operand in op.operands:
                declared.add(operand.name)
            for result in op.results:
                declared.add(result.name)
            for attr in op.attrs:
                declared.add(attr.name)
            for region in op.regions:
                declared.add(region.name)
            declared |= implicit_fields

            for constraint in op.constraints:
                for arg in constraint.args:
                    assert arg in declared, f"{op.name}: constraint {constraint} references undeclared field '{arg}'. Declared: {declared}"


class TestFormatElementCoverage:
    """Every format element type must appear in at least one test op."""

    def _all_element_types(self) -> set[type]:
        """Recursively collect all element types used across all ops."""
        types: set[type] = set()
        for op in ALL_TEST_OPS:
            self._collect_types(op.format, types)
        return types

    def _collect_types(
        self,
        elements: tuple[FormatElement, ...],
        types: set[type],
    ) -> None:
        for elem in elements:
            types.add(type(elem))
            if isinstance(elem, OptionalGroup | Scope):
                self._collect_types(elem.elements, types)

    def test_all_element_types_covered(self) -> None:
        used = self._all_element_types()
        required = {
            Ref,
            Refs,
            TypedRefs,
            Attr,
            SymbolRef,
            TypeOf,
            TypesOf,
            ResultType,
            ResultTypeList,
            Keyword,
            AttrDict,
            AttrTable,
            OperandDict,
            RegionTable,
            Region,
            IndexList,
            BindingList,
            BlockArgs,
            FuncArgs,
            PredicateList,
            OptionalGroup,
        }
        missing = required - used
        assert not missing, f"Format element types not covered: {missing}"


class TestCanonicalExamples:
    """Verify examples follow canonical form (one op per line, regions indented)."""

    def test_no_leading_whitespace(self) -> None:
        """Canonical examples have no leading indentation."""
        for op in ALL_TEST_OPS:
            for example in op.examples:
                first_line = example.split("\n")[0]
                assert not first_line.startswith("  "), f"{op.name}: example has leading indent: {first_line!r}"

    def test_region_ops_indented(self) -> None:
        """Ops inside regions are indented by 2 spaces."""
        # Lines that are structural (closing brace, else separator) are
        # at the outer indentation level, not indented.
        structural_prefixes = ("}", "} else {", "} launch {")
        for op in ALL_TEST_OPS:
            for example in op.examples:
                lines = example.split("\n")
                if len(lines) > 1:
                    for line in lines[1:-1]:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if any(stripped == p for p in structural_prefixes):
                            continue
                        assert line.startswith("  "), f"{op.name}: region body line not indented: {line!r}"
