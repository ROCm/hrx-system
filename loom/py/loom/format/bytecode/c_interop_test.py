# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-language bytecode coverage for bytecode-stable IR structure."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.low import ALL_LOW_OPS
from loom.dialect.test import (
    ALL_TEST_OPS,
    ALL_TEST_PARAMETERIZED_ATTRS,
    ALL_TEST_TYPES,
    test_array_type,
    test_matrix_type,
    test_scope_type,
    test_tile_attr,
)
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.ir import (
    BF16,
    CanonicalAttrDict,
    EnumArrayAttr,
    Module,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    RegisterType,
    SignedEnumSetAttr,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
)


def _run_loom_format(arguments: list[object]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"loom-format exited with {result.returncode}:\n{result.stderr}"
        )
    return result


def _interop_module() -> tuple[Module, RegisterType]:
    parser = Parser()
    parser.register_ops(ALL_LOW_OPS)
    parser.register_ops(ALL_TEST_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    parser.register_types(ALL_TEST_TYPES)
    parser.register_parameterized_attrs(ALL_TEST_PARAMETERIZED_ATTRS)
    module = parser.parse(
        "// Bytecode interoperability coverage.\n"
        "\n"
        "low.func.decl target<test.low.core> "
        "@typed_identity(%arg: reg<test.ptr : vector<4xi16>>) -> "
        "(reg<test.ptr : vector<4xi16>>)\n"
        "\n"
        "// Descriptor-backed attribute coverage.\n"
        "test.func @descriptor_attrs() {\n"
        "\n"
        "// Explicit entry block coverage.\n"
        "^entry:\n"
        "  test.enum_array_attrs [low, high, low] "
        "using [middle, <42>, middle]\n"
        "  test.symbol_array_attrs "
        "[@other_record, @parameterized_record, @other_record] "
        "using [@parameterized_record]\n"
        "  test.symbol_array_attrs [] using []\n"
        "  test.symbol_array_attrs []\n"
        "  test.symbol_set_attrs [@parameterized_record, @other_record]\n"
        "\n"
        "  // Grouped operation coverage.\n"
        "  test.parameterized_attr "
        "#test.options<mode = fast, scopes = [subgroup, <254>], "
        "element_type = bf16, tile = #test.tile<width = 16>, "
        "target = @parameterized_record, "
        "tiles = [#test.tile<width = 4>, #test.tile<width = 8>]>\n"
        "  test.parameterized_attr "
        "#test.options<mode = precise, scopes = []>\n"
        "\n"
        "  test.parameterized_attr #test.options<mode = fast>\n"
        "  test.parameterized_attr_array "
        "[#test.tile<width = 8>, #test.options<mode = precise>, "
        "#test.tile<width = 8>, "
        "#test.feature_set<[low, -middle, high]>] "
        "using [#test.tile<width = 4>]\n"
        "  test.signed_enum_set_attrs [low, -middle, high] using []\n"
        "  test.yield\n"
        "}\n"
        "test.record @parameterized_record "
        "{options = #test.options<mode = precise, scopes = []>}\n"
        "test.record @other_record\n"
        "test.decl @parameterized_types("
        "%scope: test.scope<subgroup>, "
        "%matrix: test.matrix<bf16, scope = workgroup, rows = 16, "
        "target = @parameterized_record>, "
        "%packed: test.array<bf16>, "
        "%aligned: test.array<bf16, alignment = 32>, "
        "%metadata: test.array<bf16, metadata = "
        "{tile = #test.tile<width = 8>}>)\n"
    )
    register_types = [
        value.type for value in module.values if isinstance(value.type, RegisterType)
    ]
    if len(register_types) != 2 or register_types[0] != register_types[1]:
        raise AssertionError("typed declaration did not preserve its signature types")
    return module, register_types[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("expected the C loom-format binary path")
    loom_format = Path(sys.argv[1])
    module, register_type = _interop_module()

    with tempfile.TemporaryDirectory(prefix="loom-bytecode-interop-") as temp_dir:
        temp_path = Path(temp_dir)
        python_bytecode_path = temp_path / "python.loombc"
        c_bytecode_path = temp_path / "c.loombc"
        python_bytecode_path.write_bytes(write_module(module))

        _run_loom_format(
            [
                loom_format,
                "--from=bc",
                "--to=bc",
                f"--output={c_bytecode_path}",
                python_bytecode_path,
            ]
        )
        loaded = read_module(
            c_bytecode_path.read_bytes(),
            parameterized_attrs=ALL_TEST_PARAMETERIZED_ATTRS,
            type_defs=ALL_TEST_TYPES,
        )
        if loaded.file_header != ("Bytecode interoperability coverage.",):
            raise AssertionError("file header did not survive C bytecode")
        if not any(value.type == register_type for value in loaded.values):
            raise AssertionError(
                "Python reader did not recover the C-written structural register type"
            )
        attribute_function = next(
            symbol.op for symbol in loaded.symbols if symbol.name == "descriptor_attrs"
        )
        if attribute_function is None:
            raise AssertionError("Python reader did not recover descriptor_attrs")
        if not attribute_function.leading_blank_line or attribute_function.comments != (
            "Descriptor-backed attribute coverage.",
        ):
            raise AssertionError("symbol source trivia did not survive C bytecode")
        entry_block = attribute_function.regions[0].blocks[0]
        if not entry_block.leading_blank_line or entry_block.comments != (
            "Explicit entry block coverage.",
        ):
            raise AssertionError("block source trivia did not survive C bytecode")
        enum_op = entry_block.ops[0]
        if enum_op.attributes["required_values"] != EnumArrayAttr([1, 255, 1]):
            raise AssertionError("required enum array did not survive C bytecode")
        if enum_op.attributes["optional_values"] != EnumArrayAttr([7, 42, 7]):
            raise AssertionError("open enum array did not survive C bytecode")
        symbol_ops = entry_block.ops[1:4]
        if symbol_ops[0].attributes["dependencies"] != SymbolNameArray(
            [
                SymbolName("other_record"),
                SymbolName("parameterized_record"),
                SymbolName("other_record"),
            ]
        ):
            raise AssertionError("symbol array lost order or duplicate names")
        if symbol_ops[0].attributes["available"] != SymbolNameArray(
            [SymbolName("parameterized_record")]
        ):
            raise AssertionError("optional symbol array changed")
        if symbol_ops[1].attributes["available"] != SymbolNameArray():
            raise AssertionError("present empty symbol array changed")
        if "available" in symbol_ops[2].attributes:
            raise AssertionError("absent symbol array became present")
        symbol_set_op = next(
            op for op in entry_block.ops if op.name == "test.symbol_set_attrs"
        )
        if symbol_set_op.attributes["symbols"] != SymbolNameSet(
            [SymbolName("other_record"), SymbolName("parameterized_record")]
        ):
            raise AssertionError("symbol set lost canonical name order")
        options = [op.attributes["options"] for op in entry_block.ops[5:8]]
        if not entry_block.ops[5].leading_blank_line or entry_block.ops[5].comments != (
            "Grouped operation coverage.",
        ):
            raise AssertionError("commented op source trivia did not survive bytecode")
        if entry_block.ops[6].leading_blank_line:
            raise AssertionError("adjacent operations gained a blank line")
        if not entry_block.ops[7].leading_blank_line or entry_block.ops[7].comments:
            raise AssertionError("blank-only op source trivia did not survive bytecode")
        if not all(isinstance(value, ParameterizedAttr) for value in options):
            raise AssertionError("Python reader did not recover parameterized attrs")
        first, present_empty, absent = options
        if first.get("mode") != 1 or first.get("scopes") != EnumArrayAttr([2, 254]):
            raise AssertionError(
                "parameterized enum payload did not survive C bytecode"
            )
        if first.get("element_type") != BF16:
            raise AssertionError(
                "parameterized type payload did not survive C bytecode"
            )
        tile = first.get("tile")
        if not isinstance(tile, ParameterizedAttr) or tile.get("width") != 16:
            raise AssertionError("nested parameterized attr did not survive C bytecode")
        if first.get("target") != SymbolName("parameterized_record"):
            raise AssertionError("parameterized symbol did not survive C bytecode")
        if first.get("tiles") != ParameterizedAttrArray(
            [test_tile_attr(width=4), test_tile_attr(width=8)]
        ):
            raise AssertionError(
                "nested parameterized array did not survive C bytecode"
            )
        if (
            not present_empty.has("scopes")
            or present_empty.get("scopes") != EnumArrayAttr()
        ):
            raise AssertionError("present empty parameter did not survive C bytecode")
        if absent.has("scopes"):
            raise AssertionError("absent parameter became present in C bytecode")
        array_op = next(
            op
            for op in entry_block.ops
            if op.name == "test.parameterized_attr_array"
        )
        array_values = array_op.attributes["values"]
        if not isinstance(array_values, ParameterizedAttrArray):
            raise AssertionError("Python reader did not recover parameterized array")
        if tuple(value.family_name for value in array_values) != (
            "test.tile",
            "test.options",
            "test.tile",
            "test.feature_set",
        ):
            raise AssertionError("mixed parameterized array lost element order")
        if array_values.values[0] != array_values.values[2]:
            raise AssertionError("repeated parameterized array value changed")
        if array_values.values[3].get("features") != SignedEnumSetAttr([1, 255], [7]):
            raise AssertionError("parameterized signed enum set changed")
        if array_op.attributes["tiles"] != ParameterizedAttrArray(
            [test_tile_attr(width=4)]
        ):
            raise AssertionError("exact-family parameterized array changed")
        signed_set_op = next(
            op
            for op in entry_block.ops
            if op.name == "test.signed_enum_set_attrs"
        )
        if signed_set_op.attributes["required_features"] != SignedEnumSetAttr(
            [1, 255], [7]
        ):
            raise AssertionError("signed enum set did not survive C bytecode")
        if signed_set_op.attributes["optional_features"] != SignedEnumSetAttr():
            raise AssertionError("present empty signed enum set changed")
        record = next(
            symbol.op
            for symbol in loaded.symbols
            if symbol.name == "parameterized_record"
        )
        if record is None:
            raise AssertionError("Python reader did not recover parameterized_record")
        record_options = record.attributes["dict"]["options"]
        if not isinstance(record_options, ParameterizedAttr):
            raise AssertionError("record dict lost its parameterized attribute")
        if (
            not record_options.has("scopes")
            or record_options.get("scopes") != EnumArrayAttr()
        ):
            raise AssertionError("record dict lost present empty parameter")
        parameterized_types = next(
            symbol.op
            for symbol in loaded.symbols
            if symbol.name == "parameterized_types"
        )
        if parameterized_types is None:
            raise AssertionError("Python reader did not recover parameterized_types")
        argument_types = tuple(
            loaded.values[value_id].type for value_id in parameterized_types.operands
        )
        expected_types = (
            test_scope_type(scope="subgroup"),
            test_matrix_type(
                element_type=BF16,
                scope="workgroup",
                rows=16,
                target=SymbolName("parameterized_record"),
            ),
            test_array_type(element_type=BF16),
            test_array_type(element_type=BF16, alignment=32),
            test_array_type(
                element_type=BF16,
                metadata=CanonicalAttrDict((("tile", test_tile_attr(width=8)),)),
            ),
        )
        if argument_types != expected_types or not all(
            isinstance(value, ParameterizedType) for value in argument_types
        ):
            raise AssertionError("descriptor-backed types did not survive C bytecode")


if __name__ == "__main__":
    main()
