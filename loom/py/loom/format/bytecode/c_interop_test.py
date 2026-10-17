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
from dataclasses import replace
from pathlib import Path

from loom.builder import IRBuilder
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.cfg import ALL_CFG_OPS
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.low import ALL_LOW_OPS
from loom.dialect.scf import ALL_SCF_OPS
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
from loom.format.text.printer import Printer
from loom.ir import (
    BF16,
    F32,
    INDEX,
    Block,
    CanonicalAttrDict,
    DynamicDim,
    DynamicEncoding,
    EnumArrayAttr,
    FunctionType,
    Module,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    PredicateListAttr,
    Region,
    RegisterType,
    ShapedType,
    SignedEnumSetAttr,
    StringArrayAttr,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
    TypeKind,
)
from loom.verify import verify_module


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
        '  test.string_array_attrs ["gfx12-generic", "gfx1250", '
        '"gfx12-generic"] using [] '
        '{fallbacks = ["gfx11-generic", "gfx12-generic"]}\n'
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


def _roundtrip_through_c(loom_format: Path, source: Module | str) -> Module:
    with tempfile.TemporaryDirectory(prefix="loom-bytecode-interop-") as temp_dir:
        temp_path = Path(temp_dir)
        source_format = "text" if isinstance(source, str) else "bc"
        source_path = temp_path / (
            "python.loom" if isinstance(source, str) else "python.loombc"
        )
        c_bytecode_path = temp_path / "c.loombc"
        if isinstance(source, str):
            source_path.write_text(source)
        else:
            source_path.write_bytes(write_module(source))

        _run_loom_format(
            [
                loom_format,
                f"--from={source_format}",
                "--to=bc",
                f"--output={c_bytecode_path}",
                source_path,
            ]
        )
        return read_module(
            c_bytecode_path.read_bytes(),
            parameterized_attrs=ALL_TEST_PARAMETERIZED_ATTRS,
            type_defs=ALL_TEST_TYPES,
        )


def _predicate_capture_module() -> Module:
    parser = Parser()
    for operations in (ALL_FUNC_OPS, ALL_SCF_OPS, ALL_TEST_OPS):
        parser.register_ops(operations)
    module = parser.parse(
        "func.decl @first(%extent: index) where [ge(%extent, 1)]\n"
        "func.decl @unsigned(%lhs: i32, %rhs: i32) "
        "where [ule(%lhs, %rhs)]\n"
        "func.def @capture(%condition: i1, %extent: index) "
        "where [ge(%extent, 2)] {\n"
        "  scf.if %condition {\n"
        "    %inner = test.constant 4 : index\n"
        "    %bounded = test.assume %condition "
        "[eq(%extent, 7), eq(%inner, 4)] : i1\n"
        "  }\n"
        "  func.return\n"
        "}\n"
    )
    nested = module.body.ops[2].regions[0].blocks[0].ops[0].regions[0].blocks[0]
    # Changing a display name leaves the resolved outer and inner IDs intact.
    module.values[nested.ops[0].results[0]].name = "extent"
    return module


def _assert_predicate_identities(module: Module) -> None:
    first, unsigned, capture = module.body.ops
    assert first.attributes["predicates"][0].args[0].value == first.operands[0]
    unsigned_predicate = unsigned.attributes["predicates"][0]
    assert unsigned_predicate.kind == "ule"
    assert unsigned_predicate.args[0].value == unsigned.operands[0]
    assert unsigned_predicate.args[1].value == unsigned.operands[1]
    body = capture.regions[0].blocks[0]
    extent = body.arg_ids[1]
    assert capture.attributes["predicates"][0].args[0].value == extent
    nested = body.ops[0].regions[0].blocks[0]
    inner = nested.ops[0].results[0]
    predicates = nested.ops[1].attributes["predicates"]
    assert extent != inner
    assert predicates[0].args[0].value == extent
    assert predicates[1].args[0].value == inner


def _assert_module_structure(module: Module, register_type: RegisterType) -> Block:
    if module.file_header != ("Bytecode interoperability coverage.",):
        raise AssertionError("file header did not survive C bytecode")
    if not any(value.type == register_type for value in module.values):
        raise AssertionError(
            "Python reader did not recover the C-written structural register type"
        )
    attribute_function = next(
        (symbol.op for symbol in module.symbols if symbol.name == "descriptor_attrs"),
        None,
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
    return entry_block


def _assert_enum_and_symbol_attrs(entry_block: Block) -> None:
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


def _assert_parameterized_attrs(entry_block: Block) -> None:
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
        raise AssertionError("parameterized enum payload did not survive C bytecode")
    if first.get("element_type") != BF16:
        raise AssertionError("parameterized type payload did not survive C bytecode")
    tile = first.get("tile")
    if not isinstance(tile, ParameterizedAttr) or tile.get("width") != 16:
        raise AssertionError("nested parameterized attr did not survive C bytecode")
    if first.get("target") != SymbolName("parameterized_record"):
        raise AssertionError("parameterized symbol did not survive C bytecode")
    if first.get("tiles") != ParameterizedAttrArray(
        [test_tile_attr(width=4), test_tile_attr(width=8)]
    ):
        raise AssertionError("nested parameterized array did not survive C bytecode")
    if (
        not present_empty.has("scopes")
        or present_empty.get("scopes") != EnumArrayAttr()
    ):
        raise AssertionError("present empty parameter did not survive C bytecode")
    if absent.has("scopes"):
        raise AssertionError("absent parameter became present in C bytecode")


def _assert_parameterized_arrays(entry_block: Block) -> None:
    array_op = next(
        op for op in entry_block.ops if op.name == "test.parameterized_attr_array"
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
        op for op in entry_block.ops if op.name == "test.signed_enum_set_attrs"
    )
    if signed_set_op.attributes["required_features"] != SignedEnumSetAttr(
        [1, 255], [7]
    ):
        raise AssertionError("signed enum set did not survive C bytecode")
    if signed_set_op.attributes["optional_features"] != SignedEnumSetAttr():
        raise AssertionError("present empty signed enum set changed")

    string_array_op = next(
        op for op in entry_block.ops if op.name == "test.string_array_attrs"
    )
    if string_array_op.attributes["required_values"] != StringArrayAttr(
        ["gfx12-generic", "gfx1250", "gfx12-generic"]
    ):
        raise AssertionError("string array lost order or duplicate values")
    if string_array_op.attributes["optional_values"] != StringArrayAttr():
        raise AssertionError("present empty string array changed")
    if string_array_op.attributes["dict"]["fallbacks"] != StringArrayAttr(
        ["gfx11-generic", "gfx12-generic"]
    ):
        raise AssertionError("generic dict string array changed")


def _assert_symbol_payloads(module: Module) -> None:
    record = next(
        (
            symbol.op
            for symbol in module.symbols
            if symbol.name == "parameterized_record"
        ),
        None,
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
        (
            symbol.op
            for symbol in module.symbols
            if symbol.name == "parameterized_types"
        ),
        None,
    )
    if parameterized_types is None:
        raise AssertionError("Python reader did not recover parameterized_types")
    argument_types = tuple(
        module.values[value_id].type for value_id in parameterized_types.operands
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


def _assert_cfg_identities(module: Module) -> None:
    entry, forward, exit = module.body.ops[0].regions[0].blocks
    assert entry.ops[0].successors[0] is forward
    assert forward.ops[1].successors[0] is exit
    assert entry.ops[0].operands[0] == entry.arg_ids[2]
    assert forward.ops[0].operands[0] == forward.arg_ids[0]
    assert exit.ops[0].operands[0] == forward.arg_ids[0]
    argument = module.values[forward.arg_ids[0]]
    assert argument.type.dims[0] == DynamicDim(entry.arg_ids[0])
    assert argument.type.encoding == DynamicEncoding(entry.arg_ids[1])


def _test_cfg_interop(loom_format: Path) -> None:
    parser = Parser()
    printer = Printer()
    for format in (parser, printer):
        format.register_types(ALL_BUILTIN_TYPES)
        for operations in (ALL_FUNC_OPS, ALL_CFG_OPS, ALL_TEST_OPS):
            format.register_ops(operations)
    module = parser.parse(
        "func.def @f(%extent: index, %layout: encoding, "
        "%value: tile<[%extent]xf32, %layout>) {\n"
        "  cfg.br ^forward(%value: tile<[%extent]xf32, %layout>)\n"
        "^forward(%forwarded: tile<[%extent]xf32, %layout>):\n"
        "  test.use %forwarded : tile<[%extent]xf32, %layout>\n"
        "  cfg.br ^exit\n"
        "^exit:\n"
        "  test.use %forwarded : tile<[%extent]xf32, %layout>\n"
        "  func.return\n"
        "}\n",
        verify=True,
    )
    _assert_cfg_identities(module)
    region = module.body.ops[0].regions[0]
    entry, forward, exit = region.blocks
    region.blocks[:] = [entry, exit, forward]
    for loaded in (
        read_module(write_module(module)),
        _roundtrip_through_c(loom_format, module),
        _roundtrip_through_c(loom_format, printer.print_module(module)),
    ):
        _assert_cfg_identities(loaded)
        _assert_cfg_identities(parser.parse(printer.print_module(loaded), verify=True))


def _test_region_argument_interop(loom_format: Path) -> None:
    parser = Parser()
    printer = Printer()
    for format in (parser, printer):
        format.register_types(ALL_BUILTIN_TYPES)
        for operations in (ALL_FUNC_OPS, ALL_SCF_OPS, ALL_TEST_OPS):
            format.register_ops(operations)
    module = parser.parse(
        "func.def @loop(%extent: index, %layout: encoding, "
        "%input: tile<[%extent]xf32, %layout>, "
        "%lower: index, %upper: index, %step: index) {\n"
        "  %result = scf.for %iv = [%lower to %upper step %step]"
        "(%iter = %input : tile<[%extent]xf32, %layout>) "
        "-> (tile<[%extent]xf32, %layout>) {\n"
        "    scf.yield %iter : tile<[%extent]xf32, %layout>\n"
        "  }\n"
        "  test.block_args %input : tile<[%extent]xf32, %layout> "
        "do(%arg: tile<[%extent]xf32, %layout>) {\n"
        "    test.use %arg : tile<[%extent]xf32, %layout>\n"
        "    test.yield\n"
        "  }\n"
        "  func.return\n"
        "}\n"
        "test.split_func @projection(%extent: index, %alternate: index, %layout: encoding, "
        "%input: tile<[%extent]xf32, %layout>) {\n"
        "  test.use %input : tile<[%extent]xf32, %layout>\n"
        "  test.yield\n"
        "} launch {\n"
        "  test.use %input : tile<[%extent]xf32, %layout>\n"
        "  test.yield\n"
        "}\n",
        verify=True,
    )
    for source in (module, printer.print_module(module)):
        loaded = _roundtrip_through_c(loom_format, source)
        for candidate in (
            loaded,
            parser.parse(printer.print_module(loaded), verify=True),
        ):
            entry = candidate.body.ops[0].regions[0].blocks[0]
            for op in entry.ops[:2]:
                nested = op.regions[0].blocks[0]
                argument_id = nested.arg_ids[-1]
                argument = candidate.values[argument_id]
                assert argument.type.dims[0] == DynamicDim(entry.arg_ids[0])
                assert argument.type.encoding == DynamicEncoding(entry.arg_ids[1])
                assert nested.ops[0].operands[0] == argument_id
            config, body = candidate.body.ops[1].regions
            assert set(config.blocks[0].arg_ids).isdisjoint(body.blocks[0].arg_ids)
            for region in (config, body):
                extent, _alternate, layout, input = region.blocks[0].arg_ids
                assert candidate.values[input].type.dims[0] == DynamicDim(extent)
                assert candidate.values[input].type.encoding == DynamicEncoding(layout)
                assert region.blocks[0].ops[0].operands[0] == input

    # Equal structural shapes must not hide a reference to the wrong peer.
    config_args = module.body.ops[1].regions[0].blocks[0].arg_ids
    argument = module.values[config_args[3]]
    argument.type = replace(argument.type, dims=(DynamicDim(config_args[1]),))
    with tempfile.TemporaryDirectory(prefix="loom-projected-argument-") as temp_dir:
        source_path = Path(temp_dir) / "invalid.loombc"
        source_path.write_bytes(write_module(module))
        result = subprocess.run(
            [loom_format, "--from=bc", "--to=text", source_path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "error [TYPE/013]" in result.stderr


def _test_tied_signature_interop(loom_format: Path) -> None:
    parser, printer = Parser(), Printer()
    for format in (parser, printer):
        format.register_types(ALL_BUILTIN_TYPES)
        format.register_ops(ALL_FUNC_OPS)
    module = parser.parse(
        "func.decl @shape(%input: index) -> "
        "(%extent: %input as index, tensor<[%extent]xf32>) "
        "where [range(%extent, 1, 32)]\n"
        "func.decl @window(%input: buffer) -> (%output: %input as view<16xf32>)\n"
        "func.def @identity(%input: index) -> (%extent: %input as index) "
        "where [range(%extent, 1, 32)] {\n"
        "  func.return %input : index\n"
        "}\n",
        verify=True,
    )
    for op in module.body.ops:
        module.values[op.results[0]].name = ""
    for source in (module, printer.print_module(module)):
        loaded = _roundtrip_through_c(loom_format, source)
        for candidate in (
            loaded,
            parser.parse(printer.print_module(loaded), verify=True),
        ):
            shape, window, identity = candidate.body.ops
            for op in (shape, window, identity):
                assert op.tied_results[0].operand_index == 0
                assert op.tied_results[0].result_index == 0
            assert candidate.values[shape.results[1]].type.dims[0] == DynamicDim(
                shape.results[0]
            )
            for op in (shape, identity):
                assert op.attributes["predicates"][0].args[0].value == op.results[0]
            assert (
                candidate.values[window.results[0]].type
                != candidate.values[window.operands[0]].type
            )


def _fixture_sources(fixture: Path) -> list[str]:
    return [
        "\n".join(
            line
            for line in case.partition("// ----")[0].splitlines()
            if not line.startswith("// RUN:")
        ).lstrip()
        for case in fixture.read_text().split("// ====")
    ]


def _test_scoped_type_interop(
    loom_format: Path, loom_link: Path, fixture: Path
) -> None:
    parser = Parser()
    printer = Printer()
    for operations in (ALL_TEST_OPS, ALL_LOW_OPS):
        parser.register_ops(operations)
        printer.register_ops(operations)
    parser.register_types((*ALL_BUILTIN_TYPES, *ALL_TEST_TYPES))
    printer.register_types((*ALL_BUILTIN_TYPES, *ALL_TEST_TYPES))
    parser.register_parameterized_attrs(ALL_TEST_PARAMETERIZED_ATTRS)
    [source] = _fixture_sources(fixture)
    module = parser.parse(source)
    canonical = printer.print_module(module)
    for candidate in (
        read_module(
            write_module(module),
            type_defs=ALL_TEST_TYPES,
            parameterized_attrs=ALL_TEST_PARAMETERIZED_ATTRS,
        ),
        _roundtrip_through_c(loom_format, module),
        _roundtrip_through_c(loom_format, source),
    ):
        assert printer.print_module(candidate) == canonical
        nested = candidate.body.ops[0]
        width, height, value = nested.regions[0].blocks[0].arg_ids
        value_type = candidate.values[value].type
        child = value_type.get("element_type").get("element_type")
        assert child.dims == (DynamicDim(width),)
        assert value_type.get("metadata")["shape"].dims == (DynamicDim(height),)
        register = candidate.body.ops[1]
        width, value = register.operands
        assert candidate.values[value].type.value_type.dims == (DynamicDim(width),)
        entry = candidate.body.ops[2].regions[0].blocks[0]
        attribute_type = entry.ops[0].attributes["dict"]["shape"].get("element_type")
        assert attribute_type.get("element_type").dims == (
            DynamicDim(entry.arg_ids[0]),
        )
        packed = candidate.body.ops[3]
        width, layout, value = packed.regions[0].blocks[0].arg_ids
        packed_type = candidate.values[value].type
        child = packed_type.get("element_type")
        assert child.dims[0] == DynamicDim(width)
        assert child.encoding == DynamicEncoding(layout)
        assert child.alignment == 2
        metadata = packed_type.get("metadata")
        assert metadata["fixed"].alignment == 1
        callback = metadata["callback"]
        assert callback.arg_types[0].alignment == 1
        assert callback.result_types[0].alignment == 1

    with tempfile.TemporaryDirectory(
        prefix="loom-scoped-selected-interop-"
    ) as directory:
        source_path = Path(directory) / "python.loombc"
        selected_path = Path(directory) / "selected.loombc"
        source_path.write_bytes(write_module(module))
        result = subprocess.run(
            [
                loom_link,
                source_path,
                "--root=nested",
                "--root=type_attribute",
                "--root=packed_views",
                "--to=bc",
                f"--output={selected_path}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        selected = read_module(
            selected_path.read_bytes(),
            type_defs=ALL_TEST_TYPES,
            parameterized_attrs=ALL_TEST_PARAMETERIZED_ATTRS,
        )
    assert {symbol.name for symbol in selected.symbols} == {
        "nested",
        "type_attribute",
        "packed_views",
    }
    selected_text = printer.print_module(selected)
    assert "metadata = {shape = vector<[%height]xf32>}" in selected_text
    assert "element_type = test.array<vector<[%width]xf32>>" in selected_text
    assert "view<[%width]x2x3xi64, %layout, align(2)>" in selected_text
    assert printer.print_module(parser.parse(selected_text)) == selected_text


def _test_predicate_attribute_interop(
    loom_format: Path, loom_link: Path, fixture: Path
) -> None:
    parser = Parser()
    printer = Printer()
    for operations in (ALL_TEST_OPS, ALL_FUNC_OPS, ALL_INDEX_OPS):
        parser.register_ops(operations)
        printer.register_ops(operations)
    parser.register_types((*ALL_BUILTIN_TYPES, *ALL_TEST_TYPES))
    printer.register_types((*ALL_BUILTIN_TYPES, *ALL_TEST_TYPES))
    for source in _fixture_sources(fixture):
        module = parser.parse(source)
        canonical = printer.print_module(module)
        if module.symbols[0].name == "static_predicates":
            argument = module.body.ops[0].operands[0]
            metadata = module.values[argument].type.get("metadata")
            assert metadata["empty"] == PredicateListAttr()
            assert metadata["integers"] == []
            assert metadata["empty"] != metadata["integers"]
        for candidate in (
            read_module(write_module(module), type_defs=ALL_TEST_TYPES),
            _roundtrip_through_c(loom_format, module),
            _roundtrip_through_c(loom_format, source),
        ):
            assert printer.print_module(candidate) == canonical
            assert printer.print_module(parser.parse(canonical)) == canonical
        with tempfile.TemporaryDirectory(prefix="loom-predicate-interop-") as directory:
            source_path = Path(directory) / "source.loombc"
            selected_path = Path(directory) / "selected.loombc"
            source_path.write_bytes(write_module(module))
            result = subprocess.run(
                [
                    loom_link,
                    source_path,
                    f"--root={module.symbols[0].name}",
                    "--to=bc",
                    f"--output={selected_path}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            selected = read_module(selected_path.read_bytes(), type_defs=ALL_TEST_TYPES)
        selected_text = printer.print_module(selected)
        assert selected_text == canonical, (canonical, selected_text)


def _test_shared_projected_graph(loom_format: Path, loom_link: Path) -> None:
    builder = IRBuilder()
    builder.register_ops(ALL_TEST_OPS)
    extent = builder.value("extent", INDEX)
    root = ShapedType(TypeKind.VECTOR, F32, (DynamicDim(extent.id),))
    for _ in range(3):
        root = FunctionType((root, root), (root,))
    value = builder.value("payload", root)
    projected = builder.module.clone_func_signature_args([extent.id, value.id])
    regions = [Region(blocks=[Block(arg_ids=projected)]), Region(blocks=[Block()])]
    builder.build(
        "test.split_func",
        func_args=[extent, value],
        attributes={"callee": "shared"},
        regions=regions,
    )
    for region in regions:
        builder.set_insertion_block(region.blocks[0])
        builder.build("test.yield")
    verify_module(builder.module, ops=ALL_TEST_OPS).raise_if_errors()
    printer = Printer()
    printer.register_ops(ALL_TEST_OPS)
    canonical = printer.print_module(builder.module)
    full = _roundtrip_through_c(loom_format, builder.module)
    assert printer.print_module(full) == canonical
    with tempfile.TemporaryDirectory(prefix="loom-shared-graph-interop-") as directory:
        source_path = Path(directory) / "source.loombc"
        selected_path = Path(directory) / "selected.loombc"
        source_path.write_bytes(write_module(builder.module, op_decls=ALL_TEST_OPS))
        result = subprocess.run(
            [
                loom_link,
                source_path,
                "--root=shared",
                "--to=bc",
                f"--output={selected_path}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        selected = read_module(selected_path.read_bytes())
    verify_module(selected, ops=ALL_TEST_OPS).raise_if_errors()
    text = printer.print_module(selected)
    assert text == canonical
    parser = Parser()
    parser.register_ops(ALL_TEST_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    assert printer.print_module(parser.parse(text, verify=True)) == canonical


def main() -> None:
    if len(sys.argv) != 5:
        raise ValueError(
            "expected C loom-format, loom-link, predicate and scoped-type fixture paths"
        )
    _test_predicate_attribute_interop(
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    )
    _test_scoped_type_interop(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4]))
    _test_shared_projected_graph(Path(sys.argv[1]), Path(sys.argv[2]))
    source_module, register_type = _interop_module()
    loaded_module = _roundtrip_through_c(Path(sys.argv[1]), source_module)
    source_symbols = {symbol.name: symbol for symbol in source_module.symbols}
    for symbol in loaded_module.symbols:
        source_op = source_symbols[symbol.name].op
        assert source_op is not None and source_op.location_id != 0
        assert symbol.op is not None
        assert loaded_module.locations.get(symbol.op.location_id) == (
            source_module.locations.get(source_op.location_id)
        )
    entry_block = _assert_module_structure(loaded_module, register_type)
    _assert_enum_and_symbol_attrs(entry_block)
    _assert_parameterized_attrs(entry_block)
    _assert_parameterized_arrays(entry_block)
    _assert_symbol_payloads(loaded_module)
    captured = _predicate_capture_module()
    _assert_predicate_identities(captured)
    _assert_predicate_identities(read_module(write_module(captured)))
    captured_from_c = _roundtrip_through_c(Path(sys.argv[1]), captured)
    _assert_predicate_identities(captured_from_c)
    parser = Parser()
    printer = Printer()
    for operations in (ALL_FUNC_OPS, ALL_SCF_OPS, ALL_TEST_OPS):
        parser.register_ops(operations)
        printer.register_ops(operations)
    for module in (captured, captured_from_c):
        text = printer.print_module(module)
        _assert_predicate_identities(parser.parse(text))
    _test_cfg_interop(Path(sys.argv[1]))
    _test_region_argument_interop(Path(sys.argv[1]))
    _test_tied_signature_interop(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
