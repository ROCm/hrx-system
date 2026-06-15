# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Library-form MLIR HAL kernel importer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.builder import ValueRef
from loom.diagnostics import DiagnosticEngine
from loom.importers.core import (
    ImportBodyReport,
    ImportOptions,
    ImportResult,
    KernelArgumentSpec,
    KernelLaunchConfigSpec,
    KernelModuleSpec,
    create_kernel_module,
    kernel_module_ops,
    sanitize_symbol,
)
from loom.importers.mlir.api import import_iree_ir
from loom.importers.mlir.attrs import MlirAttributeDecoder
from loom.importers.mlir.convert_gpu import gpu_dimension_axis
from loom.importers.mlir.converter import convert_function_body, walk_operations
from loom.importers.mlir.model import Binding, KernelFacts, MlirConversionContext
from loom.importers.mlir.names import build_source_name_overrides
from loom.importers.mlir.types import MlirTypeConverter
from loom.ir import INDEX, Module, rebuild_value_metadata
from loom.verify import verify_module

AXIS_BY_INDEX = {
    0: "x",
    1: "y",
    2: "z",
}


@dataclass(frozen=True, slots=True)
class MlirImportOptions(ImportOptions):
    """Options specific to MLIR HAL kernel imports."""

    kernel: str | None = None
    prefer_abi3_extensions: bool = False


def import_mlir_file(
    path: str | Path,
    *,
    options: MlirImportOptions | None = None,
) -> ImportResult:
    source_path = Path(path)
    return import_mlir_module(source_path.read_text(), options=options)


def import_mlir_module(
    text: str,
    *,
    options: MlirImportOptions | None = None,
) -> ImportResult:
    options = options or MlirImportOptions()
    diagnostics = DiagnosticEngine()
    chunk = choose_chunk(split_input_file(text), options.kernel)
    function_name = extract_function_name(chunk, options.kernel)

    ir = import_iree_ir(prefer_abi3_extensions=options.prefer_abi3_extensions)
    with ir.Context():
        parsed_module = ir.Module.parse(chunk)
        source_names = build_source_name_overrides(parsed_module.operation, chunk)
        loom_module, facts = _import_parsed_module(
            parsed_module,
            function_name,
            source_names=source_names,
            diagnostics=diagnostics,
        )

    diagnostics.raise_if_errors()
    if options.verify_structure:
        verify_module(
            loom_module,
            ops=kernel_module_ops(facts.target_format or "unknown"),
            diagnostics=diagnostics,
        )
        diagnostics.raise_if_errors()
    return ImportResult(
        module=loom_module,
        diagnostics=diagnostics,
        report=facts if options.include_report else None,
    )


def _import_parsed_module(
    parsed_module: Any,
    function_name: str,
    *,
    source_names: dict[object, str],
    diagnostics: DiagnosticEngine,
) -> tuple[Module, KernelFacts]:
    operations = list(walk_operations(parsed_module.operation))
    operation_counts = Counter(operation_name(op) for op in operations)
    root = parsed_module.operation
    function_operation = find_symbol_operation(root, "func.func", function_name)
    if function_operation is None:
        diagnostics.error(f"func.func @{function_name} not found in parsed MLIR")
        diagnostics.raise_if_errors()
        raise AssertionError("unreachable after diagnostics.raise_if_errors")
    executable_operation = first_operation(root, "hal.executable")
    variant_operation = first_operation(root, "hal.executable.variant")
    export_operation = first_operation(root, "hal.executable.export")
    target_driver, target_format = parse_target(variant_operation)
    export_name = symbol_name(export_operation)
    type_converter = MlirTypeConverter()
    bindings = parse_bindings(function_operation, type_converter)
    workgroup_size = parse_workgroup_size(function_operation)
    subgroup_size = parse_subgroup_size(function_operation)
    loom_module, converted_body = build_loom_module(
        function_operation,
        bindings,
        target_format=target_format,
        export_name=export_name,
        function_name=function_name,
        workgroup_size=workgroup_size,
        diagnostics=diagnostics,
        type_converter=type_converter,
        source_names=source_names,
    )
    facts = KernelFacts(
        executable_name=symbol_name(executable_operation),
        variant_name=symbol_name(variant_operation),
        target_driver=target_driver,
        target_format=target_format,
        export_name=export_name,
        function_name=function_name,
        workgroup_size=workgroup_size,
        subgroup_size=subgroup_size,
        bindings=bindings,
        operation_counts=operation_counts,
        converted_body=converted_body,
    )
    return loom_module, facts


def split_input_file(text: str) -> list[str]:
    chunks: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip() == "// -----":
            chunks.append([])
        else:
            chunks[-1].append(line)
    return [chunk for lines in chunks if (chunk := "\n".join(lines).strip())]


def choose_chunk(chunks: list[str], kernel_name: str | None) -> str:
    if not chunks:
        raise ValueError("input is empty")
    if kernel_name is None:
        return chunks[0]
    matches = [chunk for chunk in chunks if chunk_mentions_kernel(chunk, kernel_name)]
    if not matches:
        raise ValueError(f"no split-input chunk mentions kernel/export @{kernel_name}")
    if len(matches) > 1:
        raise ValueError(
            f"kernel/export @{kernel_name} appears in {len(matches)} chunks"
        )
    return matches[0]


def chunk_mentions_kernel(chunk: str, kernel_name: str) -> bool:
    for line in chunk.splitlines():
        if line_symbol_after(line, "func.func") == kernel_name:
            return True
        if line_symbol_after(line, "hal.executable.export") == kernel_name:
            return True
    return False


def extract_function_name(text: str, requested: str | None) -> str:
    func_matches = [
        symbol
        for line in text.splitlines()
        if (symbol := line_symbol_after(line, "func.func")) is not None
    ]
    export_matches = [
        symbol
        for line in text.splitlines()
        if (symbol := line_symbol_after(line, "hal.executable.export")) is not None
    ]
    if requested and requested in func_matches:
        return requested
    if requested and requested in export_matches:
        return requested
    if not func_matches:
        raise ValueError("input has no func.func implementation body")
    if requested:
        raise ValueError(
            f"input has no func.func @{requested}; found {', '.join(func_matches)}"
        )
    if len(func_matches) > 1:
        raise ValueError(
            "input has multiple funcs; pass an explicit kernel. "
            f"Found: {', '.join(func_matches)}"
        )
    return func_matches[0]


def line_symbol_after(line: str, op_spelling: str) -> str | None:
    if op_spelling not in line:
        return None
    at = line.find("@", line.find(op_spelling))
    if at < 0:
        return None
    end = at + 1
    while end < len(line) and is_symbol_char(line[end]):
        end += 1
    if end == at + 1:
        return None
    return line[at + 1 : end]


def is_symbol_char(char: str) -> bool:
    return char.isalnum() or char in {"_", ".", "$", "-"}


def operation_name(operation: object) -> str:
    op = getattr(operation, "operation", operation)
    return str(getattr(op, "name", "<unknown>"))


def first_operation(root_operation: Any, op_name: str) -> Any | None:
    for operation in walk_operations(root_operation):
        if operation_name(operation) == op_name:
            return operation
    return None


def find_symbol_operation(
    root_operation: Any,
    op_name: str,
    symbol: str,
) -> Any | None:
    for operation in walk_operations(root_operation):
        if operation_name(operation) == op_name and symbol_name(operation) == symbol:
            return operation
    return None


def symbol_name(operation: Any | None) -> str | None:
    if operation is None:
        return None
    attr = operation.attributes.get("sym_name")
    return attr.value if attr is not None else None


def parse_workgroup_size(function_operation: Any) -> tuple[int, int, int] | None:
    translation_info = function_operation.attributes.get("translation_info")
    if translation_info is None:
        return None
    values = list(translation_info.workgroup_size)
    if len(values) == 1:
        values += [1, 1]
    if len(values) == 2:
        values += [1]
    if len(values) != 3:
        raise ValueError(f"unsupported workgroup_size rank: {values}")
    return (int(values[0]), int(values[1]), int(values[2]))


def parse_subgroup_size(function_operation: Any) -> int | None:
    translation_info = function_operation.attributes.get("translation_info")
    return translation_info.subgroup_size if translation_info is not None else None


def parse_target(variant_operation: Any | None) -> tuple[str | None, str | None]:
    if variant_operation is None:
        return None, None
    target = variant_operation.attributes.get("target")
    if target is None:
        return None, None
    text = str(target)
    prefix = '#hal.executable.target<"'
    if not text.startswith(prefix) or not text.endswith('">'):
        return None, None
    body = text[len(prefix) : -2]
    parts = body.split('", "', 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def parse_bindings(
    function_operation: Any,
    type_converter: MlirTypeConverter,
) -> tuple[Binding, ...]:
    bindings: list[Binding] = []
    for operation in walk_operations(function_operation):
        if operation_name(operation) != "hal.interface.binding.subspan":
            continue
        binding_attr = operation.attributes.get("binding")
        if binding_attr is None:
            raise ValueError(
                f"binding subspan has no binding attr: {first_line(operation)}"
            )
        alignment_attr = operation.attributes.get("alignment")
        flags_attr = operation.attributes.get("flags")
        offset = operation.operands[0].get_name() if len(operation.operands) else None
        source_type = str(operation.results[0].type)
        view_type = type_converter.memref_to_view_type_text(source_type)
        bindings.append(
            Binding(
                result=operation.results[0].get_name(),
                binding=binding_attr.value,
                alignment=str(alignment_attr.value)
                if alignment_attr is not None
                else None,
                offset=offset,
                flags=str(flags_attr) if flags_attr is not None else None,
                source_type=source_type,
                view_type=view_type,
            )
        )
    if not bindings:
        raise ValueError("kernel has no hal.interface.binding.subspan ops")
    return tuple(sorted(bindings, key=lambda binding: binding.binding))


def build_loom_module(
    function_operation: Any,
    bindings: tuple[Binding, ...],
    *,
    target_format: str | None,
    export_name: str | None,
    function_name: str,
    workgroup_size: tuple[int, int, int] | None,
    diagnostics: DiagnosticEngine,
    type_converter: MlirTypeConverter,
    source_names: dict[object, str],
) -> tuple[Module, ImportBodyReport]:
    target_preset = target_format or "unknown"
    target_symbol = sanitize_symbol(target_preset)
    kernel_name = export_name or function_name
    workgroup = workgroup_size or (1, 1, 1)
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset=target_preset,
            target_symbol=target_symbol,
            export_symbol=kernel_name,
            launch_config=KernelLaunchConfigSpec(workgroup_size=workgroup),
            callee=kernel_name,
            arguments=[
                KernelArgumentSpec(
                    ordinal=binding.binding,
                    name=f"binding{binding.binding}",
                )
                for binding in bindings
            ],
        )
    )
    module = shell.module
    builder = shell.builder
    binding_args = shell.body_arguments_by_ordinal

    topology_values: dict[tuple[str, str], ValueRef] = {}
    prelude_values: dict[Any, tuple[ValueRef, str | None]] = {}
    attr_decoder = MlirAttributeDecoder()

    with builder.insertion_block(shell.body_block):
        binding_by_result = {binding.result: binding for binding in bindings}
        for operation in walk_operations(function_operation):
            name = operation_name(operation)
            if name == "gpu.thread_id":
                result = operation.results[0]
                axis = gpu_dimension_axis(operation.attributes.get("dimension"))
                if axis is not None:
                    key = ("workitem", axis)
                    value = topology_values.get(key)
                    if value is None:
                        value = builder.kernel.workitem_id(
                            dimension=axis,
                            results=[INDEX],
                            name={"x": "tidx", "y": "tidy", "z": "tidz"}[axis],
                        )
                        topology_values[key] = value
                    prelude_values[result] = (value, "index")
            elif name == "hal.interface.workgroup.id":
                result = operation.results[0]
                dimension = attr_decoder.integer(operation.attributes.get("dimension"))
                axis = workgroup_axis(dimension)
                if axis is not None:
                    key = ("workgroup", axis)
                    value = topology_values.get(key)
                    if value is None:
                        value = builder.kernel.workgroup_id(
                            dimension=axis,
                            results=[INDEX],
                            name=f"wg{axis}",
                        )
                        topology_values[key] = value
                    prelude_values[result] = (value, "index")

        context = MlirConversionContext.with_prelude(
            builder,
            prelude_values,
            diagnostics=diagnostics,
            preview_block=shell.body_block,
            type_converter=type_converter,
            source_names=source_names,
            binding_args=binding_args,
            bindings_by_result=binding_by_result,
        )
        converted_body = convert_function_body(function_operation, context)
        builder.kernel.return_()

    rebuild_value_metadata(module)
    return module, converted_body


def first_line(operation: Any) -> str:
    return str(operation).splitlines()[0].strip()


def workgroup_axis(dimension: int | None) -> str | None:
    if dimension is None:
        return None
    return AXIS_BY_INDEX.get(dimension, str(dimension))


def format_import_report(
    facts: KernelFacts,
    *,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> str:
    lines = [
        f"# {facts.export_name or facts.function_name} MLIR Import Report",
        "",
    ]
    if source_path is not None:
        lines.append(f"Source MLIR: `{source_path}`")
    if output_path is not None:
        lines.append(f"Generated Loom: `{output_path}`")
    if source_path is not None or output_path is not None:
        lines.append("")
    lines += [
        "## Kernel Facts",
        "",
        f"- executable: `{facts.executable_name or '?'}`",
        f"- variant: `{facts.variant_name or '?'}`",
        f"- export: `{facts.export_name or facts.function_name}`",
        f"- target: `{facts.target_driver or '?'}` / `{facts.target_format or '?'}`",
        f"- workgroup_size: `{facts.workgroup_size or '?'}`",
        "- subgroup_size: "
        f"`{facts.subgroup_size if facts.subgroup_size is not None else '?'}`",
        "",
        "## Bindings",
        "",
    ]
    for binding in facts.bindings:
        parts = [
            f"- binding {binding.binding}: "
            f"`{binding.source_type}` -> `{binding.view_type}`",
        ]
        if binding.alignment:
            parts.append(f"alignment `{binding.alignment}`")
        if binding.flags:
            parts.append(f"flags `{binding.flags}`")
        lines.append(", ".join(parts))
    lines += [
        "",
        "## Converted Body Slice",
        "",
    ]
    if facts.converted_body.lines:
        lines.append("```loom")
        lines.extend(facts.converted_body.lines)
        lines.append("```")
    else:
        lines.append("No body ops were converted.")
    lines += [
        "",
        "## Converter Records",
        "",
    ]
    if facts.converted_body.converted:
        for record in facts.converted_body.converted:
            lines.append(f"- `{record.source}` -> `{' ; '.join(record.target)}`")
    else:
        lines.append("- none")
    if facts.converted_body.blocked:
        lines += [
            "",
            "## Blocked Body Ops",
            "",
        ]
        for record in facts.converted_body.blocked:
            lines.append(f"- `{record.source}` -> {record.target[0]}")
    if facts.converted_body.unsupported_counts:
        lines += [
            "",
            "## Unsupported Op Counts",
            "",
        ]
        for name, count in sorted(facts.converted_body.unsupported_counts.items()):
            lines.append(f"- `{name}`: {count}")
    return "\n".join(lines) + "\n"
