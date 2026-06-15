# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU dense target reference constants and maps."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_common_reg_class_ids,
    amdgpu_descriptor_ref_keys,
    amdgpu_immediate_encoding_id_items,
    build_amdgpu_core_descriptor_set_from_spec,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import target_relative_name  # noqa: E402


def _parse_isa_xml_argument(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition(":")
    if not separator or not key or not path:
        raise ValueError("AMDGPU target-ref --isa-xml entries must be key:path pairs")
    return key, Path(path)


def _parse_isa_xml_arguments(values: Sequence[str]) -> dict[str, AmdgpuIsaFactSource]:
    paths: dict[str, Path] = {}
    specs: dict[str, AmdgpuIsaFactSource] = {}
    for value in values:
        key, path = _parse_isa_xml_argument(value)
        if key in paths:
            if paths[key] != path:
                raise ValueError(f"AMDGPU target-ref ISA XML key '{key}' has conflicting paths '{paths[key]}' and '{path}'")
            continue
        paths[key] = path
        specs[key] = parse_amdgpu_isa_xml_path(path)
    return specs


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _descriptor_ref_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{_c_identifier(target_relative_name('amdgpu', key))}"


def _reg_class_id_constant_name(reg_class_name: str) -> str:
    return f"LOOM_AMDGPU_REG_CLASS_ID_{_c_identifier(target_relative_name('amdgpu', reg_class_name))}"


def _immediate_encoding_id_constant_name(name: str) -> str:
    return f"LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_{_c_identifier(name)}"


def _u16_literal(value: int) -> str:
    return f"UINT16_C({value})"


def _descriptor_set_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    return f"kAmdgpu{''.join(part[:1].upper() + part[1:] for part in suffix.split('.') if part)}DescriptorRefOrdinals"


def _materialize_descriptor_ref_tables(
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
    descriptor_set_keys: Sequence[str],
) -> list[tuple[int, str, list[int | None]]]:
    if not descriptor_set_keys:
        raise ValueError("AMDGPU target-ref source generation requires at least one --descriptor-set")
    descriptor_ref_keys = amdgpu_descriptor_ref_keys()
    descriptor_set_infos_by_key = {info.key: info for info in sorted_descriptor_set_infos()}
    descriptor_set_tables: list[tuple[int, str, list[int | None]]] = []
    for descriptor_set_key in descriptor_set_keys:
        try:
            descriptor_set_info = descriptor_set_infos_by_key[descriptor_set_key]
        except KeyError as exc:
            raise ValueError(f"AMDGPU target-ref generator got unknown descriptor set '{descriptor_set_key}'") from exc
        try:
            spec = isa_specs[descriptor_set_info.isa_xml_key]
        except KeyError as exc:
            raise ValueError(f"AMDGPU target-ref generator is missing ISA XML key '{descriptor_set_info.isa_xml_key}' for descriptor set '{descriptor_set_info.key}'") from exc
        descriptor_set = build_amdgpu_core_descriptor_set_from_spec(
            descriptor_set_info.generator_target,
            spec,
        )
        if descriptor_set.key != descriptor_set_info.key:
            raise ValueError(f"AMDGPU descriptor-set builder '{descriptor_set_info.generator_target}' produced '{descriptor_set.key}', expected '{descriptor_set_info.key}'")
        descriptor_ordinals = {descriptor.key: ordinal for ordinal, descriptor in enumerate(descriptor_set.descriptors)}
        descriptor_set_tables.append(
            (
                amdgpu_descriptor_set_ordinal(descriptor_set_info.key),
                descriptor_set_info.key,
                [descriptor_ordinals.get(key) for key in descriptor_ref_keys],
            )
        )
    descriptor_set_tables.sort()
    return descriptor_set_tables


def _emit_header() -> str:
    descriptor_ref_keys = amdgpu_descriptor_ref_keys()
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.refs.amdgpu_target_refs"),
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_",
        "",
        "#include <stdint.h>",
        "",
        '#include "loom/codegen/low/descriptors.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "typedef uint16_t loom_amdgpu_descriptor_ref_t;",
        "",
        "#define LOOM_AMDGPU_DESCRIPTOR_REF_NONE UINT16_MAX",
        f"#define LOOM_AMDGPU_DESCRIPTOR_REF_COUNT {_u16_literal(len(descriptor_ref_keys))}",
        "",
    ]
    lines.extend(f"#define {_descriptor_ref_constant_name(key)} {_u16_literal(index)}" for index, key in enumerate(descriptor_ref_keys))
    lines.append("")
    lines.extend(f"#define {_reg_class_id_constant_name(reg_class_name)} {reg_class_id}u" for reg_class_name, reg_class_id in amdgpu_common_reg_class_ids())
    lines.append("")
    lines.extend(f"#define {_immediate_encoding_id_constant_name(name)} {encoding_id}u" for name, encoding_id in amdgpu_immediate_encoding_id_items())
    lines.extend(
        [
            "",
            "uint32_t loom_amdgpu_descriptor_ref_ordinal(",
            "    const loom_low_descriptor_set_t* descriptor_set,",
            "    loom_amdgpu_descriptor_ref_t descriptor_ref);",
            "",
            "const loom_low_descriptor_t* loom_amdgpu_descriptor_ref_descriptor(",
            "    const loom_low_descriptor_set_t* descriptor_set,",
            "    loom_amdgpu_descriptor_ref_t descriptor_ref);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            "#endif  // LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_source(
    *,
    public_header: str,
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
    descriptor_set_keys: Sequence[str],
) -> str:
    descriptor_set_tables = _materialize_descriptor_ref_tables(isa_specs, descriptor_set_keys)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.refs.amdgpu_target_refs"),
        "",
        f'#include "{public_header}"',
        "",
    ]
    for _, descriptor_set_key, descriptor_ordinals in descriptor_set_tables:
        table_name = _descriptor_set_table_name(descriptor_set_key)
        lines.append(f"static const uint32_t {table_name}[] = {{")
        for descriptor_ordinal in descriptor_ordinals:
            if descriptor_ordinal is None:
                lines.append("    LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,")
            else:
                lines.append(f"    {descriptor_ordinal}u,")
        lines.append("};")
        lines.append("")

    max_descriptor_set_ordinal = max(descriptor_set_ordinal for descriptor_set_ordinal, _, _ in descriptor_set_tables)
    tables_by_ordinal = {descriptor_set_ordinal: descriptor_set_key for descriptor_set_ordinal, descriptor_set_key, _ in descriptor_set_tables}
    lines.append("static const uint32_t* const kAmdgpuDescriptorRefOrdinalTables[] = {")
    for descriptor_set_ordinal in range(max_descriptor_set_ordinal + 1):
        ordinal_descriptor_set_key = tables_by_ordinal.get(descriptor_set_ordinal)
        table_expr = _descriptor_set_table_name(ordinal_descriptor_set_key) if ordinal_descriptor_set_key is not None else "NULL"
        lines.append(f"    {table_expr},")
    lines.append("};")
    lines.append("")
    lines.extend(
        [
            "uint32_t loom_amdgpu_descriptor_ref_ordinal(",
            "    const loom_low_descriptor_set_t* descriptor_set,",
            "    loom_amdgpu_descriptor_ref_t descriptor_ref) {",
            "  if (descriptor_set == NULL ||",
            "      descriptor_ref >= LOOM_AMDGPU_DESCRIPTOR_REF_COUNT) {",
            "    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;",
            "  }",
            "  const uint16_t descriptor_set_ordinal =",
            "      descriptor_set->descriptor_set_ordinal;",
            "  if (descriptor_set_ordinal >=",
            "      sizeof(kAmdgpuDescriptorRefOrdinalTables) /",
            "          sizeof(kAmdgpuDescriptorRefOrdinalTables[0])) {",
            "    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;",
            "  }",
            "  const uint32_t* descriptor_ordinals =",
            "      kAmdgpuDescriptorRefOrdinalTables[descriptor_set_ordinal];",
            "  if (descriptor_ordinals == NULL) {",
            "    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;",
            "  }",
            "  return descriptor_ordinals[descriptor_ref];",
            "}",
            "",
            "const loom_low_descriptor_t* loom_amdgpu_descriptor_ref_descriptor(",
            "    const loom_low_descriptor_set_t* descriptor_set,",
            "    loom_amdgpu_descriptor_ref_t descriptor_ref) {",
            "  return loom_low_descriptor_set_descriptor_at(",
            "      descriptor_set,",
            "      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref));",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU dense target reference constants and maps.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated target-ref header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated target-ref source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/refs/target_refs.h",
        help="Public include path for the generated header.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as key:path.",
    )
    parser.add_argument(
        "--descriptor-set",
        action="append",
        default=[],
        help="Descriptor-set key to materialize into the generated source.",
    )
    args = parser.parse_args(argv)

    isa_specs = _parse_isa_xml_arguments(args.isa_xml)
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(
        _emit_header(),
        encoding="utf-8",
    )
    args.source.write_text(
        _emit_source(
            public_header=args.public_header,
            isa_specs=isa_specs,
            descriptor_set_keys=args.descriptor_set,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
