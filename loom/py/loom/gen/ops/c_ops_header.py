# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C ops.h declaration generation for Loom dialect ops."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dsl import ATTR_TYPE_FLAGS, Op
from loom.fields import compute_layout
from loom.gen.ops import c_builders
from loom.gen.ops.c_enum_attrs import collect_shared_enums as _collect_shared_enums
from loom.gen.ops.c_enum_attrs import enum_c_type as _enum_c_type
from loom.gen.ops.c_enum_attrs import enum_case_c_ident as _enum_case_c_ident
from loom.gen.ops.c_names import COPYRIGHT
from loom.gen.ops.c_names import c_dialect_enum as _c_dialect_enum
from loom.gen.ops.c_names import c_enum_name as _c_enum_name
from loom.gen.ops.c_names import c_prefix as _c_prefix
from loom.gen.ops.c_names import guard_name as _guard_name
from loom.gen.support.generated_file import line_comment_header


def generate_ops_h(dialect_name: str, dialect_id: int, ops: Sequence[Op]) -> str:
    """Generates the ops.h header for a dialect."""
    lines: list[str] = []
    guard = _guard_name(dialect_name)
    dialect_enum = _c_dialect_enum(dialect_name)
    shared_enums = _collect_shared_enums(dialect_name, ops)

    lines.append(COPYRIGHT)
    lines.extend(
        line_comment_header(
            "//",
            generator="loom.gen.ops.c_tables",
            regenerate="python3 loom/py/loom/gen/run.py c_tables --in-place",
        )
    )
    lines.append("// clang-format off")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append('#include "loom/ops/op_defs.h"')
    enum_includes = sorted(
        {attr_def.enum_def.c_include for op in ops for attr_def in op.attrs if attr_def.attr_type == "enum" and attr_def.enum_def is not None and attr_def.enum_def.c_include is not None}
    )
    lines.extend(f'#include "{include}"' for include in enum_includes)
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")

    # Op kind enum.
    lines.append("enum {")
    for i, op in enumerate(ops):
        enum_name = _c_enum_name(op)
        lines.append(f"  {enum_name} = LOOM_OP_KIND({dialect_enum}, {i}),")
    lines.append(f"  LOOM_OP_{dialect_name.upper()}_COUNT_ = {len(ops)},")
    lines.append("};")

    lines.append("")

    # Flag bit defines — emitted once per unique flags enum.
    emitted_flag_enums: set[str] = set()
    for op in ops:
        for attr_def in op.attrs:
            if attr_def.attr_type != ATTR_TYPE_FLAGS or attr_def.enum_def is None:
                continue
            if attr_def.enum_def.name in emitted_flag_enums:
                continue
            emitted_flag_enums.add(attr_def.enum_def.name)
            enum_prefix = "LOOM_" + op.namespace.upper() + "_" + attr_def.enum_def.name.upper()
            if attr_def.enum_def.doc:
                lines.append(f"// {attr_def.enum_def.doc}")
            lines.extend(f"#define {enum_prefix}_{_enum_case_c_ident(case.keyword)} ((uint8_t){case.value})" for case in attr_def.enum_def.cases)
            lines.append("")

    # Enum attr C enums. When the same EnumDef object is shared by
    # multiple ops (e.g., CallingConv used by func.def, func.decl,
    # func.template, func.ukernel), emit it once with a dialect-level
    # name (loom_func_cc_t) instead of duplicating per-op.
    open_enum_ids = {id(attr_def.enum_def) for op in ops for attr_def in op.attrs if (attr_def.attr_type == "enum" and attr_def.open_enum and attr_def.enum_def is not None)}

    # Emit shared enums first.
    for enum_id, (c_prefix, const_prefix, enum_def) in shared_enums.items():
        if enum_def.doc:
            lines.append(f"// {enum_def.doc}")
        max_value = max(c.value for c in enum_def.cases)
        if enum_id in open_enum_ids:
            lines.append(f"typedef uint8_t {c_prefix}_t;")
            lines.append(f"typedef enum {c_prefix}_e {{")
            lines.extend(f"  {const_prefix}_{_enum_case_c_ident(case.keyword)} = {case.value}," for case in enum_def.cases)
            lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
            lines.append(f"}} {c_prefix}_e;")
        else:
            lines.append(f"typedef enum {c_prefix}_e {{")
            lines.extend(f"  {const_prefix}_{_enum_case_c_ident(case.keyword)} = {case.value}," for case in enum_def.cases)
            lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
            lines.append(f"}} {c_prefix}_t;")
        lines.append("")

    # Emit per-op enums (only for EnumDefs not already emitted as shared).
    emitted_enum_defs: set[str] = set()
    for op in ops:
        for attr_def in op.attrs:
            if attr_def.attr_type != "enum" or attr_def.enum_def is None:
                continue
            if attr_def.enum_def.c_type is not None:
                continue
            if id(attr_def.enum_def) in shared_enums:
                continue
            key = f"{op.name}:{attr_def.name}"
            if key in emitted_enum_defs:
                continue
            emitted_enum_defs.add(key)
            c_prefix = _c_prefix(op) + "_" + attr_def.name
            enum_tag = c_prefix + "_e"
            const_prefix = c_prefix.upper()
            if attr_def.enum_def.doc:
                lines.append(f"// {attr_def.enum_def.doc}")
            max_value = max(c.value for c in attr_def.enum_def.cases)
            if attr_def.open_enum:
                lines.append(f"typedef uint8_t {c_prefix}_t;")
                lines.append(f"typedef enum {enum_tag} {{")
                lines.extend(f"  {const_prefix}_{_enum_case_c_ident(case.keyword)} = {case.value}," for case in attr_def.enum_def.cases)
                lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
                lines.append(f"}} {enum_tag};")
            else:
                lines.append(f"typedef enum {enum_tag} {{")
                lines.extend(f"  {const_prefix}_{_enum_case_c_ident(case.keyword)} = {case.value}," for case in attr_def.enum_def.cases)
                lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
                lines.append(f"}} {c_prefix}_t;")
            lines.append("")

    # Per-op sections.
    emitted_canonicalize_declarations: set[str] = set()
    emitted_type_transfer_declarations: set[str] = set()
    for op in ops:
        prefix = _c_prefix(op)
        enum_name = _c_enum_name(op)
        layout = compute_layout(op)

        # Assembly format comment from first example, or synthesized.
        if op.examples:
            asm_fmt_lines = op.examples[0].split("\n")
        else:
            asm_fmt_lines = [op.name]
        doc = op.doc or f"{op.name} operation."

        lines.append(f"// {enum_name}: {doc}")
        lines.extend(f"// {line}" for line in asm_fmt_lines)

        # ISA check.
        lines.append(f"LOOM_DEFINE_ISA({prefix}_isa, {enum_name})")

        # Accessors.
        for operand in op.operands:
            desc = layout.fields[operand.name]
            if layout.segmented_operands:
                if operand.variadic:
                    lines.append(f"LOOM_DEFINE_SEGMENTED_OPERANDS({prefix}_{operand.name}, {desc.index})")
                elif operand.optional:
                    lines.append(f"LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND({prefix}_{operand.name}, {desc.index})")
                else:
                    lines.append(f"LOOM_DEFINE_SEGMENTED_OPERAND({prefix}_{operand.name}, {desc.index})")
            elif operand.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_OPERANDS({prefix}_{operand.name}, {desc.index})")
            elif operand.optional:
                lines.append(f"LOOM_DEFINE_OPTIONAL_OPERAND({prefix}_{operand.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_OPERAND({prefix}_{operand.name}, {desc.index})")

        for result in op.results:
            desc = layout.fields[result.name]
            if result.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_RESULTS({prefix}_{result.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_RESULT({prefix}_{result.name}, {desc.index})")

        for successor in op.successors:
            desc = layout.fields[successor.name]
            if successor.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_SUCCESSORS({prefix}_{successor.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_SUCCESSOR({prefix}_{successor.name}, {desc.index})")

        # Regular attribute accessors (excludes flags attrs).
        non_flags_index = 0
        for attr_def in op.attrs:
            if attr_def.attr_type == ATTR_TYPE_FLAGS:
                lines.append(f"LOOM_DEFINE_INSTANCE_FLAGS({prefix}_{attr_def.name})")
                continue
            desc_index = non_flags_index
            non_flags_index += 1
            macro_map = {
                "i64": "LOOM_DEFINE_ATTR_I64",
                "f64": "LOOM_DEFINE_ATTR_F64",
                "string": "LOOM_DEFINE_ATTR_STRING",
                "bool": "LOOM_DEFINE_ATTR_BOOL",
                "i64_array": "LOOM_DEFINE_ATTR_I64_ARRAY",
                "bytes": "LOOM_DEFINE_ATTR_BYTES",
                "predicate_list": "LOOM_DEFINE_ATTR_PREDICATE_LIST",
                "dict": "LOOM_DEFINE_ATTR_DICT",
                "encoding": "LOOM_DEFINE_ATTR_ENCODING",
                "enum": "LOOM_DEFINE_ATTR_ENUM",
                "symbol": "LOOM_DEFINE_ATTR_SYMBOL",
                "type": "LOOM_DEFINE_ATTR_TYPE",
                "any": "LOOM_DEFINE_ATTR_ANY",
            }
            macro = macro_map.get(attr_def.attr_type)
            if attr_def.attr_type == "enum" and attr_def.enum_def:
                enum_type = _enum_c_type(op, attr_def, shared_enums)
                lines.append(f"LOOM_DEFINE_ATTR_ENUM_TYPED({prefix}_{attr_def.name}, {desc_index}, {enum_type})")
            elif macro:
                lines.append(f"{macro}({prefix}_{attr_def.name}, {desc_index})")

        for region_def in op.regions:
            desc = layout.fields[region_def.name]
            if region_def.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_REGIONS({prefix}_{region_def.name}, {desc.index})")
            elif region_def.optional:
                lines.append(f"LOOM_DEFINE_OPTIONAL_REGION({prefix}_{region_def.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_REGION({prefix}_{region_def.name}, {desc.index})")

        # Builder declaration.
        lines.extend(c_builders.generate_builder_header_lines(op, shared_enums))

        # Canonicalize function declaration (hand-written, linked in).
        if op.canonicalize and op.canonicalize not in emitted_canonicalize_declarations:
            lines.append(f"iree_status_t {op.canonicalize}(loom_op_t* op, loom_rewriter_t* rewriter);")
            emitted_canonicalize_declarations.add(op.canonicalize)

        # Effective traits function declaration (hand-written, linked in).
        if op.effective_traits:
            lines.append(f"loom_trait_flags_t {op.effective_traits}(const loom_op_t* op);")

        # Fact inference function declaration (hand-written, linked in).
        if op.facts:
            lines.append(f"iree_status_t {op.facts}(")
            lines.append("    loom_fact_context_t* context,")
            lines.append("    const loom_module_t* module, const loom_op_t* op,")
            lines.append("    const loom_value_facts_t* operand_facts,")
            lines.append("    loom_value_facts_t* result_facts);")

        # Semantic type-transfer function declaration (hand-written, linked in).
        if op.type_transfer and op.type_transfer not in emitted_type_transfer_declarations:
            lines.append(f"iree_status_t {op.type_transfer}(")
            lines.append("    loom_type_transfer_context_t* context,")
            lines.append("    const loom_module_t* module, loom_op_t* op);")
            emitted_type_transfer_declarations.add(op.type_transfer)

        # Verify function declaration (hand-written, linked in).
        if op.verify:
            lines.append(f"iree_status_t {op.verify}(")
            lines.append("    const loom_module_t* module, const loom_op_t* op,")
            lines.append("    iree_diagnostic_emitter_t emitter);")

        lines.append("")

    # Registration function.
    lines.append(f"// Returns the vtable array for the {dialect_name} dialect.")
    lines.append(f"const loom_op_vtable_t* const* loom_{dialect_name}_dialect_vtables(")
    lines.append("    iree_host_size_t* out_count);")
    lines.append("")

    lines.append(f"// Returns the dense semantic metadata array for the {dialect_name} dialect.")
    lines.append(f"const loom_op_semantics_t* loom_{dialect_name}_dialect_op_semantics(")
    lines.append("    iree_host_size_t* out_count);")
    lines.append("")
    lines.append(f"// Returns semantic metadata for a {dialect_name} op kind, or empty metadata.")
    lines.append(f"loom_op_semantics_t loom_{dialect_name}_op_semantics(")
    lines.append("    loom_op_kind_t kind);")
    lines.append("")

    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")

    return "\n".join(lines)
