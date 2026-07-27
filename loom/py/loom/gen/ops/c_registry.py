# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C op registry generation for production dialect registration."""

from __future__ import annotations

from typing import Any

from loom.dsl import Op
from loom.gen.ops.c_names import GENERATED_HEADER, c_dialect_enum, c_dialect_include_path


def generate_op_registry(
    dialects: list[tuple[Any, list[Op]]],
) -> tuple[str, str, str]:
    """Generate op_registry.h, op_registry_tables.h, and op_registry_tables.c."""
    header = [GENERATED_HEADER]
    header.append("#ifndef LOOM_OPS_OP_REGISTRY_H_")
    header.append("#define LOOM_OPS_OP_REGISTRY_H_")
    header.append("")
    header.append('#include "iree/base/api.h"')
    header.append('#include "loom/ir/context.h"')
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append('extern "C" {')
    header.append("#endif")
    header.append("")
    header.append("// Registers production dialect vtables and built-in encoding families.")
    header.append("//")
    header.append("// The context must have been initialized and must not have been")
    header.append("// finalized yet. The test dialect is intentionally not registered here;")
    header.append("// developer tools and tests that need it must opt in explicitly.")
    header.append("iree_status_t loom_op_registry_register_all_dialects(")
    header.append("    loom_context_t* context);")
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append("}")
    header.append("#endif")
    header.append("")
    header.append("#endif  // LOOM_OPS_OP_REGISTRY_H_")
    header.append("")

    tables_header = [GENERATED_HEADER]
    tables_header.append("#ifndef LOOM_OPS_OP_REGISTRY_TABLES_H_")
    tables_header.append("#define LOOM_OPS_OP_REGISTRY_TABLES_H_")
    tables_header.append("")
    tables_header.append('#include "loom/ops/op_registry.h"')
    tables_header.append('#include "loom/ops/op_defs.h"')
    tables_header.append("")
    tables_header.append("typedef const loom_op_vtable_t* const* (*")
    tables_header.append("    loom_op_registry_dialect_vtables_fn_t)(")
    tables_header.append("    iree_host_size_t* out_count);")
    tables_header.append("")
    tables_header.append("typedef const loom_op_semantics_t* (*")
    tables_header.append("    loom_op_registry_dialect_semantics_fn_t)(")
    tables_header.append("    iree_host_size_t* out_count);")
    tables_header.append("")
    tables_header.append("typedef struct loom_op_registry_dialect_registration_t {")
    tables_header.append("  loom_dialect_id_t dialect_id;")
    tables_header.append("  loom_op_registry_dialect_vtables_fn_t vtables_fn;")
    tables_header.append("  loom_op_registry_dialect_semantics_fn_t semantics_fn;")
    tables_header.append("} loom_op_registry_dialect_registration_t;")
    tables_header.append("")
    tables_header.append("extern const loom_op_registry_dialect_registration_t")
    tables_header.append("    loom_op_registry_dialects[];")
    tables_header.append("extern const iree_host_size_t loom_op_registry_dialect_count;")
    tables_header.append("")
    tables_header.append("#endif  // LOOM_OPS_OP_REGISTRY_TABLES_H_")
    tables_header.append("")

    source = [GENERATED_HEADER]
    source.append('#include "loom/ops/op_registry_tables.h"')
    source.append("")
    source.extend(f'#include "{c_dialect_include_path(dialect)}/ops.h"' for dialect, _ops in sorted(dialects, key=lambda item: item[0].name))
    source.append("")
    source.append("const loom_op_registry_dialect_registration_t")
    source.append("    loom_op_registry_dialects[] = {")
    for dialect, _ops in sorted(dialects, key=lambda item: item[0].dialect_id):
        source.append(f"    {{{c_dialect_enum(dialect.name)}, loom_{dialect.name}_dialect_vtables, loom_{dialect.name}_dialect_op_semantics}},")
    source.append("};")
    source.append("")
    source.append("const iree_host_size_t loom_op_registry_dialect_count =")
    source.append(f"    {len(dialects)};")
    source.append("")

    return "\n".join(header), "\n".join(tables_header), "\n".join(source)
