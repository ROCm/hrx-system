// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_REFERENCE_TYPE_H_
#define LOOM_OPS_REFERENCE_TYPE_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves an authored declaration through its generated type contract. False
// means the symbol is undefined or lacks its two required type-key strings.
// Used at construction and verification boundaries; indexed fields require no
// traversal or repeated validation of string contents.
bool loom_reference_type_declaration_key(const loom_module_t* module,
                                         loom_symbol_ref_t declaration,
                                         loom_string_id_t* out_namespace_id,
                                         loom_string_id_t* out_name_id);

// Constructs and interns a managed handle at the source/builder boundary.
// An unresolved or non-type declaration is an invalid source argument.
iree_status_t loom_reference_type_make(loom_module_t* module,
                                       loom_reference_ownership_t ownership,
                                       loom_symbol_ref_t declaration,
                                       loom_type_t* out_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_REFERENCE_TYPE_H_
