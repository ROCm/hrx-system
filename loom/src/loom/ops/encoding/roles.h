// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Semantic role queries for encoding values.

#ifndef LOOM_OPS_ENCODING_ROLES_H_
#define LOOM_OPS_ENCODING_ROLES_H_

#include "loom/ir/encoding.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a diagnostic-facing description of |role|.
iree_string_view_t loom_encoding_role_description(loom_encoding_role_t role);

// Returns the semantic role of a static encoding instance.
loom_encoding_role_t loom_encoding_static_role(const loom_module_t* module,
                                               const loom_encoding_t* encoding);

// Returns the semantic role of an SSA value with type `encoding`.
loom_encoding_role_t loom_encoding_value_role(const loom_module_t* module,
                                              loom_value_id_t value_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_ROLES_H_
