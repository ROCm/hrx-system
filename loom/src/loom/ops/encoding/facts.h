// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared fact construction for encoding values.

#ifndef LOOM_OPS_ENCODING_FACTS_H_
#define LOOM_OPS_ENCODING_FACTS_H_

#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds exact value facts for a verified static encoding specification.
// |result_type| supplies the role carried by the SSA value.
iree_status_t loom_encoding_static_value_facts(loom_fact_context_t* context,
                                               const loom_module_t* module,
                                               uint16_t encoding_id,
                                               loom_type_t result_type,
                                               loom_value_facts_t* out_facts);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_FACTS_H_
