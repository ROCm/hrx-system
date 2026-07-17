// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target capability facts exposed through value-fact contexts.

#ifndef LOOM_TARGET_CAPABILITY_FACTS_H_
#define LOOM_TARGET_CAPABILITY_FACTS_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_fact_context_t loom_fact_context_t;

// Queries an unsigned integer target capability fact from |context|.
//
// The generic "target" namespace is backed by loom_target_snapshot_t fields.
// Target-family namespaces are owned by their target implementations and must
// not be inferred from target names by generic compiler code.
bool loom_target_fact_context_query_u64(const loom_fact_context_t* context,
                                        iree_string_view_t namespace_name,
                                        iree_string_view_t key,
                                        uint64_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CAPABILITY_FACTS_H_
