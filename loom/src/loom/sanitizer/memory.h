// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SANITIZER_MEMORY_H_
#define LOOM_SANITIZER_MEMORY_H_

#include <stdbool.h>

#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

// Queries the target-independent memory space of |view| from the current
// analysis facts. Returns false when no usable view-reference fact is known.
bool loom_sanitizer_query_view_memory_space(
    const loom_rewriter_t* rewriter, loom_value_id_t view,
    loom_value_fact_memory_space_t* out_memory_space);

// Returns true when access assertions may observe memory in |memory_space|.
bool loom_sanitizer_access_memory_space_is_observable(
    loom_value_fact_memory_space_t memory_space);

// Returns true when access assertions may observe |view|. Unknown memory-space
// facts are conservatively observable so target lowering can diagnose or lower
// assertions created by frontends that do not carry precise storage facts.
bool loom_sanitizer_access_view_is_observable(const loom_rewriter_t* rewriter,
                                              loom_value_id_t view);

#endif  // LOOM_SANITIZER_MEMORY_H_
