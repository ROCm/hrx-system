// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Parses and verifies Loom source text for the analysis tools.
//
// A session owns one LoomC context, configured with every built-in target IR
// dialect, and one workspace. Loading text produces a verified public module
// plus its read-only native view for analysis. loom-analyze loads every input
// once; loom-reduce also checks rewritten text in memory before writing it.

#ifndef LOOM_TOOLS_LOOM_ANALYZE_SESSION_H_
#define LOOM_TOOLS_LOOM_ANALYZE_SESSION_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loomc/loomc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_analyze_session_t loom_analyze_session_t;

// Creates a session with the configured target IR environment.
iree_status_t loom_analyze_session_create(iree_allocator_t allocator,
                                          loom_analyze_session_t** out_session);

// Releases the session. Modules it loaded must be released first.
void loom_analyze_session_free(loom_analyze_session_t* session);

// Parses and verifies |contents| as text named |filename|. |contents| is
// borrowed and must outlive |*out_module|. Diagnostics print to |diagnostics|
// when non-NULL and are discarded otherwise.
//
// Returns OK with |*out_accepted| false when the text has diagnostics; the
// module, if any, is still returned for release. On acceptance
// |*out_native_module| is the verified native view borrowed from the module.
iree_status_t loom_analyze_session_load(loom_analyze_session_t* session,
                                        iree_string_view_t filename,
                                        iree_const_byte_span_t contents,
                                        FILE* diagnostics,
                                        loomc_module_t** out_module,
                                        const loom_module_t** out_native_module,
                                        bool* out_accepted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_ANALYZE_SESSION_H_
