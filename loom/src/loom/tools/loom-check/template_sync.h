// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Template synchronization for loom-check target expectation suites.
//
// A target suite may declare a file-level `// TEMPLATE: <path>` directive. In
// --update mode, loom-check uses the template's cases as the authoritative case
// list while preserving the target suite's file-level RUN directive and any
// case-local REQUIRES/XFAIL directives that already exist in the target file.
// Case identity is the single func-like definition in each case body.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEMPLATE_SYNC_H_
#define LOOM_TOOLS_LOOM_CHECK_TEMPLATE_SYNC_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/tools/loom-check/check.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rebuilds |target_source| from |template_source| when |target_file| declares a
// TEMPLATE directive.
//
// The target file preamble is preserved, including its authoritative RUN line.
// Template RUN lines and expected sections are not copied. Existing target
// cases with matching func-like definitions keep their expected section,
// case-local REQUIRES/XFAIL directives, and diagnostic annotations; stale
// target-only cases are omitted. The rebuilt text is written into |new_source|;
// |*out_changed| reports whether it differs from |target_source|.
iree_status_t loom_check_template_sync_build_source(
    iree_string_view_t target_source, const loom_check_file_t* target_file,
    iree_string_view_t target_filename, iree_string_view_t template_source,
    iree_string_view_t template_filename, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    iree_allocator_t host_allocator, iree_string_builder_t* new_source,
    bool* out_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEMPLATE_SYNC_H_
