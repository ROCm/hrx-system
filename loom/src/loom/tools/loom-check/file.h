// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// File-level loom-check processing.

#ifndef LOOM_TOOLS_LOOM_CHECK_FILE_H_
#define LOOM_TOOLS_LOOM_CHECK_FILE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/json_output.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_check_process_options_t {
  // Rewrites expected sections and synchronized template cases in-place.
  bool update;
  // Prints PASS/FAIL/SKIP for every case, not just failures.
  bool verbose;
  // Emits structured JSON file results to stdout.
  bool json_enabled;
  // JSON result detail selected when |json_enabled| is true.
  loom_check_json_output_mode_t json_output_mode;
} loom_check_process_options_t;

// Reads a source from stdin or a file path, then processes it.
//
// |path| is "-" or empty for stdin, otherwise a filesystem path.
iree_status_t loom_check_read_and_process(
    iree_string_view_t path, const loom_check_process_options_t* options,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_host_size_t* pass_count, iree_host_size_t* fail_count,
    iree_host_size_t* skip_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_FILE_H_
