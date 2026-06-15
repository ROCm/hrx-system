// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Symbol pruning policy shared by module-level symbol transforms.
//
// The policy names the module-boundary roots ordinary DCE must retain and the
// private symbols that may be erased when no liveness analysis reaches them.

#ifndef LOOM_TRANSFORMS_SYMBOL_SYMBOL_PRUNING_H_
#define LOOM_TRANSFORMS_SYMBOL_SYMBOL_PRUNING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_symbol_pruning_result_t {
  // Number of module symbols erased.
  uint32_t symbol_count;

  // Number of erased symbols implementing FuncLike.
  uint32_t function_like_count;
} loom_symbol_pruning_result_t;

typedef uint32_t loom_symbol_pruning_flags_t;

typedef enum loom_symbol_pruning_flag_bits_e {
  // Retain private source-level func/kernel entries with target records.
  LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES = 1u << 0,
} loom_symbol_pruning_flag_bits_e;

typedef struct loom_symbol_pruning_options_t {
  // Root and erasure policy flags.
  loom_symbol_pruning_flags_t flags;
} loom_symbol_pruning_options_t;

// Returns true when |symbol| is private module-local state that may be erased
// once no liveness edge reaches it.
bool loom_symbol_pruning_symbol_is_erasable(const loom_module_t* module,
                                            const loom_symbol_t* symbol);

// Root query for loom_symbol_liveness_compute using the ordinary symbol-DCE
// retention policy. The user data may be NULL or a
// loom_symbol_pruning_options_t pointer.
bool loom_symbol_pruning_symbol_is_root(void* user_data,
                                        const loom_module_t* module,
                                        loom_symbol_id_t symbol_id,
                                        const loom_symbol_t* symbol);

// Erases private symbols that are not live according to |liveness|.
iree_status_t loom_symbol_pruning_erase_unreachable(
    loom_module_t* module, const loom_symbol_liveness_t* liveness,
    const loom_symbol_pruning_options_t* options, iree_arena_allocator_t* arena,
    loom_symbol_pruning_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_SYMBOL_PRUNING_H_
