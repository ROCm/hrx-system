// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage-dependence propagation for kernel launch configurations and their
// required-inline helper summaries.

#ifndef LOOM_OPS_KERNEL_LAUNCH_CONFIG_DEPENDENCE_H_
#define LOOM_OPS_KERNEL_LAUNCH_CONFIG_DEPENDENCE_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_kernel_launch_config_dependence_t {
  // Module owning every tracked SSA value and maintained use index.
  const loom_module_t* module;
  // Caller-owned storage for membership and worklist growth.
  iree_arena_allocator_t* arena;
  // Module-value-indexed dependence membership bits.
  uint64_t* dependent_bits;
  // Number of entries in dependent_bits.
  iree_host_size_t dependent_word_count;
  // Newly dependent values awaiting indexed user propagation.
  loom_value_id_t* dependent_values;
  // Number of initialized entries in dependent_values.
  iree_host_size_t dependent_value_count;
  // Allocated entry capacity in dependent_values.
  iree_host_size_t dependent_value_capacity;
} loom_kernel_launch_config_dependence_t;

// Returns whether no storage-derived values have been discovered.
static inline bool loom_kernel_launch_config_dependence_is_empty(
    const loom_kernel_launch_config_dependence_t* dependence) {
  return dependence->dependent_value_count == 0;
}

// Initializes an empty dependence set over |module|. Storage grows lazily
// except for the directly indexed membership bitmap.
iree_status_t loom_kernel_launch_config_dependence_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_kernel_launch_config_dependence_t* out_dependence);

// Seeds one valid module-local SSA value. Duplicate and invalid values are
// ignored so callers may feed optional operation results directly.
iree_status_t loom_kernel_launch_config_dependence_mark(
    loom_kernel_launch_config_dependence_t* dependence,
    loom_value_id_t value_id);

// Seeds every result of |op|.
iree_status_t loom_kernel_launch_config_dependence_mark_results(
    loom_kernel_launch_config_dependence_t* dependence, const loom_op_t* op);

// Propagates all current seeds once through maintained SSA uses, type and
// attribute indexes, and structured value relations. Newly discovered values
// extend the same worklist, so the result is the complete reachable closure.
iree_status_t loom_kernel_launch_config_dependence_propagate(
    loom_kernel_launch_config_dependence_t* dependence);

// Returns whether |value_id| belongs to the propagated dependence set.
bool loom_kernel_launch_config_dependence_contains(
    const loom_kernel_launch_config_dependence_t* dependence,
    loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_KERNEL_LAUNCH_CONFIG_DEPENDENCE_H_
