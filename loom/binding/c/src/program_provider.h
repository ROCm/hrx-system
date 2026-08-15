// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_PROVIDER_H_
#define LOOMC_PROGRAM_PROVIDER_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loomc/program_plan.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable provider of a linked program-root representation.
typedef struct loomc_program_provider_t {
  // Returns true when this provider owns |root_op| as a selectable root.
  //
  // Selection runs before provider-specific semantic verification. Providers
  // may inspect structurally safe operation identity and definition state but
  // cannot assume authored semantics have been verified.
  bool (*owns_root)(const loom_module_t* module, const loom_op_t* root_op);

  // Prepares one exact plan from explicitly selected roots in a sealed module.
  //
  // |root_ops| preserves caller order and points into |sealed_module|. The
  // sealed module contains no transient function-version sidecar and does not
  // borrow the source module. The returned plan retains everything needed for
  // later unit inspection and compilation.
  loomc_status_t (*prepare)(loomc_workspace_t* workspace,
                            const loomc_module_t* sealed_module,
                            const loom_op_t* const* root_ops,
                            loomc_host_size_t root_count,
                            const loomc_program_plan_options_t* options,
                            loomc_result_t* result, loomc_allocator_t allocator,
                            loomc_program_plan_t** out_program_plan);
} loomc_program_provider_t;

// Static program-provider table linked into one target environment product.
typedef struct loomc_program_provider_set_t {
  // Borrowed provider descriptors with process lifetime.
  const loomc_program_provider_t* const* values;

  // Number of entries in |values|.
  loomc_host_size_t count;
} loomc_program_provider_set_t;

// One provider and its exact selected roots.
typedef struct loomc_program_provider_selection_t {
  // Provider owning every selected root.
  const loomc_program_provider_t* provider;

  // Arena-owned root operations in caller order.
  const loom_op_t** root_ops;

  // Number of entries in |root_ops|.
  loomc_host_size_t root_count;
} loomc_program_provider_selection_t;

// Creates a borrowed provider-set view.
static inline loomc_program_provider_set_t loomc_program_provider_set_make(
    const loomc_program_provider_t* const* values, loomc_host_size_t count) {
  return (loomc_program_provider_set_t){
      .values = values,
      .count = count,
  };
}

// Resolves explicit root names and selects their common owning provider.
//
// Root operation storage is allocated from |arena|. Root names may optionally
// begin with '@'. Missing, duplicate, unowned, or mixed-provider roots fail
// loudly.
LOOMC_API_PRIVATE loomc_status_t loomc_program_provider_select_roots(
    const loomc_program_provider_set_t* provider_set,
    const loom_module_t* module, const loomc_string_view_t* root_names,
    loomc_host_size_t root_count, iree_arena_allocator_t* arena,
    loomc_program_provider_selection_t* out_selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_PROVIDER_H_
