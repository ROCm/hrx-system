// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler product, root, target, and format request resolution.

#ifndef LOOM_COMPILE_REQUEST_H_
#define LOOM_COMPILE_REQUEST_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/target/selection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Language-level product selected from compile roots.
typedef enum loom_compile_product_e {
  LOOM_COMPILE_PRODUCT_INVALID = 0,
  LOOM_COMPILE_PRODUCT_KERNEL = 1,
  LOOM_COMPILE_PRODUCT_COMMAND = 2,
  LOOM_COMPILE_PRODUCT_MODULE = 3,
} loom_compile_product_t;

// Returns the stable public name of |product|.
iree_string_view_t loom_compile_product_name(loom_compile_product_t product);

// Resolved language-level product and roots for one compilation.
typedef struct loom_compile_product_selection_t {
  // Product inferred from or constrained by the selected roots.
  loom_compile_product_t product;
  // Selected roots, preserving explicit order/duplicates. Derived roots own
  // their names in the caller arena. Empty selects the entire module.
  iree_string_view_list_t roots;
  // Common target family authored on selected kernel roots, or NULL.
  const loom_target_fact_type_t* target_fact_type;
  // Number of selected kernel roots without an authored target.
  iree_host_size_t untargeted_kernel_count;
} loom_compile_product_selection_t;

// Explicit target selected for one compile request.
typedef struct loom_compile_target_selection_t {
  // Immutable structured target profile selected from the environment.
  const loom_target_profile_t* profile;
  // Borrowed family and selector spelling supplied by the caller.
  loom_target_specification_t specification;
} loom_compile_target_selection_t;

// User constraints applied while resolving one compilation request.
typedef struct loom_compile_request_options_t {
  // Explicit root names, or an empty list to derive selection from the module.
  iree_string_view_list_t roots;
  // Optional product selection or explicit-root assertion.
  iree_string_view_t product;
  // Optional exact artifact format.
  iree_string_view_t format;
  // Optional family-qualified target profile.
  iree_string_view_t target;
  // Canonical root names to exclude after product inference and before
  // specialization and materialization. Cannot be combined with |roots|.
  iree_string_view_list_t excluded_roots;
} loom_compile_request_options_t;

// Fully resolved compile request borrowing immutable configured state.
typedef struct loom_compile_request_t {
  // Language-level product and root selection.
  loom_compile_product_selection_t selection;
  // Target-owned artifact emitter for kernel or module products. Command
  // products have no target emitter.
  const loom_target_emitter_t* target_emitter;
  // Explicit target selected by the caller, or empty for authored targets.
  loom_compile_target_selection_t explicit_target;
} loom_compile_request_t;

// Returns true when portable command emission was selected.
static inline bool loom_compile_request_is_command(
    const loom_compile_request_t* request) {
  return request != NULL &&
         request->selection.product == LOOM_COMPILE_PRODUCT_COMMAND;
}

// Resolves one homogeneous product and its compile roots, an optional explicit
// target, and a target emitter. Explicit roots are borrowed. Otherwise the
// product selects its complete default root set; an omitted product infers
// command, kernel, then module. Exclusions apply after inference and derived
// names are copied into |arena|. Emitter resolution never probes an emitter by
// compiling. An omitted format selects the target family's unique canonical
// kernel or module emitter, or the target-independent command format.
iree_status_t loom_compile_request_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    const loom_target_environment_t* target_environment,
    iree_arena_allocator_t* arena, loom_compile_request_t* out_request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_COMPILE_REQUEST_H_
