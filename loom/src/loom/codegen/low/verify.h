// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-local verification for target-low functions.
//
// This verifier is intentionally separate from the generic structural verifier:
// it needs an explicit linked descriptor registry selected by the caller, while
// generic IR verification must stay target-package agnostic.

#ifndef LOOM_CODEGEN_LOW_VERIFY_H_
#define LOOM_CODEGEN_LOW_VERIFY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_verify_context_t loom_low_verify_context_t;
typedef struct loom_low_verify_module_context_t
    loom_low_verify_module_context_t;
typedef struct loom_low_verify_provider_t loom_low_verify_provider_t;

typedef iree_status_t (*loom_low_verify_provider_begin_module_fn_t)(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_module_context_t* context, void** out_provider_state);
typedef iree_status_t (*loom_low_verify_provider_begin_function_fn_t)(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state);
typedef iree_status_t (*loom_low_verify_provider_verify_op_fn_t)(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_resolved_descriptor_packet_t* packet);
typedef iree_status_t (*loom_low_verify_provider_end_function_fn_t)(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state);
typedef iree_status_t (*loom_low_verify_provider_end_module_fn_t)(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_module_context_t* context, void* provider_state);

struct loom_low_verify_provider_t {
  // Stable provider name available to callbacks.
  iree_string_view_t name;
  // Initializes target-owned module-level verification state.
  loom_low_verify_provider_begin_module_fn_t begin_module;
  // Initializes target-owned function-local verification state.
  loom_low_verify_provider_begin_function_fn_t begin_function;
  // Verifies one op as part of the generic low verifier's body walk.
  loom_low_verify_provider_verify_op_fn_t verify_op;
  // Finalizes target-owned function-local verification state.
  loom_low_verify_provider_end_function_fn_t end_function;
  // Finalizes target-owned module-level verification state.
  loom_low_verify_provider_end_module_fn_t end_module;
};

typedef struct loom_low_verify_provider_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const loom_low_verify_provider_t* const* values;
} loom_low_verify_provider_list_t;

// Creates a low verification provider list from borrowed storage.
static inline loom_low_verify_provider_list_t
loom_low_verify_provider_list_make(
    const loom_low_verify_provider_t* const* values, iree_host_size_t count) {
  return (loom_low_verify_provider_list_t){
      /*.count=*/count,
      /*.values=*/values,
  };
}

// Returns an empty low verification provider list.
static inline loom_low_verify_provider_list_t
loom_low_verify_provider_list_empty(void) {
  return (loom_low_verify_provider_list_t){0};
}

// Returns true if |list| has no low verification providers.
static inline bool loom_low_verify_provider_list_is_empty(
    loom_low_verify_provider_list_t list) {
  return list.count == 0;
}

typedef struct loom_low_verify_options_t {
  // Descriptor registry available to this verification run. The registry is
  // target-owned static data; IR verification only uses it to resolve selected
  // descriptor sets and packet semantics.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional runtime/device target overlay applied when compatible with a low
  // function's target record.
  loom_target_selection_t target_selection;
  // Structured diagnostic emitter for user IR failures.
  iree_diagnostic_emitter_t emitter;
  // Optional target-owned low verification providers.
  loom_low_verify_provider_list_t provider_list;
  // Maximum number of errors to emit before aborting the walk. 0 means no
  // limit.
  uint32_t max_errors;
} loom_low_verify_options_t;

typedef struct loom_low_verify_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of warning diagnostics emitted.
  uint32_t warning_count;
} loom_low_verify_result_t;

typedef struct loom_low_verify_scratch_t {
  // Required value-id indexed u32 scratch storage for register-part masks.
  loom_value_u32_scratch_t* value_scratch;
} loom_low_verify_scratch_t;

// Returns low verification scratch backed by |module|'s reusable scratch.
static inline loom_low_verify_scratch_t loom_low_verify_scratch_for_module(
    loom_module_t* module) {
  return (loom_low_verify_scratch_t){
      /*.value_scratch=*/&module->scratch.values,
  };
}

// Verifies descriptor-local low function bodies in |module|. This checks that
// each low.func target resolves to a descriptor set and that descriptor-backed
// low.op/low.const packets match the selected descriptor rows. Verification is
// logically read-only on IR but mutates |scratch|.
iree_status_t loom_low_verify_module(const loom_module_t* module,
                                     const loom_low_verify_options_t* options,
                                     loom_low_verify_scratch_t* scratch,
                                     loom_low_verify_result_t* out_result);

// Returns the module being verified.
const loom_module_t* loom_low_verify_module_context_module(
    const loom_low_verify_module_context_t* context);

// Returns the scratch arena retained for the full low verification run.
iree_arena_allocator_t* loom_low_verify_module_context_arena(
    loom_low_verify_module_context_t* context);

// Returns true when the shared diagnostic limit has been reached.
bool loom_low_verify_module_context_should_stop(
    const loom_low_verify_module_context_t* context);

// Returns the module being verified.
const loom_module_t* loom_low_verify_context_module(
    const loom_low_verify_context_t* context);

// Returns the low function definition being verified.
const loom_op_t* loom_low_verify_context_function_op(
    const loom_low_verify_context_t* context);

// Returns the body region of the low function being verified, or NULL for
// declarations and malformed definitions rejected by generic verification.
const loom_region_t* loom_low_verify_context_function_body(
    const loom_low_verify_context_t* context);

// Returns the resolved target selected for the low function being verified.
const loom_low_resolved_target_t* loom_low_verify_context_target(
    const loom_low_verify_context_t* context);

// Returns the scratch arena reset after the current function verification.
iree_arena_allocator_t* loom_low_verify_context_arena(
    loom_low_verify_context_t* context);

// Returns the module-level state produced by this provider's begin_module hook.
void* loom_low_verify_context_provider_module_state(
    const loom_low_verify_context_t* context);

// Returns true when the shared diagnostic limit has been reached.
bool loom_low_verify_context_should_stop(
    const loom_low_verify_context_t* context);

// Emits a structured low verification diagnostic through the shared accounting
// path. Status is reserved for diagnostic sink failures.
iree_status_t loom_low_verify_context_emit(
    loom_low_verify_context_t* context, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_VERIFY_H_
