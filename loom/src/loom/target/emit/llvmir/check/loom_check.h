// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR loom-check emit adapter and LLVM tool requirements.

#ifndef LOOM_TARGET_EMIT_LLVMIR_LOOM_CHECK_H_
#define LOOM_TARGET_EMIT_LLVMIR_LOOM_CHECK_H_

#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef const loom_llvmir_target_profile_provider_t*(
    IREE_API_PTR* loom_llvmir_loom_check_target_profile_provider_fn_t)(void);

typedef struct loom_llvmir_loom_check_emit_provider_t {
  // Base loom-check emit provider embedded as the first field for dispatch.
  loom_check_emit_provider_t provider;
  // Target profile provider functions linked into this runner.
  const loom_llvmir_loom_check_target_profile_provider_fn_t*
      target_profile_provider_functions;
  // Number of entries in |target_profile_provider_functions|.
  iree_host_size_t target_profile_provider_function_count;
} loom_llvmir_loom_check_emit_provider_t;

bool loom_llvmir_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider, iree_string_view_t target_name);

iree_status_t loom_llvmir_loom_check_emit_provider_check_requirements(
    const loom_check_emit_provider_t* provider,
    const loom_check_case_t* test_case, loom_check_result_t* result,
    bool* out_continue_execution);

iree_status_t loom_llvmir_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request);

iree_status_t loom_llvmir_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider, iree_string_builder_t* builder);

#define LOOM_LLVMIR_LOOM_CHECK_EMIT_PROVIDER_INITIALIZER(                  \
    target_profile_provider_functions_value,                               \
    target_profile_provider_function_count_value)                          \
  {                                                                        \
      .provider =                                                          \
          {                                                                \
              .name = IREE_SVL("llvmir"),                                  \
              .match = loom_llvmir_loom_check_emit_provider_matches,       \
              .check_requirements =                                        \
                  loom_llvmir_loom_check_emit_provider_check_requirements, \
              .execute = loom_llvmir_loom_check_emit_provider_execute,     \
              .append_names =                                              \
                  loom_llvmir_loom_check_emit_provider_append_names,       \
          },                                                               \
      .target_profile_provider_functions =                                 \
          target_profile_provider_functions_value,                         \
      .target_profile_provider_function_count =                            \
          target_profile_provider_function_count_value,                    \
  }

// Requirement provider for LLVM tools and llc target-profile checks.
extern const loom_check_requirement_provider_t
    loom_llvmir_loom_check_requirement_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_LLVMIR_LOOM_CHECK_H_
