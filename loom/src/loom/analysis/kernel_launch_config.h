// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel launch configuration analysis.
//
// This layer evaluates launch configuration regions under caller-provided
// workload arguments and target contracts. It sits above raw kernel dialect
// helpers and below public API wrappers, tools, and artifact producers.

#ifndef LOOM_KERNEL_LAUNCH_CONFIG_H_
#define LOOM_KERNEL_LAUNCH_CONFIG_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/module.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_kernel_launch_config_field_flag_bits_e {
  // workgroup_count is present.
  LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT = 0x1u,

  // workgroup_size is present.
  LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE = 0x2u,

  // subgroup_size is present.
  LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE = 0x4u,

  // workgroup_storage_bytes is present.
  LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES = 0x8u,
} loom_kernel_launch_config_field_flag_bits_t;

typedef uint32_t loom_kernel_launch_config_field_flags_t;

typedef enum loom_kernel_launch_config_failure_e {
  // Evaluation succeeded.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE = 0,

  // The requested function symbol was not found.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_FUNCTION_NOT_FOUND = 1,

  // The requested symbol is not a source kernel definition.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NOT_KERNEL = 2,

  // Workload argument count does not match the launch-config region signature.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_COUNT = 3,

  // A workload argument cannot be represented from an i64 value.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_TYPE = 4,

  // Target contract resolution emitted diagnostics.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_TARGET_CONTRACT = 5,

  // Required workgroup count could not be resolved.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_COUNT = 6,

  // Required workgroup size could not be resolved.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_SIZE = 7,

  // Required subgroup size could not be resolved.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_SUBGROUP_SIZE = 8,

  // Required workgroup-local storage byte count could not be resolved.
  LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_STORAGE_BYTES = 9,
} loom_kernel_launch_config_failure_t;

static inline bool loom_kernel_launch_config_has_failure(
    loom_kernel_launch_config_failure_t failure) {
  return failure != LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE;
}

typedef struct loom_kernel_launch_config_options_t {
  // Kernel function symbol to evaluate, with or without a leading @.
  iree_string_view_t function_symbol;

  // Workload argument values supplied as signed 64-bit integers.
  const int64_t* workload_arguments;

  // Number of entries in workload_arguments.
  iree_host_size_t workload_argument_count;

  // Fields that must be present for evaluation to succeed.
  loom_kernel_launch_config_field_flags_t required_fields;

  // Structured diagnostic emitter for target-contract diagnostics.
  iree_diagnostic_emitter_t diagnostic_emitter;
} loom_kernel_launch_config_options_t;

typedef struct loom_kernel_launch_config_t {
  // Present evaluated fields.
  loom_kernel_launch_config_field_flags_t fields;

  // Optional concrete workgroup count.
  loom_target_dispatch_workgroup_count_t workgroup_count;

  // Optional concrete local workgroup size.
  loom_target_workgroup_size_t workgroup_size;

  // Optional concrete subgroup size.
  uint32_t subgroup_size;

  // Optional concrete workgroup-local storage byte count.
  uint64_t workgroup_storage_bytes;

  // Evaluation failure code, or NONE on success.
  loom_kernel_launch_config_failure_t failure;
} loom_kernel_launch_config_t;

bool loom_kernel_launch_config_fields_are_valid(
    loom_kernel_launch_config_field_flags_t fields);

iree_status_t loom_kernel_launch_config_try_evaluate_direct(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    const loom_kernel_launch_config_options_t* options,
    loom_kernel_launch_config_t* out_config, bool* out_evaluated);

iree_status_t loom_kernel_launch_config_evaluate(
    loom_module_t* module, iree_arena_block_pool_t* block_pool,
    const loom_kernel_launch_config_options_t* options,
    loom_kernel_launch_config_t* out_config);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_KERNEL_LAUNCH_CONFIG_H_
