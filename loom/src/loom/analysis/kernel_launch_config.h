// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-produced kernel launch configuration evaluation.

#ifndef LOOM_KERNEL_LAUNCH_CONFIG_H_
#define LOOM_KERNEL_LAUNCH_CONFIG_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/pass/value_facts.h"
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

  // workgroup_cluster_size is present.
  LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_CLUSTER_SIZE = 0x10u,
} loom_kernel_launch_config_field_flag_bits_t;

typedef uint32_t loom_kernel_launch_config_field_flags_t;

typedef struct loom_kernel_launch_config_t {
  // Present evaluated fields.
  loom_kernel_launch_config_field_flags_t fields;

  // Optional concrete workgroup count.
  loom_target_dispatch_workgroup_count_t workgroup_count;

  // Optional concrete local workgroup size.
  loom_target_workgroup_size_t workgroup_size;

  // Optional concrete cooperative workgroup cluster size.
  loom_target_workgroup_cluster_size_t workgroup_cluster_size;

  // Optional concrete subgroup size.
  uint32_t subgroup_size;

  // Optional concrete workgroup-local storage byte count.
  uint64_t workgroup_storage_bytes;
} loom_kernel_launch_config_t;

// Result positions in the compiler-produced launch-config function ABI.
typedef enum loom_kernel_launch_config_result_e {
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_X = 0,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Y = 1,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Z = 2,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_X = 3,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Y = 4,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Z = 5,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X = 6,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Y = 7,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Z = 8,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE = 9,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES = 10,
  LOOM_KERNEL_LAUNCH_CONFIG_RESULT_COUNT = 11,
} loom_kernel_launch_config_result_t;

// Bound compiler-produced host function ready for launch evaluation.
//
// Construction is infallible for functions from a verified launch-config
// program. Public artifact loaders validate the function ABI before binding.
typedef struct loom_kernel_launch_config_function_t {
  // Pure host function implementing the launch calculation.
  loom_func_like_t function;

  // Public executable export name borrowed from the module string table.
  iree_string_view_t name;

  // Positional workload argument value IDs borrowed from |function|.
  const loom_value_id_t* argument_ids;

  // Positional result value IDs following loom_kernel_launch_config_result_t.
  const loom_value_id_t* result_ids;

  // Number of entries in |argument_ids|.
  uint16_t argument_count;
} loom_kernel_launch_config_function_t;

// Binds a verified compiler-produced host function for repeated evaluation.
loom_kernel_launch_config_function_t loom_kernel_launch_config_function_bind(
    const loom_module_t* module, loom_func_like_t function);

// Evaluates one compiler-produced launch function for raw workload bits.
//
// |fact_owner| is reusable caller-owned scratch. Evaluation invalidates its
// active scope before returning while retaining allocated table capacity.
iree_status_t loom_kernel_launch_config_function_evaluate(
    const loom_module_t* module,
    const loom_kernel_launch_config_function_t* function,
    const uint64_t* workload_argument_bits,
    iree_host_size_t workload_argument_count,
    loom_pass_value_fact_owner_t* fact_owner,
    loom_kernel_launch_config_t* out_config);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_KERNEL_LAUNCH_CONFIG_H_
