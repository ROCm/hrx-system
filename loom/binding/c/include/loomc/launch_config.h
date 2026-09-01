// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LAUNCH_CONFIG_H_
#define LOOMC_LAUNCH_CONFIG_H_

#include <stdint.h>

#include "loomc/base.h"

/// @file
/// Concrete kernel launch configurations.
///
/// This dependency-light header defines the target-independent result of
/// evaluating a compiled kernel's launch policy. The format-specific program
/// that produces the result belongs to its target package.

#ifdef __cplusplus
extern "C" {
#endif

/// Complete concrete configuration for one compiled kernel launch.
///
/// Callers zero-initialize this structure, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG`, and set `structure_size` to
/// `sizeof(loomc_launch_config_t)` before passing it to a launch-policy
/// evaluator.
typedef struct loomc_launch_config_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Unordered extension chain for additional launch properties.
  void* next;

  /// Number of workgroups dispatched along each dimension.
  loomc_dimension3_t workgroup_count;

  /// Number of invocations in each workgroup along each dimension.
  ///
  /// Every dimension is nonzero after successful evaluation.
  loomc_dimension3_t workgroup_size;

  /// Number of workgroups in each cooperative cluster dimension.
  ///
  /// Kernels that do not use clustered launch report `{1, 1, 1}`.
  loomc_dimension3_t workgroup_cluster_size;

  /// Number of invocations in each hardware subgroup.
  ///
  /// Zero indicates that the compiled target has no fixed subgroup size.
  uint32_t subgroup_size;

  /// Total workgroup-local storage required per workgroup, in bytes.
  ///
  /// This is the final total capacity required by the compiled kernel,
  /// including storage derived from workload facts during compilation. Target
  /// adapters map this total requirement to their launch ABI; it is not
  /// necessarily an API's additional dynamic-shared-memory argument.
  uint64_t workgroup_storage_bytes;
} loomc_launch_config_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LAUNCH_CONFIG_H_
