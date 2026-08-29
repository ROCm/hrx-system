// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private ABI shared by launch-configuration program producers and consumers.

#ifndef LOOM_CODEGEN_LOW_LAUNCH_CONFIG_ABI_H_
#define LOOM_CODEGEN_LOW_LAUNCH_CONFIG_ABI_H_

#include "iree/base/api.h"

enum {
  // Prefix on every private VM export name. Consumers remove exactly one byte
  // before exposing the corresponding compiled-kernel export name. This keeps
  // names such as `initialize` from acquiring VM program semantics. Applying
  // the same prefix to every name preserves their public lexicographic order.
  LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX = '$',

  // Number of bytes occupied by |LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX|.
  LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH = 1,

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
};

// Returns the private symbol reserved for the companion's execution target.
static inline iree_string_view_t loom_kernel_launch_config_target_name(void) {
  return IREE_SV("__loom_launch_config_vm_target");
}

#endif  // LOOM_CODEGEN_LOW_LAUNCH_CONFIG_ABI_H_
