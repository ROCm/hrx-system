// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private ABI shared by launch-configuration program producers and consumers.

#ifndef LOOM_TARGET_EMIT_VM_LAUNCH_CONFIG_ABI_H_
#define LOOM_TARGET_EMIT_VM_LAUNCH_CONFIG_ABI_H_

#include "iree/base/api.h"

enum {
  // Reserved prefix on every launch-config VM export name. Consumers require
  // the prefix and remove exactly one byte before exposing the corresponding
  // compiled-kernel export name. Applying the same prefix to every name
  // preserves their public lexicographic order.
  LOOM_VM_LAUNCH_CONFIG_EXPORT_PREFIX = '$',

  // Number of bytes occupied by |LOOM_VM_LAUNCH_CONFIG_EXPORT_PREFIX|.
  LOOM_VM_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH = 1,

  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_X = 0,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Y = 1,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Z = 2,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_X = 3,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Y = 4,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Z = 5,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X = 6,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Y = 7,
  LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Z = 8,
  LOOM_VM_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE = 9,
  LOOM_VM_LAUNCH_CONFIG_RESULT_COUNT = 10,
};

// Returns the typed export metadata key carrying required workgroup storage.
static inline iree_string_view_t
loom_vm_launch_config_workgroup_storage_metadata_key(void) {
  return IREE_SV("loom.launch.workgroup_storage_bytes");
}

#endif  // LOOM_TARGET_EMIT_VM_LAUNCH_CONFIG_ABI_H_
