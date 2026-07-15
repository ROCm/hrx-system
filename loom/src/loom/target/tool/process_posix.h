// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_TOOL_PROCESS_POSIX_H_
#define LOOM_TARGET_TOOL_PROCESS_POSIX_H_

#include <dirent.h>
#include <spawn.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_tool_posix_spawn_policy_t {
  // OS-specific child process attributes.
  posix_spawnattr_t attributes;
  // Whether |attributes| must be passed to spawn and destroyed.
  bool attributes_initialized;
  // Descriptor directory retained through spawn by enumeration-based policies.
  DIR* descriptor_directory;
} loom_tool_posix_spawn_policy_t;

// Configures the OS-specific child descriptor inheritance policy.
iree_status_t loom_tool_posix_spawn_policy_initialize(
    posix_spawn_file_actions_t* file_actions,
    loom_tool_posix_spawn_policy_t* out_policy);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TOOL_PROCESS_POSIX_H_
