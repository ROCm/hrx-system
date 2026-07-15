// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process_platform.h"

#if defined(IREE_PLATFORM_MACOS)

#include <string.h>

#include "loom/target/tool/process_posix.h"

static iree_status_t loom_tool_darwin_status(int error_number,
                                             const char* message) {
  return iree_make_status(iree_status_code_from_errno(error_number),
                          "%s (%d: %s)", message, error_number,
                          strerror(error_number));
}

iree_status_t loom_tool_posix_spawn_policy_initialize(
    posix_spawn_file_actions_t* file_actions,
    loom_tool_posix_spawn_policy_t* out_policy) {
  (void)file_actions;
  memset(out_policy, 0, sizeof(*out_policy));
  int spawn_result = posix_spawnattr_init(&out_policy->attributes);
  if (spawn_result != 0) {
    return loom_tool_darwin_status(
        spawn_result, "failed to initialize process spawn attributes");
  }
  out_policy->attributes_initialized = true;
  spawn_result = posix_spawnattr_setflags(&out_policy->attributes,
                                          POSIX_SPAWN_CLOEXEC_DEFAULT);
  if (spawn_result != 0) {
    return loom_tool_darwin_status(
        spawn_result, "failed to restrict inherited process descriptors");
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_MACOS
