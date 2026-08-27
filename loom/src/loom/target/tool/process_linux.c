// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif  // _DEFAULT_SOURCE

#include "loom/target/tool/process_platform.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include "loom/target/tool/process_posix.h"

#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#define LOOM_TOOL_HAS_POSIX_SPAWN_CLOSEFROM __GLIBC_PREREQ(2, 34)
#else
#define LOOM_TOOL_HAS_POSIX_SPAWN_CLOSEFROM 0
#endif

static iree_status_t loom_tool_linux_status(int error_number,
                                            const char* message) {
  return iree_make_status(iree_status_code_from_errno(error_number),
                          "%s (%d: %s)", message, error_number,
                          strerror(error_number));
}

static bool loom_tool_parse_descriptor_name(const char* name, int* out_fd) {
  if (*name == '\0') {
    return false;
  }
  int fd = 0;
  for (const char* cursor = name; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
    int digit = *cursor - '0';
    if (fd > (INT_MAX - digit) / 10) {
      return false;
    }
    fd = fd * 10 + digit;
  }
  *out_fd = fd;
  return true;
}

iree_status_t loom_tool_posix_spawn_policy_initialize(
    posix_spawn_file_actions_t* file_actions,
    loom_tool_posix_spawn_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));

#if LOOM_TOOL_HAS_POSIX_SPAWN_CLOSEFROM
  int spawn_result =
      posix_spawn_file_actions_addclosefrom_np(file_actions, STDERR_FILENO + 1);
  if (spawn_result != 0) {
    return loom_tool_linux_status(
        spawn_result, "failed to restrict inherited process descriptors");
  }
  return iree_ok_status();
#else
  out_policy->descriptor_directory = opendir("/proc/self/fd");
  if (out_policy->descriptor_directory == NULL) {
    return loom_tool_linux_status(errno,
                                  "failed to enumerate open descriptors");
  }

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    errno = 0;
    struct dirent* entry = readdir(out_policy->descriptor_directory);
    if (entry == NULL) {
      if (errno != 0) {
        status = loom_tool_linux_status(
            errno, "failed while enumerating open descriptors");
      }
      break;
    }
    int fd = -1;
    if (!loom_tool_parse_descriptor_name(entry->d_name, &fd) ||
        fd <= STDERR_FILENO) {
      continue;
    }
    int spawn_result = posix_spawn_file_actions_addclose(file_actions, fd);
    if (spawn_result != 0) {
      status = loom_tool_linux_status(
          spawn_result, "failed to exclude descriptor from child process");
    }
  }
  return status;
#endif  // LOOM_TOOL_HAS_POSIX_SPAWN_CLOSEFROM
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
