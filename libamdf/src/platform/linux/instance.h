// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_INSTANCE_H_
#define AMDF_SRC_PLATFORM_LINUX_INSTANCE_H_

#include <pthread.h>
#include <stddef.h>

#include "libamdf/src/platform/instance.h"

struct amdf_platform_instance_t {
  // Host allocator copied from the public provider instance.
  amdf_allocator_t host_allocator;
  // Owned sysfs root used for discovery without process-global state.
  int sysfs_descriptor;
  // Qualified native host page length in bytes, cached before discovery.
  size_t page_size;
  // Serializes cold native connection preparation and device teardown.
  pthread_mutex_t native_mutex;
};

#endif  // AMDF_SRC_PLATFORM_LINUX_INSTANCE_H_
