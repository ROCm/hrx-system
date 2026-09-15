// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_INSTANCE_H_
#define AMDF_SRC_PLATFORM_WINDOWS_INSTANCE_H_

#include "libamdf/src/platform/instance.h"
#include "libamdf/src/platform/windows/kmt_api.h"

struct amdf_platform_instance_t {
  // Host allocator copied from the public provider instance.
  amdf_allocator_t host_allocator;
  // Immutable KMT procedure table owned by this platform instance.
  amdf_kmt_api_t kmt;
  // Serializes cold native connection preparation and device teardown.
  SRWLOCK native_lock;
};

#endif  // AMDF_SRC_PLATFORM_WINDOWS_INSTANCE_H_
