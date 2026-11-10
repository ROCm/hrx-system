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
#include "libamdf/src/platform/linux/dxg/connection.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

// Native Linux service selected on the first native discovery or open request.
typedef enum amdf_linux_platform_kind_e {
  // Host services exist, but no native accelerator service has been selected.
  AMDF_LINUX_PLATFORM_KIND_UNPREPARED = 0,
  // Kernel device files discovered through sysfs and operated through DRM/KFD.
  AMDF_LINUX_PLATFORM_KIND_NATIVE_DEVICE = 1,
  // WSL GPU service operated through /dev/dxg and KMT.
  AMDF_LINUX_PLATFORM_KIND_DXG = 2,
} amdf_linux_platform_kind_t;

struct amdf_platform_instance_t {
  // Host allocator copied from the public provider instance.
  amdf_allocator_t host_allocator;
  // Native service selected under native_mutex; immutable after preparation.
  amdf_linux_platform_kind_t kind;
  // Owned sysfs root for native device discovery, or -1 for DXG instances.
  int sysfs_descriptor;
  // Owned DXG connection, or descriptor -1 for native device instances.
  amdf_linux_dxg_connection_t connection;
  // Typed KMT view borrowing connection and host_allocator.
  amdf_linux_dxg_kmt_t kmt;
  // Qualified native host page length in bytes, cached before discovery.
  size_t page_size;
  // Optional owned-page exporter and its cached availability and limits.
  amdf_platform_host_export_t host_export;
  // Serializes cold native connection preparation and device teardown.
  pthread_mutex_t native_mutex;
};

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Selects the native discovery service under the instance's native-state lock.
// A failed attempt leaves the instance unprepared and may be retried. Success
// binds the service for all subsequent native discovery and endpoint requests.
amdf_status_t amdf_linux_instance_prepare_native(
    amdf_platform_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_LINUX_INSTANCE_H_
