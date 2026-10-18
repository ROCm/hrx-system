// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_HOST_EXPORT_H_
#define AMDF_SRC_PLATFORM_HOST_EXPORT_H_

#include "libamdf/src/platform/host_memory.h"

// Optional instance-owned transport for exporting immutable owned host backing.
// Discovery is cold; capability queries copy these cached facts without IO.
typedef struct amdf_platform_host_export_t {
  // Cached prerequisite result; construction can still fail after discovery.
  amdf_status_t availability;
  // Transport produced by this platform, or NONE without such a mechanism.
  amdf_external_memory_type_t type;
  // Configured per-export byte limit; zero when discovery could not establish
  // it.
  uint64_t maximum_byte_length;
  // Platform-owned native service handle, interpreted only by its
  // implementation.
  intptr_t native_handle;
} amdf_platform_host_export_t;

#ifdef __cplusplus
extern "C" {
#endif

// Discovers the optional service and records any failure in availability.
// Absence never prevents ordinary instance construction.
void amdf_platform_host_export_initialize(amdf_platform_host_export_t* service);

// Releases the service handle; allocated exports hold their own native
// references.
amdf_status_t amdf_platform_host_export_deinitialize(
    amdf_platform_host_export_t* service);

// Creates a fresh transport identity for these same owned pages. The caller
// imports it once into one independently owned attachment, then releases it.
// No implicit synchronization is shared with other exports of these pages.
// Failure leaves output unchanged. Geometry and service admission are validated
// by the construction plan; native resource admission remains fallible.
amdf_status_t amdf_platform_host_export_memory(
    const amdf_platform_host_export_t* service,
    const amdf_platform_host_backing_t* backing, uint64_t byte_length,
    amdf_external_memory_t* out_memory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_HOST_EXPORT_H_
