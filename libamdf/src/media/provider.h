// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_PROVIDER_H_
#define AMDF_SRC_MEDIA_PROVIDER_H_

#include "amdf/media.h"
#include "libamdf/src/device.h"
#include "libamdf/src/kernel_queue.h"

// Native providers embed these prefixes. The extension dispatches through the
// exact activated device; neither public operations nor owners select a UMD.
typedef struct amdf_media_device_t {
  // Core lifetime, memory services, and immutable execution families.
  amdf_device_t base;
  // Implementation selected before publication, with the public call shape.
  const amdf_media_api_t* operations;
} amdf_media_device_t;

struct amdf_media_image_t {
  // Borrowed device whose native image identity this owner retains.
  amdf_media_device_t* device;
};

struct amdf_media_session_t {
  // Borrowed queue defining codec execution order and provider identity.
  amdf_kernel_queue_t* queue;
};

struct amdf_media_command_t {
  // Borrowed immutable session defining configuration and provider identity.
  amdf_media_session_t* session;
};

// Returns operations for an already-validated media activation.
static inline const amdf_media_api_t* amdf_media_device_operations(
    const amdf_device_t* device) {
  return ((const amdf_media_device_t*)device)->operations;
}

#ifdef __cplusplus
extern "C" {
#endif

// Transactional activation bound by the optional provider build. The common
// extension has validated the public prefix, endpoint, and output pointer.
amdf_status_t amdf_media_provider_device_create(
    amdf_endpoint_t* endpoint,
    const amdf_media_device_create_info_t* create_info,
    amdf_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_PROVIDER_H_
