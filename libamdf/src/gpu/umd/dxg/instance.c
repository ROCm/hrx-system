// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/instance.h"

amdf_status_t amdf_gpu_umd_instance_prepare(
    amdf_gpu_umd_instance_t** state, amdf_native_lifetime_t native_lifetime,
    amdf_allocator_t host_allocator) {
  // The platform instance already owns WDDM's shared KMT connection. Native
  // device and paging objects remain on each explicitly created device.
  (void)state;
  (void)native_lifetime;
  (void)host_allocator;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_instance_destroy(amdf_gpu_umd_instance_t* instance) {
  (void)instance;
  return AMDF_STATUS_OK;
}
