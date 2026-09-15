// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/instance.h"

#include <stddef.h>

#include "libamdf/src/child_tracker.h"

amdf_status_t AMDF_CALL amdf_endpoint_enumerate(
    amdf_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count) {
  if (out_count == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (instance == NULL || (capacity != 0 && summaries == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_platform_endpoint_enumerate(instance->platform, capacity,
                                          summaries, out_count);
}

amdf_platform_instance_t* amdf_instance_platform(amdf_instance_t* instance) {
  return instance->platform;
}

amdf_allocator_t amdf_instance_host_allocator(const amdf_instance_t* instance) {
  return instance->host_allocator;
}

amdf_native_lifetime_t amdf_instance_native_lifetime(
    const amdf_instance_t* instance) {
  return instance->native_lifetime;
}

amdf_status_t amdf_instance_register_endpoint(amdf_instance_t* instance) {
  return amdf_child_tracker_register(&instance->children);
}

void amdf_instance_unregister_endpoint(amdf_instance_t* instance) {
  amdf_child_tracker_unregister(&instance->children);
}
