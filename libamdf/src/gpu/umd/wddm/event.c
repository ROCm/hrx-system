// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/event.h"

// The native WDDM producer protocol is distinct from KFD signal mailboxes.
// This provider exposes kernel-queue completion separately and advertises no
// native-event producer capability.
amdf_status_t amdf_gpu_umd_event_create(
    amdf_gpu_umd_device_t* device, amdf_gpu_umd_event_t** out_event,
    amdf_gpu_umd_event_result_t* out_result) {
  (void)device;
  (void)out_event;
  (void)out_result;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_event_signal(amdf_gpu_umd_event_t* event) {
  (void)event;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_event_wait(amdf_gpu_umd_event_t* event,
                                      uint64_t timeout_nanoseconds) {
  (void)event;
  (void)timeout_nanoseconds;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_event_destroy(amdf_gpu_umd_event_t* event) {
  (void)event;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}
