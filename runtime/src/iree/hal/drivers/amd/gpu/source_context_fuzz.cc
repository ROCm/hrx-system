// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/source_context.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_hal_amd_gpu_source_context_registry_t registry;
  iree_hal_amd_gpu_source_context_registry_initialize(iree_allocator_system(),
                                                      &registry);
  const iree_hal_amd_gpu_source_context_t* context = nullptr;
  iree_status_t status = iree_hal_amd_gpu_source_context_registry_register(
      &registry, iree_make_const_byte_span(data, size), &context);
  if (iree_status_is_ok(status)) {
    const uint64_t count =
        size >= 32 ? iree_unaligned_load_le_u32(data + 8) : 0;
    for (uint64_t i = 0; i <= count; ++i) {
      iree_hal_device_event_site_t site;
      iree_hal_amd_gpu_source_context_resolve(context, i, &site);
    }
  }
  iree_status_free(status);
  iree_hal_amd_gpu_source_context_registry_deinitialize(&registry);
  return 0;
}
