// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device/library.h"

#include "iree/hal/drivers/vulkan/device/kernels.h"

iree_const_byte_span_t iree_hal_vulkan_device_library_lookup(
    iree_string_view_t file_name) {
  const iree_file_toc_t* toc = iree_hal_vulkan_device_kernels_create();
  for (iree_host_size_t i = 0; i < iree_hal_vulkan_device_kernels_size(); ++i) {
    if (iree_string_view_equal(iree_make_cstring_view(toc[i].name),
                               file_name)) {
      return iree_make_const_byte_span(toc[i].data, toc[i].size);
    }
  }
  IREE_ASSERT(false, "Vulkan device module is missing from the library");
  return iree_const_byte_span_empty();
}
