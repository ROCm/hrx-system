// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/image/metadata.h"

void iree_hal_amd_gpu_metadata_deinitialize(
    iree_hal_amd_gpu_metadata_t* metadata) {
  if (!metadata) {
    return;
  }
  iree_allocator_free(metadata->host_allocator, metadata->storage);
  memset(metadata, 0, sizeof(*metadata));
}
