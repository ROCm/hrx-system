// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/device_flags.h"

#include <string.h>

#include "iree/base/tooling/flags.h"

IREE_FLAG(string, device, "amdgpu", "Shared serving HAL device URI.");
IREE_FLAG(string, pool_backing, "elastic",
          "elastic demand-commits parameter/state slabs; fixed backs storage "
          "eagerly and supports device address sanitization.");
IREE_FLAG(int64_t, slab_bytes, 2097152,
          "Physical slab size; zero selects allocator recommendation.");
IREE_FLAG(int64_t, memory_bytes, 0,
          "Shared physical parameter/state budget; zero imposes no serving "
          "limit. Invocation workspace is accounted separately.");

iree_status_t loom_serve_device_create_from_flags(
    loom_serve_device_t** out_device, iree_allocator_t host_allocator) {
  *out_device = NULL;
  if ((strcmp(FLAG_pool_backing, "elastic") &&
       strcmp(FLAG_pool_backing, "fixed")) ||
      FLAG_slab_bytes < 0 || FLAG_memory_bytes < 0 ||
      (FLAG_memory_bytes && !strcmp(FLAG_pool_backing, "fixed"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool_backing must be elastic or fixed; slab_bytes "
                            "and memory_bytes must be nonnegative; a physical "
                            "budget requires elastic backing");
  }
  const loom_serve_device_options_t options = {
      .uri = iree_make_cstring_view(FLAG_device),
      .backing = !strcmp(FLAG_pool_backing, "elastic")
                     ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                     : LOOM_SERVE_DEVICE_BACKING_FIXED,
      .slab_size = (iree_device_size_t)FLAG_slab_bytes,
      .memory_limit = (uint64_t)FLAG_memory_bytes,
  };
  return loom_serve_device_create(&options, out_device, host_allocator);
}
