// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/registration/driver_module.h"

#include "iree/hal/drivers/amd/xdna/driver.h"

static iree_status_t iree_hal_amd_xdna_driver_factory_enumerate(
    void* self, iree_host_size_t* out_count,
    const iree_hal_driver_info_t** out_infos) {
  static const iree_hal_driver_info_t info = {
      .driver_name = IREE_SVL("xdna"),
      .full_name = IREE_SVL("AMD XDNA execution using libamdf"),
  };
  *out_count = 1;
  *out_infos = &info;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_driver_factory_try_create(
    void* self, iree_string_view_t name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(name, IREE_SV("xdna"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "driver factory only provides 'xdna'");
  }
  return iree_hal_amd_xdna_driver_create(host_allocator, out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_amd_xdna_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .enumerate = iree_hal_amd_xdna_driver_factory_enumerate,
      .try_create = iree_hal_amd_xdna_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
