// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/registration/driver_module.h"

#include "iree/hal/drivers/amd/driver.h"

static iree_status_t iree_hal_amd_driver_factory_enumerate(
    void* self, iree_host_size_t* out_count,
    const iree_hal_driver_info_t** out_infos) {
  static const iree_hal_driver_info_t infos[] = {
      {.driver_id = 0,
       .driver_name = IREE_SVL("xdna"),
       .full_name = IREE_SVL("AMD XDNA execution using libamdf")},
      {.driver_id = 0,
       .driver_name = IREE_SVL("rdna"),
       .full_name = IREE_SVL("AMD RDNA execution using libamdf")},
  };
  *out_count = IREE_ARRAYSIZE(infos);
  *out_infos = infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  return iree_hal_amd_driver_create(AMDF_NATIVE_LIFETIME_PROCESS,
                                    host_allocator, out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_amd_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .enumerate = iree_hal_amd_driver_factory_enumerate,
      .try_create = iree_hal_amd_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
