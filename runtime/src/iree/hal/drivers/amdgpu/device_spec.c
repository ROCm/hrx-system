// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec.h"

#include <string.h>

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_amdgpu_device_spec_find_facet(
    const iree_hal_device_spec_t* device_spec) {
  IREE_ASSERT_ARGUMENT(device_spec);
  return iree_hal_device_spec_find_facet(
      device_spec,
      iree_make_cstring_view(IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_ID));
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_device_spec_decode_facet(
    const iree_hal_device_spec_facet_t* facet,
    iree_hal_amdgpu_device_spec_t* out_spec) {
  IREE_ASSERT_ARGUMENT(facet);
  IREE_ASSERT_ARGUMENT(out_spec);
  if (!iree_string_view_equal(
          facet->schema_id,
          iree_make_cstring_view(IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_ID))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "facet is not an AMDGPU device spec");
  }
  if (facet->schema_version != IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_VERSION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU device spec facet version %u is not supported",
        facet->schema_version);
  }
  memset(out_spec, 0, sizeof(*out_spec));
  if (facet->payload.data_length != sizeof(*out_spec)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU device spec payload must be exactly %" PRIhsz
        " bytes; got %" PRIhsz,
        sizeof(*out_spec), facet->payload.data_length);
  }
  memcpy(out_spec, facet->payload.data, sizeof(*out_spec));
  return iree_ok_status();
}
