// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_location.h"

#include <string.h>

iree_status_t loom_sanitizer_make_site_location(
    loom_module_t* module, loom_location_id_t source_location,
    const loom_sanitizer_site_payload_t* payload,
    loom_location_id_t* out_location) {
  IREE_ASSERT_ARGUMENT(out_location);
  *out_location = LOOM_LOCATION_UNKNOWN;

  iree_host_size_t storage_length = 0;
  if (!iree_host_size_checked_add(LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH,
                                  payload->extension_data.data_length,
                                  &storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site payload length overflow");
  }

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&module->arena, storage_length, (void**)&storage));

  iree_host_size_t encoded_length = 0;
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_payload_encode(
      payload, iree_make_byte_span(storage, storage_length), &encoded_length));
  if (encoded_length < storage_length) {
    memset(storage + encoded_length, 0, storage_length - encoded_length);
  }
  return loom_module_add_location(
      module,
      loom_location_tagged(LOOM_LOCATION_TAG_SANITIZER_SITE, source_location,
                           storage, (uint32_t)encoded_length),
      out_location);
}
