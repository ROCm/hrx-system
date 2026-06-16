// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"
#include "iree/hal/remote/protocol/bootstrap.h"

#define FUZZ_ASSERT(condition) \
  do {                         \
    if (!(condition)) {        \
      __builtin_trap();        \
    }                          \
  } while (0)

static void FuzzParseCatalog(iree_const_byte_span_t catalog_bytes,
                             uint32_t device_ordinal) {
  iree_const_byte_span_t spec_bytes = iree_const_byte_span_empty();
  iree_status_t status = iree_hal_remote_bootstrap_device_catalog_select_spec(
      catalog_bytes, device_ordinal, &spec_bytes);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }

  FUZZ_ASSERT(spec_bytes.data != NULL);
  FUZZ_ASSERT(spec_bytes.data_length > 0);
  FUZZ_ASSERT(spec_bytes.data >= catalog_bytes.data);
  FUZZ_ASSERT(spec_bytes.data + spec_bytes.data_length <=
              catalog_bytes.data + catalog_bytes.data_length);

  iree_hal_device_spec_t* spec = NULL;
  status = iree_hal_remote_bootstrap_device_catalog_parse_spec(
      catalog_bytes, device_ordinal, iree_allocator_system(), &spec);
  if (iree_status_is_ok(status)) {
    FUZZ_ASSERT(spec != NULL);
    FUZZ_ASSERT(iree_hal_device_spec_identity(spec) != NULL);
    (void)iree_hal_device_spec_digest(spec);
    iree_hal_device_spec_release(spec);
  } else {
    iree_status_ignore(status);
  }
}

static void FuzzWrappedSpec(const uint8_t* data, size_t size) {
  if (size == 0 || size > IREE_HOST_SIZE_MAX) return;

  const iree_host_size_t spec_offset =
      sizeof(iree_hal_remote_bootstrap_device_catalog_header_t) +
      sizeof(iree_hal_remote_bootstrap_device_spec_entry_t);
  iree_host_size_t padded_spec_length = 0;
  if (!iree_host_size_checked_align((iree_host_size_t)size, 8,
                                    &padded_spec_length)) {
    return;
  }
  iree_host_size_t catalog_length = 0;
  if (!iree_host_size_checked_add(spec_offset, padded_spec_length,
                                  &catalog_length)) {
    return;
  }

  uint8_t* catalog_data = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), catalog_length, (void**)&catalog_data);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }
  memset(catalog_data, 0, catalog_length);

  iree_hal_remote_bootstrap_device_catalog_header_t* header =
      (iree_hal_remote_bootstrap_device_catalog_header_t*)catalog_data;
  header->magic = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_MAGIC;
  header->version = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_VERSION;
  header->flags = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_FLAG_NONE;
  header->device_count = 1;

  iree_hal_remote_bootstrap_device_spec_entry_t* entry =
      (iree_hal_remote_bootstrap_device_spec_entry_t*)(catalog_data +
                                                       sizeof(*header));
  entry->device_ordinal = 0;
  entry->flags = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_SPEC_ENTRY_FLAG_NONE;
  entry->spec_offset = spec_offset;
  entry->spec_length = (uint64_t)size;
  memcpy(catalog_data + spec_offset, data, size);

  FuzzParseCatalog(iree_make_const_byte_span(catalog_data, catalog_length),
                   /*device_ordinal=*/0);
  iree_allocator_free(iree_allocator_system(), catalog_data);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_const_byte_span_t full_catalog = iree_make_const_byte_span(data, size);
  FuzzParseCatalog(full_catalog, /*device_ordinal=*/0);
  FuzzWrappedSpec(data, size);

  if (size >= sizeof(uint32_t)) {
    uint32_t device_ordinal = 0;
    memcpy(&device_ordinal, data, sizeof(device_ordinal));
    iree_const_byte_span_t prefixed_catalog = iree_make_const_byte_span(
        data + sizeof(device_ordinal), size - sizeof(device_ordinal));
    FuzzParseCatalog(prefixed_catalog, device_ordinal);
  }

  return 0;
}
