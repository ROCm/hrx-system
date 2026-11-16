// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#if defined(HRX_HAS_IREE_AMDXDNA_DRIVER)

#include "amdxdna_executable_reuse_key.h"

#include <stdint.h>
#include <string.h>

hrx_status_t hrx_amdxdna_executable_reuse_key_initialize(
    const hrx_amdxdna_xclbin_identity_t* xclbin_identity,
    const hrx_amdxdna_executable_entry_point_t* entry_point,
    const hrx_amdxdna_executable_run_t* run,
    hrx_amdxdna_executable_reuse_key_t* out_key) {
  if (!xclbin_identity || !entry_point || !run || !out_key) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "reuse key arguments must be non-NULL");
  }
  memset(out_key, 0, sizeof(*out_key));

  const uint32_t context_mode = (uint32_t)entry_point->context_mode;
  const size_t header_length =
      sizeof(xclbin_identity->hash_1) + sizeof(xclbin_identity->hash_2) +
      sizeof(xclbin_identity->data_length) + sizeof(context_mode) +
      sizeof(entry_point->pdi_ordinal);
  if (entry_point->name.size > SIZE_MAX - header_length) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "entry point name is too long");
  }
  const size_t descriptor_length = header_length + entry_point->name.size;
  uint8_t* descriptor = out_key->inline_descriptor;
  if (descriptor_length > sizeof(out_key->inline_descriptor)) {
    hrx_status_t status = hrx_host_allocator_malloc(
        hrx_host_allocator_system(), descriptor_length,
        (void**)&out_key->heap_descriptor);
    if (!hrx_status_is_ok(status)) return status;
    descriptor = out_key->heap_descriptor;
  }

  // Fields are packed explicitly so the descriptor has no padding bytes.
  uint8_t* cursor = descriptor;
  memcpy(cursor, &xclbin_identity->hash_1, sizeof(xclbin_identity->hash_1));
  cursor += sizeof(xclbin_identity->hash_1);
  memcpy(cursor, &xclbin_identity->hash_2, sizeof(xclbin_identity->hash_2));
  cursor += sizeof(xclbin_identity->hash_2);
  memcpy(cursor, &xclbin_identity->data_length,
         sizeof(xclbin_identity->data_length));
  cursor += sizeof(xclbin_identity->data_length);
  memcpy(cursor, &context_mode, sizeof(context_mode));
  cursor += sizeof(context_mode);
  memcpy(cursor, &entry_point->pdi_ordinal, sizeof(entry_point->pdi_ordinal));
  cursor += sizeof(entry_point->pdi_ordinal);
  if (entry_point->name.size != 0) {
    memcpy(cursor, entry_point->name.data, entry_point->name.size);
  }

  out_key->hal_key.descriptor =
      iree_make_const_byte_span(descriptor, descriptor_length);
  out_key->hal_key.control_code = iree_make_const_byte_span(
      run->transaction.data, run->transaction.data_length);
  return hrx_ok_status();
}

void hrx_amdxdna_executable_reuse_key_deinitialize(
    hrx_amdxdna_executable_reuse_key_t* key) {
  if (!key) return;
  hrx_host_allocator_free(hrx_host_allocator_system(), key->heap_descriptor);
  memset(key, 0, sizeof(*key));
}

#endif  // HRX_HAS_IREE_AMDXDNA_DRIVER
