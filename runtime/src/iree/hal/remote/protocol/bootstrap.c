// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/protocol/bootstrap.h"

#include <string.h>

#include "iree/hal/device_spec.h"

IREE_API_EXPORT iree_status_t
iree_hal_remote_bootstrap_device_catalog_select_spec(
    iree_const_byte_span_t catalog_bytes, uint32_t device_ordinal,
    iree_const_byte_span_t* out_spec_bytes) {
  IREE_ASSERT_ARGUMENT(out_spec_bytes);
  *out_spec_bytes = iree_const_byte_span_empty();

  if (catalog_bytes.data_length <
      sizeof(iree_hal_remote_bootstrap_device_catalog_header_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote bootstrap device catalog too small: %" PRIhsz
        " bytes (need at least %zu)",
        catalog_bytes.data_length,
        sizeof(iree_hal_remote_bootstrap_device_catalog_header_t));
  }

  iree_hal_remote_bootstrap_device_catalog_header_t header;
  memcpy(&header, catalog_bytes.data, sizeof(header));
  if (header.magic != IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote bootstrap device catalog has invalid "
                            "magic 0x%08x",
                            header.magic);
  }
  if (header.version != IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote bootstrap device catalog version %u is unsupported",
        (uint32_t)header.version);
  }
  if (header.flags != IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_FLAG_NONE ||
      header.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote bootstrap device catalog reserved fields "
                            "must be zero");
  }
  if (header.device_count == 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote bootstrap device catalog is empty");
  }

  iree_host_size_t entry_table_size = 0;
  if (!iree_host_size_checked_mul(
          header.device_count,
          sizeof(iree_hal_remote_bootstrap_device_spec_entry_t),
          &entry_table_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "remote bootstrap device catalog entry table "
                            "size overflow");
  }
  iree_host_size_t entry_table_end = 0;
  if (!iree_host_size_checked_add(
          sizeof(iree_hal_remote_bootstrap_device_catalog_header_t),
          entry_table_size, &entry_table_end)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "remote bootstrap device catalog entry table "
                            "end overflow");
  }
  if (catalog_bytes.data_length < entry_table_end) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote bootstrap device catalog too small for %u entries: %" PRIhsz
        " bytes (need %" PRIhsz ")",
        header.device_count, catalog_bytes.data_length, entry_table_end);
  }

  iree_host_size_t expected_data_offset = 0;
  if (!iree_host_size_checked_align(entry_table_end, 8,
                                    &expected_data_offset)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "remote bootstrap device catalog data offset "
                            "overflow");
  }
  bool found = false;
  iree_const_byte_span_t selected_spec_bytes = iree_const_byte_span_empty();

  const uint8_t* entry_table =
      catalog_bytes.data +
      sizeof(iree_hal_remote_bootstrap_device_catalog_header_t);
  for (uint32_t i = 0; i < header.device_count; ++i) {
    iree_hal_remote_bootstrap_device_spec_entry_t entry;
    memcpy(
        &entry,
        entry_table + (iree_host_size_t)i *
                          sizeof(iree_hal_remote_bootstrap_device_spec_entry_t),
        sizeof(entry));

    if (entry.device_ordinal != i) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u has "
                              "ordinal %u",
                              i, entry.device_ordinal);
    }
    if (entry.flags != IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_SPEC_ENTRY_FLAG_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u has "
                              "reserved flags 0x%08x",
                              i, entry.flags);
    }
    if (entry.spec_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u has "
                              "empty spec bytes",
                              i);
    }
    if (entry.spec_offset > (uint64_t)IREE_HOST_SIZE_MAX ||
        entry.spec_length > (uint64_t)IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u range "
                              "exceeds host addressable size",
                              i);
    }

    iree_host_size_t spec_offset = (iree_host_size_t)entry.spec_offset;
    iree_host_size_t spec_length = (iree_host_size_t)entry.spec_length;
    if (!iree_host_size_has_alignment(spec_offset, 8)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u spec "
                              "offset %" PRIhsz " is not 8-byte aligned",
                              i, spec_offset);
    }
    if (spec_offset < expected_data_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u spec "
                              "offset %" PRIhsz
                              " overlaps prior catalog data ending at %" PRIhsz,
                              i, spec_offset, expected_data_offset);
    }

    iree_host_size_t spec_end = 0;
    if (!iree_host_size_checked_add(spec_offset, spec_length, &spec_end)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote bootstrap device catalog entry %u spec "
                              "range overflows",
                              i);
    }
    if (spec_end > catalog_bytes.data_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "remote bootstrap device catalog entry %u spec range [%" PRIhsz
          ", %" PRIhsz ") is outside catalog length %" PRIhsz,
          i, spec_offset, spec_end, catalog_bytes.data_length);
    }
    if (!iree_host_size_checked_align(spec_end, 8, &expected_data_offset)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "remote bootstrap device catalog entry %u "
                              "aligned end overflows",
                              i);
    }

    if (entry.device_ordinal == device_ordinal) {
      found = true;
      selected_spec_bytes = iree_make_const_byte_span(
          catalog_bytes.data + spec_offset, spec_length);
    }
  }

  if (expected_data_offset != catalog_bytes.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote bootstrap device catalog has %" PRIhsz
                            " trailing bytes after expected end %" PRIhsz,
                            catalog_bytes.data_length - expected_data_offset,
                            expected_data_offset);
  }
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "remote bootstrap device catalog has no device "
                            "ordinal %u",
                            device_ordinal);
  }

  *out_spec_bytes = selected_spec_bytes;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_bootstrap_device_catalog_parse_spec(
    iree_const_byte_span_t catalog_bytes, uint32_t device_ordinal,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;

  iree_const_byte_span_t spec_bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_remote_bootstrap_device_catalog_select_spec(
      catalog_bytes, device_ordinal, &spec_bytes));
  return iree_hal_device_spec_parse(spec_bytes, host_allocator, out_spec);
}
