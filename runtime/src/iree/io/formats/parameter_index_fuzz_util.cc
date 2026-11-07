// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/parameter_index_fuzz_util.h"

#include <cstdlib>
#include <cstring>

namespace {

void Assert(bool condition) {
  if (!condition) {
    std::abort();
  }
}

bool ByteSpansEqual(iree_const_byte_span_t lhs, iree_const_byte_span_t rhs) {
  return lhs.data_length == rhs.data_length &&
         (!lhs.data_length ||
          std::memcmp(lhs.data, rhs.data, lhs.data_length) == 0);
}

void ValidateEntry(const iree_io_parameter_index_entry_t* entry) {
  Assert(entry != nullptr);
  Assert(!entry->key.size || entry->key.data != nullptr);
  Assert(!entry->metadata.data_length || entry->metadata.data != nullptr);
  switch (entry->type) {
    case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
      Assert(entry->storage.splat.pattern_length != 0);
      Assert(entry->storage.splat.pattern_length <=
             IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH);
      Assert(iree_is_power_of_two_uint64(entry->storage.splat.pattern_length));
      Assert(entry->length % entry->storage.splat.pattern_length == 0);
      break;
    case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
      Assert(entry->storage.file.handle != nullptr);
      const iree_io_file_handle_primitive_t primitive =
          iree_io_file_handle_primitive(entry->storage.file.handle);
      Assert(primitive.type == IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION);
      const uint64_t file_length = primitive.value.host_allocation.data_length;
      Assert(entry->storage.file.offset <= file_length);
      Assert(entry->length <= file_length - entry->storage.file.offset);
      const uint64_t alignment = entry->storage.file.minimum_alignment;
      Assert(!alignment || iree_is_power_of_two_uint64(alignment));
      Assert(!alignment || entry->storage.file.offset % alignment == 0);
      break;
    }
    default:
      std::abort();
  }
}

void ValidateIndex(iree_io_parameter_index_t* index) {
  const iree_host_size_t count = iree_io_parameter_index_count(index);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_io_parameter_index_entry_t* entry = nullptr;
    IREE_CHECK_OK(iree_io_parameter_index_get(index, i, &entry));
    ValidateEntry(entry);

    const iree_io_parameter_index_entry_t* lookup_entry = nullptr;
    IREE_CHECK_OK(
        iree_io_parameter_index_lookup(index, entry->key, &lookup_entry));
    Assert(iree_string_view_equal(entry->key, lookup_entry->key));
  }
}

void AssertEntriesEqual(const iree_io_parameter_index_entry_t* lhs,
                        const iree_io_parameter_index_entry_t* rhs) {
  Assert(iree_string_view_equal(lhs->key, rhs->key));
  Assert(ByteSpansEqual(lhs->metadata, rhs->metadata));
  Assert(lhs->length == rhs->length);
  Assert(lhs->type == rhs->type);
  switch (lhs->type) {
    case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
      Assert(lhs->storage.splat.pattern_length ==
             rhs->storage.splat.pattern_length);
      Assert(std::memcmp(lhs->storage.splat.pattern, rhs->storage.splat.pattern,
                         lhs->storage.splat.pattern_length) == 0);
      break;
    case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
      // Empty file ranges have no observable storage location. Parsers may
      // canonicalize their offset while preserving the length and alignment.
      Assert(!lhs->length ||
             lhs->storage.file.offset == rhs->storage.file.offset);
      Assert(lhs->storage.file.minimum_alignment ==
             rhs->storage.file.minimum_alignment);
      break;
    default:
      std::abort();
  }
}

void AssertIndicesEqual(iree_io_parameter_index_t* lhs,
                        iree_io_parameter_index_t* rhs) {
  const iree_host_size_t count = iree_io_parameter_index_count(lhs);
  Assert(count == iree_io_parameter_index_count(rhs));
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_io_parameter_index_entry_t* lhs_entry = nullptr;
    const iree_io_parameter_index_entry_t* rhs_entry = nullptr;
    IREE_CHECK_OK(iree_io_parameter_index_get(lhs, i, &lhs_entry));
    IREE_CHECK_OK(iree_io_parameter_index_get(rhs, i, &rhs_entry));
    AssertEntriesEqual(lhs_entry, rhs_entry);
  }
}

iree_status_t ParseIndex(iree_io_file_handle_t* file_handle,
                         iree_io_parameter_fuzz_parse_fn_t parse_fn,
                         iree_io_parameter_index_t** out_index) {
  *out_index = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(iree_allocator_system(), out_index));
  iree_status_t status =
      parse_fn(file_handle, *out_index, iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_io_parameter_index_release(*out_index);
    *out_index = nullptr;
  }
  return status;
}

}  // namespace

void iree_io_parameter_fuzz_parse(iree_const_byte_span_t contents,
                                  iree_io_parameter_fuzz_parse_fn_t parse_fn,
                                  iree_io_parameter_index_t* expected_index) {
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(const_cast<uint8_t*>(contents.data),
                          contents.data_length),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_io_parameter_index_t* parsed_index = nullptr;
  iree_status_t status = ParseIndex(file_handle, parse_fn, &parsed_index);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    iree_io_file_handle_release(file_handle);
    Assert(expected_index == nullptr);
    return;
  }
  ValidateIndex(parsed_index);

  if (expected_index) {
    ValidateIndex(expected_index);
    AssertIndicesEqual(expected_index, parsed_index);
  } else {
    iree_io_parameter_index_t* reparsed_index = nullptr;
    IREE_CHECK_OK(ParseIndex(file_handle, parse_fn, &reparsed_index));
    ValidateIndex(reparsed_index);
    AssertIndicesEqual(parsed_index, reparsed_index);
    iree_io_parameter_index_release(reparsed_index);
  }

  iree_io_parameter_index_release(parsed_index);
  iree_io_file_handle_release(file_handle);
}
