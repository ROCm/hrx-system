// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/formats/parameter_index_fuzz_util.h"

namespace {

class Input {
 public:
  Input(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  uint8_t Next() { return position_ < size_ ? data_[position_++] : 0; }

  std::string Bytes(size_t maximum_length) {
    const size_t length =
        std::min<size_t>(Next() % (maximum_length + 1), size_ - position_);
    if (!length) {
      return {};
    }
    std::string value(reinterpret_cast<const char*>(data_ + position_), length);
    position_ += length;
    return value;
  }

 private:
  // Complete mutation bytes borrowed for one callback.
  const uint8_t* data_;
  // Number of bytes available in |data_|.
  size_t size_;
  // Offset of the next byte to consume.
  size_t position_ = 0;
};

void FuzzGeneratedArchive(const uint8_t* data, size_t size) {
  Input input(data, size);
  iree_io_parameter_archive_builder_t builder;
  IREE_CHECK_OK(iree_io_parameter_archive_builder_initialize(
      iree_allocator_system(), &builder));

  const size_t entry_count = input.Next() % 9;
  for (size_t i = 0; i < entry_count; ++i) {
    const std::string name = "parameter" + std::to_string(i);
    const std::string metadata = input.Bytes(16);
    const iree_string_view_t name_view =
        iree_make_string_view(name.data(), name.size());
    const iree_const_byte_span_t metadata_span =
        iree_make_const_byte_span(metadata.data(), metadata.size());
    if (input.Next() & 1) {
      const uint8_t pattern_length = uint8_t{1} << (input.Next() % 5);
      std::array<uint8_t, IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH> pattern =
          {};
      for (uint8_t j = 0; j < pattern_length; ++j) {
        pattern[j] = input.Next();
      }
      const uint64_t length = pattern_length * (input.Next() % 17);
      IREE_CHECK_OK(iree_io_parameter_archive_builder_add_splat_entry(
          &builder, name_view, metadata_span, pattern.data(), pattern_length,
          length));
    } else {
      const uint64_t alignment =
          (input.Next() & 1) ? uint64_t{1} << (input.Next() % 7) : 0;
      const uint64_t length = input.Next();
      IREE_CHECK_OK(iree_io_parameter_archive_builder_add_data_entry(
          &builder, name_view, metadata_span, alignment, length));
    }
  }

  iree_io_physical_size_t total_size = 0;
  IREE_CHECK_OK(
      iree_io_parameter_archive_builder_total_size(&builder, &total_size));
  std::vector<uint8_t> archive(static_cast<size_t>(total_size), 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(archive.data(), archive.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, file_handle,
                                    0, iree_allocator_system(), &stream));
  iree_io_parameter_index_t* built_index = nullptr;
  IREE_CHECK_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &built_index));
  IREE_CHECK_OK(iree_io_parameter_archive_builder_write(
      &builder, file_handle, 0, stream, built_index));

  iree_io_parameter_fuzz_parse(
      iree_make_const_byte_span(archive.data(), archive.size()),
      iree_io_parse_irpa_index, built_index);

  iree_io_parameter_index_release(built_index);
  iree_io_stream_release(stream);
  iree_io_file_handle_release(file_handle);
  iree_io_parameter_archive_builder_deinitialize(&builder);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > 1024 * 1024) {
    return 0;
  }
  iree_io_parameter_fuzz_parse(iree_make_const_byte_span(data, size),
                               iree_io_parse_irpa_index, nullptr);
  FuzzGeneratedArchive(data, size);
  return 0;
}
