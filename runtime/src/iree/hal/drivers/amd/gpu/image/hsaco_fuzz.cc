// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amd/gpu/image/asan_images.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco_test_util.h"

using iree::hal::amd::gpu::testing::Check;

namespace {

// Owns admission records and the exact writable spans supplied to image_load.
// Native instructions are treated only as bytes and never executed.
struct Image {
  // Admitted records retaining the source wrapper.
  iree_hal_amd_gpu_image_t value = {};
  // Independent host allocation for each mapped segment.
  std::vector<std::vector<uint8_t>> storage;

  ~Image() { iree_hal_amd_gpu_image_deinitialize(&value); }

  iree_status_code_t Initialize(iree_const_byte_span_t bytes,
                                size_t segment_length) {
    auto* source = iree::hal::amd::gpu::testing::ByteSequence::Create(
        bytes, segment_length);
    iree_status_t status = iree_hal_amd_gpu_hsaco_image_initialize(
        source, iree_allocator_system(), &value);
    iree_byte_sequence_release(source);
    const auto code = iree_status_code(status);
    iree_status_free(status);
    return code;
  }

  iree_status_code_t Load() {
    // The format may describe enormous BSS or sparse virtual extents. Admission
    // still covers those inputs; only this harness's writable backing is
    // bounded.
    constexpr size_t kMaximumStorage = 2 * 1024 * 1024;
    size_t total_length = 0;
    for (size_t i = 0; i < value.segment_count; ++i) {
      if (value.segments[i].length > kMaximumStorage - total_length) {
        return IREE_STATUS_RESOURCE_EXHAUSTED;
      }
      total_length += value.segments[i].length;
    }
    storage.resize(value.segment_count);
    std::vector<iree_byte_span_t> mappings(value.segment_count);
    for (size_t i = 0; i < value.segment_count; ++i) {
      storage[i].resize(value.segments[i].length, 0xA5);
      mappings[i] = iree_make_byte_span(storage[i].data(), storage[i].size());
    }
    std::vector<iree_hal_amd_gpu_image_import_binding_t> bindings(
        value.import_count);
    for (size_t i = 0; i < bindings.size(); ++i) {
      bindings[i] = {.address = 0x10000 + i * 256, .resolved = true};
    }
    const uint64_t base_address =
        UINT64_C(0x100000000) & ~(value.alignment - 1);
    iree_status_t status = iree_hal_amd_gpu_image_load(
        &value, base_address, bindings.size(), bindings.data(), mappings.size(),
        mappings.data());
    // Narrow relocation values may be unrepresentable at this placement. Both
    // source representations must still produce the same untouched mappings.
    const auto code = iree_status_code(status);
    iree_status_free(status);
    for (size_t i = 0; i < value.symbol_count; ++i) {
      Check(iree_hal_amd_gpu_image_find_symbol(&value, value.symbols[i].name) ==
            &value.symbols[i]);
    }
    return code;
  }
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const auto bytes = iree_make_const_byte_span(data, size);
  Image contiguous;
  Image segmented;
  const auto contiguous_code = contiguous.Initialize(bytes, 0);
  const auto segmented_code = segmented.Initialize(bytes, 1 + size % 257);
  Check(contiguous_code == segmented_code);
  if (contiguous_code == IREE_STATUS_OK) {
    Check(contiguous.value.byte_length == segmented.value.byte_length);
    Check(contiguous.value.segment_count == segmented.value.segment_count);
    Check(contiguous.value.symbol_count == segmented.value.symbol_count);
    Check(contiguous.value.asan_data_object_count ==
          segmented.value.asan_data_object_count);
    for (size_t i = 0; i < contiguous.value.asan_data_object_count; ++i) {
      Check(contiguous.value.asan_data_objects[i].offset ==
            segmented.value.asan_data_objects[i].offset);
      Check(contiguous.value.asan_data_objects[i].length ==
            segmented.value.asan_data_objects[i].length);
    }
    Check(contiguous.Load() == segmented.Load());
    Check(contiguous.storage == segmented.storage);
  }
  return 0;
}

extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size, size_t max_size);

// Periodically restart from real HIP output so even an empty corpus reaches
// symbols, descriptors and relocations. All other mutations remain byte-level.
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size,
                                          size_t max_size, unsigned int seed) {
  const size_t hip_count = iree_hal_amd_gpu_hip_images_size();
  const size_t count = hip_count + iree_hal_amd_gpu_asan_images_size();
  if (count && (size == 0 || seed % 32 == 0)) {
    const size_t ordinal = seed % count;
    const auto& file =
        ordinal < hip_count
            ? iree_hal_amd_gpu_hip_images_create()[ordinal]
            : iree_hal_amd_gpu_asan_images_create()[ordinal - hip_count];
    if (file.size <= max_size) {
      std::memcpy(data, file.data, file.size);
      return file.size;
    }
  }
  return LLVMFuzzerMutate(data, size, max_size);
}
