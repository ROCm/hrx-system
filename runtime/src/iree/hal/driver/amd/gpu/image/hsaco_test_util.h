// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_TEST_UTIL_H_
#define IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_TEST_UTIL_H_

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/driver/amd/gpu/image/hip_images.h"

namespace iree::hal::amd::gpu::testing {

// Fuzzer invariants remain active in optimized sanitizer builds.
inline void Check(bool condition) {
  if (!condition) {
    std::abort();
  }
}

// Borrows immutable bytes that outlive all references to the sequence.
// Splitting changes only enumeration boundaries; neither setup nor enumeration
// copies payloads, so benchmarks measure the loader's gathering and copying
// costs.
struct ByteSequence {
  // Refcounted sequence base at offset zero.
  iree_byte_sequence_t base;
  // Borrowed immutable artifact bytes.
  iree_const_byte_span_t bytes;
  // Zero exposes a contiguous view; otherwise the maximum enumerated length.
  size_t segment_length;

  static iree_byte_sequence_t* Create(iree_const_byte_span_t bytes,
                                      size_t segment_length) {
    static const iree_byte_sequence_vtable_t vtable = {
        .destroy = Destroy,
        .enumerate = Enumerate,
        .try_get_contiguous_span = Contiguous,
    };
    auto* self = new ByteSequence{};
    self->bytes = bytes;
    self->segment_length = segment_length;
    iree_byte_sequence_initialize(&vtable, bytes.data_length, &self->base);
    return &self->base;
  }

 private:
  static void Destroy(iree_byte_sequence_t* base) {
    delete reinterpret_cast<ByteSequence*>(base);
  }
  static iree_status_t Enumerate(
      const iree_byte_sequence_t* base,
      iree_byte_sequence_segment_callback_t callback) {
    const auto* self = reinterpret_cast<const ByteSequence*>(base);
    const size_t stride =
        self->segment_length ? self->segment_length : self->bytes.data_length;
    for (size_t offset = 0; offset < self->bytes.data_length;) {
      const size_t length = std::min(stride, self->bytes.data_length - offset);
      IREE_RETURN_IF_ERROR(callback.fn(
          callback.user_data,
          iree_make_const_byte_span(self->bytes.data + offset, length)));
      offset += length;
    }
    return iree_ok_status();
  }
  static bool Contiguous(const iree_byte_sequence_t* base,
                         iree_const_byte_span_t* out_span) {
    const auto* self = reinterpret_cast<const ByteSequence*>(base);
    if (self->segment_length) {
      return false;
    }
    *out_span = self->bytes;
    return true;
  }
};

// Extracts metadata from the trusted, build-time HIP compiler corpus. Arbitrary
// fuzzer bytes must go through the production readers, never this fixture walk.
inline std::vector<iree_const_byte_span_t> HipMetadataPayloads(
    const iree_file_toc_t& file) {
  std::vector<iree_const_byte_span_t> payloads;
  const auto* data = reinterpret_cast<const uint8_t*>(file.data);
  const uint64_t program_offset = iree_unaligned_load_le_u64(data + 32);
  const uint16_t program_count = iree_unaligned_load_le_u16(data + 56);
  for (uint16_t i = 0; i < program_count; ++i) {
    const uint8_t* program = data + program_offset + i * 56;
    if (iree_unaligned_load_le_u32(program) != 4) {
      continue;  // PT_NOTE.
    }
    const uint64_t offset = iree_unaligned_load_le_u64(program + 8);
    const uint64_t length = iree_unaligned_load_le_u64(program + 32);
    for (uint64_t cursor = offset; cursor < offset + length;) {
      const uint8_t* note = data + cursor;
      const uint32_t name_length = iree_unaligned_load_le_u32(note);
      const uint32_t payload_length = iree_unaligned_load_le_u32(note + 4);
      const uint32_t type = iree_unaligned_load_le_u32(note + 8);
      const uint64_t name_padded = (uint64_t{name_length} + 3) & ~uint64_t{3};
      const uint64_t payload_padded =
          (uint64_t{payload_length} + 3) & ~uint64_t{3};
      if (type == 32 && name_length == 7 &&
          std::memcmp(note + 12, "AMDGPU", 7) == 0) {
        payloads.push_back(
            iree_make_const_byte_span(note + 12 + name_padded, payload_length));
      }
      cursor += 12 + name_padded + payload_padded;
    }
  }
  return payloads;
}

}  // namespace iree::hal::amd::gpu::testing

#endif  // IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_TEST_UTIL_H_
