// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amd/gpu/image/hsaco_metadata.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco_test_util.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"

using iree::hal::amd::gpu::testing::Check;

static void Parse(size_t count, const iree_const_byte_span_t* payloads) {
  iree_hal_amd_gpu_metadata_t metadata = {};
  iree_status_t status = iree_hal_amd_gpu_hsaco_metadata_initialize(
      count, payloads, iree_allocator_system(), &metadata);
  if (!iree_status_is_ok(status)) {
    Check(metadata.storage == nullptr);
    Check(metadata.kernel_count == 0);
  } else {
    for (size_t i = 0; i < metadata.kernel_count; ++i) {
      iree_hal_amd_gpu_kernarg_layout_t* layout = nullptr;
      iree_status_t layout_status = iree_hal_amd_gpu_kernarg_layout_allocate(
          &metadata.kernels[i], iree_allocator_system(), &layout);
      if (iree_status_is_ok(layout_status)) {
        // Reconstruct native explicit arguments independently through public
        // reflection. Both paths must populate identical runtime fields.
        std::vector<uint8_t> constants(layout->constant_byte_length, 0x5a);
        std::vector<uint64_t> bindings(layout->binding_count,
                                       UINT64_C(0x123456789abcdef0));
        std::vector<uint8_t> native(layout->native_byte_length, 0);
        const auto* parameters =
            iree_hal_amd_gpu_kernarg_layout_parameters(layout);
        for (size_t j = 0; j < layout->parameter_count; ++j) {
          const auto& parameter = parameters[j];
          if (parameter.type ==
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
            iree_unaligned_store_le_u64(
                native.data() + parameter.native_abi_offset,
                bindings[parameter.offset]);
          } else if (parameter.size) {
            std::memcpy(native.data() + parameter.native_abi_offset,
                        constants.data() + parameter.offset, parameter.size);
          }
        }
        const iree_hal_amd_gpu_kernarg_inputs_t inputs = {
            .block_count = {7, 3, 1},
            .group_size = {64, 4, 1},
        };
        std::vector<uint8_t> dense_result(layout->byte_length, 0xa5);
        std::vector<uint8_t> native_result(layout->byte_length, 0xa5);
        iree_hal_amd_gpu_kernarg_layout_emplace(layout, constants.data(),
                                                bindings.data(), &inputs,
                                                dense_result.data());
        iree_hal_amd_gpu_kernarg_layout_emplace_native(
            layout, iree_make_const_byte_span(native.data(), native.size()),
            &inputs, native_result.data());
        Check(dense_result == native_result);
      } else {
        Check(layout == nullptr);
      }
      iree_status_free(layout_status);
      iree_hal_amd_gpu_kernarg_layout_free(layout);
    }
  }
  iree_status_free(status);
  iree_hal_amd_gpu_metadata_deinitialize(&metadata);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const iree_const_byte_span_t repeated[] = {
      iree_make_const_byte_span(data, size),
      iree_make_const_byte_span(data, size),
  };
  Parse(1, repeated);
  Parse(2, repeated);
  // Keep one valid note ahead of arbitrary input to exercise agreement and
  // rollback after earlier records were populated successfully.
  if (iree_hal_amd_gpu_hip_images_size()) {
    static const auto baseline =
        iree::hal::amd::gpu::testing::HipMetadataPayloads(
            iree_hal_amd_gpu_hip_images_create()[0]);
    if (!baseline.empty()) {
      const iree_const_byte_span_t pair[] = {baseline[0], repeated[0]};
      Parse(2, pair);
    }
  }
  if (size > 1) {
    const size_t split = (size_t{data[0]} | (size_t{data[1]} << 8)) % size;
    const iree_const_byte_span_t pair[] = {
        iree_make_const_byte_span(data, split),
        iree_make_const_byte_span(data + split, size - split),
    };
    Parse(2, pair);
  }
  return 0;
}

extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size, size_t max_size);

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size,
                                          size_t max_size, unsigned int seed) {
  const size_t count = iree_hal_amd_gpu_hip_images_size();
  if (count && (size == 0 || seed % 32 == 0)) {
    const auto payloads = iree::hal::amd::gpu::testing::HipMetadataPayloads(
        iree_hal_amd_gpu_hip_images_create()[seed % count]);
    if (!payloads.empty()) {
      const auto payload = payloads[seed % payloads.size()];
      if (payload.data_length <= max_size) {
        std::memcpy(data, payload.data, payload.data_length);
        return payload.data_length;
      }
    }
  }
  return LLVMFuzzerMutate(data, size, max_size);
}
