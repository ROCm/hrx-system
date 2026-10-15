// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/kernarg.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

using Kind = iree_hal_amd_gpu_metadata_argument_kind_t;
using Argument = iree_hal_amd_gpu_metadata_argument_t;

Argument Arg(Kind kind, uint32_t offset, uint32_t size) {
  return {
      .name = IREE_SV("argument"),
      .offset = offset,
      .size = size,
      .kind = kind,
      .value_kind = IREE_SV("fixture"),
  };
}

// Fixtures are normalized image records: sorted, disjoint, and in range.
// Semantic widths remain the responsibility of executable ABI admission.
class KernargTest : public ::testing::Test {
 protected:
  void TearDown() override { iree_hal_amd_gpu_kernarg_layout_free(layout_); }
  iree_status_t Build(std::vector<Argument> arguments, uint32_t size) {
    iree_hal_amd_gpu_kernarg_layout_free(layout_);
    layout_ = nullptr;
    arguments_ = std::move(arguments);
    kernel_.arguments = arguments_.data();
    kernel_.argument_count = arguments_.size();
    kernel_.kernarg_segment_size = size;
    return iree_hal_amd_gpu_kernarg_layout_allocate(
        &kernel_, iree_allocator_system(), &layout_);
  }
  // Input records borrowed only during layout construction.
  std::vector<Argument> arguments_;
  // The image parser establishes these structural invariants.
  iree_hal_amd_gpu_metadata_kernel_t kernel_ = {
      .kernarg_segment_alignment = 8,
      .has_argument_metadata = true,
  };
  // Plain layout ownership independent of metadata record storage.
  iree_hal_amd_gpu_kernarg_layout_t* layout_ = nullptr;
};

TEST_F(KernargTest, DenseConstantsBindingsAndNativeReflection) {
  IREE_ASSERT_OK(
      Build({Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE, 0, 3),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER, 8, 8),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE, 16, 4),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE, 20, 2),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER, 24, 8)},
            40));
  EXPECT_LE(sizeof(*layout_), 64u);
  EXPECT_EQ(layout_->constant_byte_length, 9u);
  EXPECT_EQ(layout_->native_byte_length, 32u);
  EXPECT_EQ(layout_->constant_copy_count, 2u);
  EXPECT_EQ(layout_->binding_count, 2u);
  ASSERT_EQ(layout_->parameter_count, 5u);
  const auto* parameters = iree_hal_amd_gpu_kernarg_layout_parameters(layout_);
  const uint16_t offsets[] = {0, 0, 3, 7, 1};
  const uint16_t native_offsets[] = {0, 8, 16, 20, 24};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(parameters[i].offset, offsets[i]);
    EXPECT_EQ(parameters[i].native_abi_offset, native_offsets[i]);
    EXPECT_EQ(parameters[i].flags,
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET);
    EXPECT_TRUE(
        iree_string_view_equal(parameters[i].name, IREE_SV("argument")));
  }
  EXPECT_EQ(parameters[1].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(parameters[2].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  // Metadata records can disappear; prepared offsets and reflection survive.
  arguments_.clear();
  const uint8_t constants[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const uint64_t bindings[] = {UINT64_C(0x1817161514131211),
                               UINT64_C(0x2827262524232221)};
  std::array<uint8_t, 42> storage;
  storage.fill(0xa5);
  iree_hal_amd_gpu_kernarg_layout_emplace(layout_, constants, bindings, nullptr,
                                          storage.data() + 1);
  const std::vector<uint8_t> expected = {
      0xa5, 1,    2,    3,    0,    0, 0, 0, 0, 0x11, 0x12, 0x13, 0x14, 0x15,
      0x16, 0x17, 0x18, 4,    5,    6, 7, 8, 9, 0,    0,    0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27, 0x28, 0, 0, 0, 0, 0,    0,    0,    0,    0xa5};
  EXPECT_EQ(std::vector<uint8_t>(storage.begin(), storage.end()), expected);
}

TEST_F(KernargTest, RecordedTemplateLeavesIndexedBindingRelocationsZero) {
  IREE_ASSERT_OK(
      Build({Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER, 0, 8),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE, 8, 4),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER, 16, 8),
             Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X,
                 24, 4)},
            32));
  uint32_t constant = 0x12345678;
  const iree_hal_amd_gpu_kernarg_inputs_t inputs = {.block_count = {17, 1, 1}};
  std::array<uint8_t, 32> bytes;
  bytes.fill(0xCD);
  iree_hal_amd_gpu_kernarg_layout_emplace(layout_, &constant, nullptr, &inputs,
                                          bytes.data());
  const auto* offsets =
      iree_hal_amd_gpu_kernarg_layout_binding_offsets(layout_);
  EXPECT_EQ(offsets[0], 0u);
  EXPECT_EQ(offsets[1], 16u);
  for (uint16_t i = 0; i < layout_->binding_count; ++i) {
    EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data() + offsets[i]), 0u);
  }
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data() + 8), constant);
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data() + 24), 17u);
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data() + 28), 0u);
}

TEST_F(KernargTest, EveryExplicitPointerKindAndGroupRelativeConstants) {
  std::vector<Argument> arguments;
  const Kind pointer_kinds[] = {
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_IMAGE,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_SAMPLER,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_PIPE,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_QUEUE};
  std::array<uint64_t, 5> addresses;
  for (size_t i = 0; i < addresses.size(); ++i) {
    addresses[i] = UINT64_C(0x123456780000) + i * 4096;
    arguments.push_back(Arg(pointer_kinds[i], i * 8, 8));
  }
  arguments.push_back(Arg(
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_DYNAMIC_SHARED_POINTER, 40, 4));
  IREE_ASSERT_OK(Build(std::move(arguments), 44));
  EXPECT_EQ(layout_->binding_count, addresses.size());
  EXPECT_EQ(layout_->constant_byte_length, 4u);
  const uint8_t relative_lds[] = {0x34, 0x12, 0, 0};
  std::array<uint8_t, 44> bytes;
  iree_hal_amd_gpu_kernarg_layout_emplace(
      layout_, relative_lds, addresses.data(), nullptr, bytes.data());
  for (size_t i = 0; i < addresses.size(); ++i) {
    EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data() + i * 8), addresses[i]);
  }
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data() + 40), 0x1234u);
}

TEST_F(KernargTest, EveryHiddenFieldUsesItsDeclaredDestinationAndWidth) {
  const iree_hal_amd_gpu_kernarg_inputs_t inputs = {
      .global_offset = {11, 12, 13},
      .block_count = {21, 22, 23},
      .group_size = {31, 32, 33},
      .remainder = {1, 2, 3},
      .grid_dimensions = 3,
      .dynamic_lds_size = 0x12345678,
      .services =
          {
              .printf_buffer = 0x100000001,
              .hostcall_buffer = 0x200000002,
              .default_queue = 0x300000003,
              .completion_action = 0x400000004,
              .grid_sync = 0x500000005,
              .heap = 0x600000006,
              .queue_pointer = 0x700000007,
          },
      .apertures = {.private_base = 0x31415926, .shared_base = 0x27182818},
  };
  // Independent ABI oracle in metadata-kind order, with deliberately sparse
  // destinations instead of the usual V5 suffix placement.
  const uint64_t values[] = {11,          12,          13,          0x100000001,
                             0x200000002, 0x300000003, 0x400000004, 0x500000005,
                             21,          22,          23,          31,
                             32,          33,          1,           2,
                             3,           3,           0x600000006, 0x12345678,
                             0x31415926,  0x27182818,  0x700000007};
  const uint32_t widths[] = {8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 2,
                             2, 2, 2, 2, 2, 2, 8, 4, 4, 4, 8};
  static_assert(IREE_ARRAYSIZE(values) ==
                IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_QUEUE_PTR -
                    IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE);
  std::vector<Argument> arguments = {
      Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE, 0, 13)};
  std::vector<uint8_t> expected((IREE_ARRAYSIZE(values) + 1) * 16, 0);
  for (size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    const uint32_t offset = (i + 1) * 16;
    arguments.push_back(Arg(
        static_cast<Kind>(
            IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GLOBAL_OFFSET_X + i),
        offset, widths[i]));
    for (uint32_t byte = 0; byte < widths[i]; ++byte) {
      expected[offset + byte] = values[i] >> (8 * byte);
    }
  }
  IREE_ASSERT_OK(Build(std::move(arguments), expected.size()));
  EXPECT_EQ(layout_->requirements, 0x1ffu);
  EXPECT_EQ(layout_->parameter_count, 0u);
  EXPECT_EQ(layout_->runtime_copy_count, IREE_ARRAYSIZE(values));
  std::vector<uint8_t> actual(expected.size(), 0xa5);
  iree_hal_amd_gpu_kernarg_layout_emplace(layout_, nullptr, nullptr, &inputs,
                                          actual.data());
  EXPECT_EQ(actual, expected);
}

TEST_F(KernargTest, NativePrefixPreservesExplicitHolesAndPopulatesRuntime) {
  IREE_ASSERT_OK(Build(
      {Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE, 4, 4),
       Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X, 12, 4),
       Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GROUP_SIZE_X, 20, 2)},
      24));
  const iree_hal_amd_gpu_kernarg_inputs_t inputs = {
      .block_count = {7, 0, 0},
      .group_size = {32, 0, 0},
  };
  const uint8_t native[] = {0xaa, 0xbb, 0xcc, 0xdd, 1, 2, 3, 4};
  std::array<uint8_t, 24> bytes;
  iree_hal_amd_gpu_kernarg_layout_emplace_native(
      layout_, iree_make_const_byte_span(native, sizeof(native)), &inputs,
      bytes.data());
  const std::array<uint8_t, 24> expected = {
      0xaa, 0xbb, 0xcc, 0xdd, 1, 2, 3, 4, 0,  0, 0, 0,
      7,    0,    0,    0,    0, 0, 0, 0, 32, 0, 0, 0};
  EXPECT_EQ(bytes, expected);
  std::array<uint8_t, 24> full;
  full.fill(0x5a);
  iree_hal_amd_gpu_kernarg_layout_emplace_native(
      layout_, iree_make_const_byte_span(full.data(), full.size()), &inputs,
      bytes.data());
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data() + 12), 7u);
  EXPECT_EQ(iree_unaligned_load_le_u16(bytes.data() + 20), 32u);
  EXPECT_EQ(bytes[0], 0x5a);
  EXPECT_EQ(bytes[23], 0x5a);
}

TEST_F(KernargTest, EmptyLayoutsNeedNoDispatchStorage) {
  for (bool has_metadata : {false, true}) {
    kernel_.has_argument_metadata = has_metadata;
    IREE_ASSERT_OK(Build({}, 0));
    EXPECT_EQ(layout_->byte_length, 0u);
    EXPECT_EQ(layout_->parameter_count, 0u);
    iree_hal_amd_gpu_kernarg_layout_emplace(layout_, nullptr, nullptr, nullptr,
                                            nullptr);
    iree_hal_amd_gpu_kernarg_layout_emplace_native(
        layout_, iree_const_byte_span_empty(), nullptr, nullptr);
  }
}

TEST_F(KernargTest, RejectsUnknownAndUnavailableLayoutsBeforeAllocation) {
  kernel_.has_argument_metadata = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Build({}, 8));
  EXPECT_EQ(layout_, nullptr);
  kernel_.has_argument_metadata = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      Build({Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_UNKNOWN, 0, 8)}, 8));
  EXPECT_EQ(layout_, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Build({}, UINT16_MAX + 1u));
  EXPECT_EQ(layout_, nullptr);
}

TEST_F(KernargTest, SemanticWidthsAreCheckedOnceDuringConstruction) {
  const Kind kinds[] = {
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_IMAGE,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_SAMPLER,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_PIPE,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_QUEUE,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_DYNAMIC_SHARED_POINTER,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GROUP_SIZE_X,
      IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_HOSTCALL_BUFFER};
  for (Kind kind : kinds) {
    SCOPED_TRACE(static_cast<int>(kind));
    for (uint32_t size : {0u, 1u, 3u, 16u}) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            Build({Arg(kind, 0, size)}, 16));
      EXPECT_EQ(layout_, nullptr);
    }
  }
}

TEST_F(KernargTest, SupportsLargestRepresentableByValueArgument) {
  IREE_ASSERT_OK(Build(
      {Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE, 0, UINT16_MAX)},
      UINT16_MAX));
  EXPECT_EQ(layout_->constant_byte_length, UINT16_MAX);
  std::vector<uint8_t> source(UINT16_MAX, 0x5a), target(UINT16_MAX, 0);
  iree_hal_amd_gpu_kernarg_layout_emplace(layout_, source.data(), nullptr,
                                          nullptr, target.data());
  EXPECT_EQ(target, source);
}

TEST_F(KernargTest, AllocationFailureLeavesNoOwner) {
  iree_hal_amd_gpu_kernarg_layout_t* output = layout_;
  int attempts = 0;
  const iree_allocator_t allocator = {
      &attempts, [](void* self, iree_allocator_command_t, const void*, void**) {
        ++*static_cast<int*>(self);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "denied");
      }};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amd_gpu_kernarg_layout_allocate(&kernel_, allocator, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(attempts, 1);
  kernel_.kernarg_segment_size = UINT16_MAX + 1u;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_kernarg_layout_allocate(&kernel_, allocator, &output));
  EXPECT_EQ(attempts, 1);
}

TEST(KernargCorpusTest, RealHipReflectionReconstructsNativeArguments) {
  const auto* files = iree_hal_amd_gpu_hip_images_create();
  for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
    SCOPED_TRACE(files[i].name);
    auto* source = testing::ByteSequence::Create(
        iree_make_const_byte_span(files[i].data, files[i].size), 17);
    iree_hal_amd_gpu_image_t image = {};
    IREE_CHECK_OK(iree_hal_amd_gpu_hsaco_image_initialize(
        source, iree_allocator_system(), &image));
    iree_byte_sequence_release(source);
    for (size_t j = 0; j < image.metadata.kernel_count; ++j) {
      const auto* kernel = &image.metadata.kernels[j];
      const bool count_producer =
          iree_string_view_equal(kernel->name, IREE_SV("image_counts"));
      iree_hal_amd_gpu_kernarg_layout_t* layout = nullptr;
      IREE_CHECK_OK(iree_hal_amd_gpu_kernarg_layout_allocate(
          kernel, iree_allocator_system(), &layout));
      ASSERT_EQ(layout->constant_byte_length, count_producer ? 0u : 4u);
      ASSERT_EQ(layout->binding_count, count_producer ? 2u : 1u);
      EXPECT_EQ(layout->parameter_count, 2u);
      EXPECT_EQ(layout->requirements, 0u);
      const auto* parameters =
          iree_hal_amd_gpu_kernarg_layout_parameters(layout);
      EXPECT_EQ(parameters[0].native_abi_offset, 0u);
      EXPECT_EQ(parameters[1].native_abi_offset, 8u);
      const uint64_t bindings[] = {0x123456789abc, 0xABCDEF987654};
      const uint8_t constants[] = {3, 2, 1, 0};
      const iree_hal_amd_gpu_kernarg_inputs_t inputs = {};
      std::vector<uint8_t> actual(layout->byte_length, 0);
      iree_hal_amd_gpu_kernarg_layout_emplace(layout, constants, bindings,
                                              &inputs, actual.data());
      EXPECT_EQ(iree_unaligned_load_le_u64(actual.data()), bindings[0]);
      if (count_producer) {
        EXPECT_EQ(iree_unaligned_load_le_u64(actual.data() + 8), bindings[1]);
      } else {
        EXPECT_EQ(iree_unaligned_load_le_u32(actual.data() + 8), 0x010203u);
      }
      iree_hal_amd_gpu_kernarg_layout_free(layout);
    }
    iree_hal_amd_gpu_image_deinitialize(&image);
  }
}

TEST_F(KernargTest, IndirectCountsRetainCoalescedNativeDestinations) {
  IREE_ASSERT_OK(Build(
      {Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X, 8, 4),
       Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Y, 12, 4),
       Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Z, 16, 4),
       Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X, 32, 4),
       Arg(IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Z, 40,
           4)},
      64));
  ASSERT_EQ(layout_->indirect_copy_count, 3u);
  arguments_.clear();
  const auto* copies = iree_hal_amd_gpu_kernarg_layout_indirect_copies(layout_);
  const std::array<std::array<uint16_t, 3>, 3> expected = {
      {{8, 0, 12}, {32, 0, 4}, {40, 8, 4}}};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(copies[i].target_offset, expected[i][0]);
    EXPECT_EQ(copies[i].source_offset, expected[i][1]);
    EXPECT_EQ(copies[i].length, expected[i][2]);
  }
}

}  // namespace
}  // namespace iree::hal::amd::gpu
