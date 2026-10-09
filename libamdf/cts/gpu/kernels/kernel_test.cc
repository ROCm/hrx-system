// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/kernel.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned.h"
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma.h"
#include "libamdf/cts/gpu/kernels/device_sdma_batched.h"
#include "libamdf/cts/gpu/kernels/device_sdma_batched_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_consumer_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_lookahead.h"
#include "libamdf/cts/gpu/kernels/device_sdma_reader_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_staged.h"
#include "libamdf/cts/gpu/kernels/device_sdma_transfer_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_upload_kernels.h"
#include "libamdf/cts/gpu/kernels/geometry_ids.h"
#include "libamdf/cts/gpu/kernels/geometry_ids_kernels.h"
#include "libamdf/cts/gpu/kernels/lds_exchange.h"
#include "libamdf/cts/gpu/kernels/lds_exchange_kernels.h"
#include "libamdf/cts/gpu/kernels/pattern_fill_unaligned.h"
#include "libamdf/cts/gpu/kernels/pattern_fill_unaligned_kernels.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip_kernels.h"
#include "libamdf/cts/gpu/kernels/product_test.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_alternate_kernels.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

using kernels::testing::CheckArgumentLayout;

TEST(KernelTest, DeviceSdmaProductsPreserveTheCallerContract) {
  using Arguments = kernels::device_sdma::Arguments;
  constexpr std::array<uint32_t, 16> kOffsets = {
      offsetof(Arguments, ring),
      offsetof(Arguments, read_index),
      offsetof(Arguments, write_index),
      offsetof(Arguments, notification),
      offsetof(Arguments, destinations),
      offsetof(Arguments, completion),
      offsetof(Arguments, records),
      offsetof(Arguments, source_address),
      offsetof(Arguments, destination_address),
      offsetof(Arguments, completion_address),
      offsetof(Arguments, capacity),
      offsetof(Arguments, round_count),
      offsetof(Arguments, seed),
      offsetof(Arguments, copy_control),
      offsetof(Arguments, fence_header),
      offsetof(Arguments, cache_flags)};
  constexpr std::array<uint32_t, 16> kLengths = {8, 8, 8, 8, 8, 8, 8, 8,
                                                 8, 8, 8, 4, 4, 4, 4, 4};
  constexpr std::array<std::string_view, 16> kKinds = {
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "global_buffer", "global_buffer", "by_value",
      "by_value",      "by_value",      "by_value",      "by_value",
      "by_value",      "by_value",      "by_value",      "by_value"};
  std::string targets;
  for (const auto& kernel : kernels::device_sdma::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    if (!targets.empty()) {
      targets += ",";
    }
    targets += kernel.target;
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, sizeof(Arguments),
                        alignof(Arguments));
    EXPECT_GE(kernel.arguments.byte_length,
              offsetof(Arguments, cache_flags) + sizeof(uint32_t));
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{1, 1, 1}));
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
  RecordProperty("device_sdma_compiled_targets", targets);
}

TEST(KernelTest, BatchedDeviceSdmaProductsPreserveTheCallerContract) {
  using Arguments = kernels::device_sdma_batched::Arguments;
  constexpr std::array<uint32_t, 19> kOffsets = {
      offsetof(Arguments, ring),
      offsetof(Arguments, read_index),
      offsetof(Arguments, write_index),
      offsetof(Arguments, notification),
      offsetof(Arguments, destinations),
      offsetof(Arguments, completion),
      offsetof(Arguments, records),
      offsetof(Arguments, statistics),
      offsetof(Arguments, source_address),
      offsetof(Arguments, destination_address),
      offsetof(Arguments, completion_address),
      offsetof(Arguments, capacity),
      offsetof(Arguments, round_count),
      offsetof(Arguments, batch_size),
      offsetof(Arguments, credit_count),
      offsetof(Arguments, seed),
      offsetof(Arguments, copy_control),
      offsetof(Arguments, fence_header),
      offsetof(Arguments, cache_flags)};
  constexpr std::array<uint32_t, 19> kLengths = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                                 8, 8, 4, 4, 4, 4, 4, 4, 4};
  constexpr std::array<std::string_view, 19> kKinds = {
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "by_value",      "by_value",      "by_value",      "by_value",
      "by_value",      "by_value",      "by_value",      "by_value",
      "by_value",      "by_value",      "by_value"};
  std::string targets;
  for (const auto& kernel : kernels::device_sdma_batched::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    if (!targets.empty()) {
      targets += ",";
    }
    targets += kernel.target;
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, sizeof(Arguments),
                        alignof(Arguments));
    EXPECT_GE(kernel.arguments.byte_length,
              offsetof(Arguments, cache_flags) + sizeof(uint32_t));
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{1, 1, 1}));
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
  RecordProperty("device_sdma_batched_compiled_targets", targets);
}

TEST(KernelTest, StagedDeviceSdmaProductsPreserveTheCallerContract) {
  using Arguments = kernels::device_sdma_staged::TransferArguments;
  constexpr std::array<uint32_t, 22> kOffsets = {
      offsetof(Arguments, ring),
      offsetof(Arguments, read_index),
      offsetof(Arguments, write_index),
      offsetof(Arguments, notification),
      offsetof(Arguments, completion),
      offsetof(Arguments, state),
      offsetof(Arguments, selection),
      offsetof(Arguments, lengths),
      offsetof(Arguments, readback),
      offsetof(Arguments, source_address),
      offsetof(Arguments, input_address),
      offsetof(Arguments, output_address),
      offsetof(Arguments, readback_address),
      offsetof(Arguments, completion_address),
      offsetof(Arguments, capacity),
      offsetof(Arguments, slot_byte_length),
      offsetof(Arguments, job_index),
      offsetof(Arguments, slot_count),
      offsetof(Arguments, phase),
      offsetof(Arguments, copy_control),
      offsetof(Arguments, fence_header),
      offsetof(Arguments, cache_flags)};
  constexpr std::array<uint32_t, 22> kLengths = {
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4};
  constexpr std::array<std::string_view, 22> kKinds = {
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "by_value",      "by_value",      "by_value",
      "by_value",      "by_value",      "by_value",      "by_value",
      "by_value",      "by_value",      "by_value",      "by_value",
      "by_value",      "by_value"};
  for (const auto& kernel : kernels::device_sdma_transfer::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, sizeof(Arguments),
                        alignof(Arguments));
    EXPECT_EQ(kernel.arguments.byte_length, 152u);
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{1, 1, 1}));
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }

  using Consumer = kernels::device_sdma_staged::ConsumerArguments;
  constexpr std::array<uint32_t, 4> kConsumerOffsets = {
      offsetof(Consumer, input), offsetof(Consumer, output),
      offsetof(Consumer, selection), offsetof(Consumer, slot_byte_length)};
  constexpr std::array<uint32_t, 4> kConsumerLengths = {8, 8, 8, 8};
  constexpr std::array<std::string_view, 4> kConsumerKinds = {
      "global_buffer", "global_buffer", "global_buffer", "by_value"};
  for (const auto& kernel : kernels::device_sdma_consumer::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kConsumerOffsets, kConsumerLengths,
                        kConsumerKinds, sizeof(Consumer), alignof(Consumer));
    EXPECT_EQ(kernel.arguments.byte_length, 32u);
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{64, 1, 1}));
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
}

TEST(KernelTest, LookaheadDeviceSdmaProductsPreserveTheCallerContract) {
  using Upload = kernels::device_sdma_lookahead::UploadArguments;
  constexpr std::array<uint32_t, 20> kOffsets = {
      offsetof(Upload, ring),
      offsetof(Upload, read_index),
      offsetof(Upload, write_index),
      offsetof(Upload, notification),
      offsetof(Upload, completion),
      offsetof(Upload, state),
      offsetof(Upload, selection),
      offsetof(Upload, request),
      offsetof(Upload, previous_readers),
      offsetof(Upload, lengths),
      offsetof(Upload, source_address),
      offsetof(Upload, input_address),
      offsetof(Upload, completion_address),
      offsetof(Upload, capacity),
      offsetof(Upload, slot_byte_length),
      offsetof(Upload, slot),
      offsetof(Upload, generation),
      offsetof(Upload, copy_control),
      offsetof(Upload, fence_header),
      offsetof(Upload, cache_flags)};
  constexpr std::array<uint32_t, 20> kLengths = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                                 8, 8, 8, 8, 8, 4, 4, 4, 4, 4};
  constexpr std::array<std::string_view, 20> kKinds = {
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "global_buffer", "by_value",      "by_value",
      "by_value",      "by_value",      "by_value",      "by_value",
      "by_value",      "by_value",      "by_value",      "by_value"};
  for (const auto& kernel : kernels::device_sdma_upload::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, sizeof(Upload),
                        alignof(Upload));
    EXPECT_EQ(kernel.arguments.byte_length, 144u);
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{1, 1, 1}));
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
  using Reader = kernels::device_sdma_lookahead::ReaderArguments;
  constexpr std::array<uint32_t, 6> kReaderOffsets = {
      offsetof(Reader, input),       offsetof(Reader, output),
      offsetof(Reader, selection),   offsetof(Reader, started),
      offsetof(Reader, round_count), offsetof(Reader, reader)};
  constexpr std::array<uint32_t, 6> kReaderLengths = {8, 8, 8, 8, 4, 4};
  constexpr std::array<std::string_view, 6> kReaderKinds = {
      "global_buffer", "global_buffer", "global_buffer",
      "global_buffer", "by_value",      "by_value"};
  for (const auto& kernel : kernels::device_sdma_reader::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kReaderOffsets, kReaderLengths, kReaderKinds,
                        sizeof(Reader), alignof(Reader));
    EXPECT_EQ(kernel.arguments.byte_length, 40u);
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{64, 1, 1}));
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
}

TEST(KernelTest, TransformProductsPreserveTheCallerContract) {
  using Arguments = kernels::transform::Arguments;
  for (const auto* set : {&kernels::transform::kKernels,
                          &kernels::transform_alternate::kKernels}) {
    for (const auto& kernel : set->variants) {
      SCOPED_TRACE(kernel.target);
      CheckArgumentLayout(kernel, kernels::transform::kArgumentByteOffsets,
                          kernels::transform::kArgumentByteLengths,
                          kernels::transform::kArgumentValueKinds,
                          sizeof(Arguments), alignof(Arguments));
      EXPECT_EQ(kernel.arguments.byte_length, 24u);
      EXPECT_EQ(kernel.required_workgroup_size,
                (std::array<uint32_t, 3>{64, 1, 1}));
      EXPECT_EQ(kernel.group_segment_byte_length, 0u);
      EXPECT_EQ(kernel.private_segment_byte_length, 0u);
      EXPECT_EQ(kernel.program.code_properties,
                kernel.wavefront_size == 32 ? 0x408u : 8u);
      EXPECT_EQ(kernel.program.argument_preload, 0u);
      EXPECT_EQ(kernel.program.resource2 & 0x1fffu, 0x84u);
    }
  }
}

TEST(KernelTest, ExecutableReplacementRetainsDescriptorsAndFetchBounds) {
  for (const auto& kernel : kernels::transform::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    const auto variants = kernels::transform_alternate::kKernels.variants;
    const auto alternate =
        std::ranges::find_if(variants, [&](const auto& item) {
          return std::strcmp(kernel.target, item.target) == 0;
        });
    ASSERT_NE(alternate, variants.end());
    EXPECT_EQ(kernel.executable.byte_length, alternate->executable.byte_length);
    EXPECT_LE((kernel.executable.byte_length + 255u) & ~255u, 4096u);
    EXPECT_EQ(kernel.executable.descriptor_byte_offset, 0u);
    EXPECT_EQ(alternate->executable.descriptor_byte_offset, 0u);
    EXPECT_EQ(kernel.entry_byte_offset, alternate->entry_byte_offset);
    // Both arithmetics remain valid if instruction fetch observes the prior
    // program. The behavioral replacement case separately checks new results.
    EXPECT_EQ(
        std::memcmp(kernel.executable.words, alternate->executable.words, 64),
        0);
  }
}

TEST(KernelTest, PrivateProductsPreserveTheCallerContract) {
  using Arguments = kernels::private_roundtrip::Arguments;
  for (const auto& kernel : kernels::private_roundtrip::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel,
                        kernels::private_roundtrip::kArgumentByteOffsets,
                        kernels::private_roundtrip::kArgumentByteLengths,
                        kernels::private_roundtrip::kArgumentValueKinds,
                        sizeof(Arguments), alignof(Arguments));
    EXPECT_EQ(kernel.arguments.byte_length, sizeof(Arguments));
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{64, 1, 1}));
    EXPECT_GT(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
}

TEST(KernelTest, ByteCopyProductsPreserveTheCallerContract) {
  using Arguments = kernels::byte_copy_unaligned::Arguments;
  constexpr std::array<uint32_t, 6> kOffsets = {
      offsetof(Arguments, source),      offsetof(Arguments, target),
      offsetof(Arguments, byte_length), offsetof(Arguments, grid_size_x),
      offsetof(Arguments, grid_size_y), offsetof(Arguments, workgroup_size_x)};
  constexpr std::array<uint32_t, 6> kLengths = {8, 8, 8, 4, 4, 4};
  constexpr std::array<std::string_view, 6> kKinds = {
      "global_buffer", "global_buffer", "by_value",
      "by_value",      "by_value",      "by_value"};
  for (const auto& kernel : kernels::byte_copy_unaligned::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, 64,
                        alignof(Arguments));
    EXPECT_GE(kernel.arguments.byte_length,
              offsetof(Arguments, workgroup_size_x) + sizeof(uint32_t));
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{64, 1, 1}));
    EXPECT_GE(kernel.maximum_workgroup_size, 64u);
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
}

TEST(KernelTest, PatternFillProductsPreserveTheCallerContract) {
  using Arguments = kernels::pattern_fill_unaligned::Arguments;
  constexpr std::array<uint32_t, 6> kOffsets = {
      offsetof(Arguments, target),      offsetof(Arguments, byte_length),
      offsetof(Arguments, pattern),     offsetof(Arguments, grid_size_x),
      offsetof(Arguments, grid_size_y), offsetof(Arguments, workgroup_size_x)};
  constexpr std::array<uint32_t, 6> kLengths = {8, 8, 8, 4, 4, 4};
  constexpr std::array<std::string_view, 6> kKinds = {
      "global_buffer", "by_value", "by_value",
      "by_value",      "by_value", "by_value"};
  for (const auto& kernel :
       kernels::pattern_fill_unaligned::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, 64,
                        alignof(Arguments));
    EXPECT_GE(kernel.arguments.byte_length,
              offsetof(Arguments, workgroup_size_x) + sizeof(uint32_t));
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{64, 1, 1}));
    EXPECT_GE(kernel.maximum_workgroup_size, 64u);
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
}

TEST(KernelTest, GeometryProductsPreserveTheCallerContract) {
  using Arguments = kernels::geometry_ids::Arguments;
  constexpr std::array<uint32_t, 7> kOffsets = {
      offsetof(Arguments, output),
      offsetof(Arguments, workgroup_size),
      offsetof(Arguments, workgroup_size) + sizeof(uint32_t),
      offsetof(Arguments, workgroup_size) + 2 * sizeof(uint32_t),
      offsetof(Arguments, output_pitches),
      offsetof(Arguments, output_pitches) + sizeof(uint32_t),
      offsetof(Arguments, epoch)};
  constexpr std::array<uint32_t, 7> kLengths = {8, 4, 4, 4, 4, 4, 4};
  constexpr std::array<std::string_view, 7> kKinds = {
      "global_buffer", "by_value", "by_value", "by_value",
      "by_value",      "by_value", "by_value"};
  for (const auto& kernel : kernels::geometry_ids::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    CheckArgumentLayout(kernel, kOffsets, kLengths, kKinds, sizeof(Arguments),
                        alignof(Arguments));
    EXPECT_EQ(kernel.arguments.byte_length, sizeof(Arguments));
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{0, 0, 0}));
    EXPECT_GE(kernel.maximum_workgroup_size, 64u);
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_EQ(kernel.group_segment_byte_length, 0u);
  }
}

TEST(KernelTest, EveryLdsProductPreservesTheCallerContract) {
  using Arguments = kernels::lds_exchange::Arguments;
  ASSERT_FALSE(kernels::lds_exchange::kKernels.variants.empty());
  for (const auto& kernel : kernels::lds_exchange::kKernels.variants) {
    SCOPED_TRACE(kernel.target);
    EXPECT_TRUE(
        std::ranges::equal(kernel.arguments.byte_offsets,
                           kernels::lds_exchange::kArgumentByteOffsets));
    EXPECT_TRUE(
        std::ranges::equal(kernel.arguments.byte_lengths,
                           kernels::lds_exchange::kArgumentByteLengths));
    EXPECT_TRUE(std::ranges::equal(kernel.arguments.value_kinds,
                                   kernels::lds_exchange::kArgumentValueKinds));
    EXPECT_EQ(kernel.arguments.byte_length, sizeof(Arguments));
    EXPECT_EQ(alignof(Arguments) % kernel.arguments.alignment, 0u);
    EXPECT_EQ(kernel.required_workgroup_size,
              (std::array<uint32_t, 3>{128, 1, 1}));
    EXPECT_EQ(kernel.workgroup_size(), 128u);
    EXPECT_EQ(kernel.group_segment_byte_length, 512u);
    EXPECT_EQ(kernel.private_segment_byte_length, 0u);
    EXPECT_TRUE(kernel.wavefront_size == 32 || kernel.wavefront_size == 64);
    EXPECT_EQ(kernel.program.code_properties,
              kernel.wavefront_size == 32 ? 0x408u : 8u);
    EXPECT_EQ(kernel.program.argument_preload, 0u);
    EXPECT_EQ(kernel.program.resource2 & 0x1fffu, 0x84u);
    EXPECT_EQ(kernel.entry_byte_offset % 256, 0u);
    EXPECT_LT(kernel.entry_byte_offset, kernel.executable.byte_length);
    EXPECT_GE(kernel.text_byte_length, kernel.entry_byte_length);
    EXPECT_EQ(std::strlen(kernel.executable.sha256), 64u);
    EXPECT_EQ(std::strlen(kernel.hsaco_sha256), 64u);
  }
}

TEST(KernelTest, PhysicalRevisionSelectsTheInstructionEncodingOverlay) {
  constexpr kernels::Kernel kVariants[] = {{.target = "gfx1250"},
                                           {.target = "gfx1250-a0"}};
  const kernels::KernelSet products = {kVariants};
  amdf_gpu_endpoint_info_t endpoint = {};
  endpoint.gfx_ip = {12, 5, 0};
  endpoint.asic_revision = 0;
  const auto* a0 = products.Find(endpoint);
  ASSERT_NE(a0, nullptr);
  EXPECT_STREQ(a0->target, "gfx1250-a0");
  endpoint.asic_revision = 1;
  const auto* b0 = products.Find(endpoint);
  ASSERT_NE(b0, nullptr);
  EXPECT_STREQ(b0->target, "gfx1250");
  EXPECT_NE(a0, b0);
  endpoint.asic_revision = 2;
  EXPECT_EQ(products.Find(endpoint), nullptr);
}

TEST(KernelTest, MissingPhysicalProductDoesNotSelectANearbyProcessor) {
  constexpr kernels::Kernel kVariants[] = {{.target = "gfx1151"}};
  const kernels::KernelSet products = {kVariants};
  amdf_gpu_endpoint_info_t endpoint = {.gfx_ip = {11, 5, 15}};
  EXPECT_EQ(products.Find(endpoint), nullptr);
}

}  // namespace
