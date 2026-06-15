// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/kernarg_layout.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

static std::vector<uint8_t> AllocateStorage(
    iree_host_size_t binding_count, iree_host_size_t constant_span_count) {
  iree_host_size_t storage_byte_length = 0;
  IREE_CHECK_OK(iree_hal_amdgpu_kernarg_layout_storage_size(
      binding_count, constant_span_count, &storage_byte_length));
  return std::vector<uint8_t>(storage_byte_length);
}

TEST(KernargLayoutTest, InitializesPackedBindingPrefixLayout) {
  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/0},
      {/*.target_qword_index=*/1},
  };
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/16,
          /*.source_byte_offset=*/0,
          /*.byte_length=*/8,
      },
  };
  std::vector<uint8_t> storage = AllocateStorage(
      IREE_ARRAYSIZE(binding_slots), IREE_ARRAYSIZE(constant_spans));
  iree_hal_amdgpu_kernarg_layout_t* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());

  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/24,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/8,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  EXPECT_EQ(layout->kernarg_byte_length, 24);
  EXPECT_EQ(layout->kernarg_alignment, 8);
  EXPECT_EQ(layout->binding_count, 2);
  EXPECT_EQ(layout->constant_span_count, 1);
  EXPECT_EQ(layout->constant_byte_length, 8);
  EXPECT_TRUE(iree_all_bits_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX |
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_CONTIGUOUS_CONSTANTS));
  EXPECT_FALSE(iree_any_bit_set(
      layout->flags, IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL));

  const iree_hal_amdgpu_kernarg_binding_slot_t* stored_bindings =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  EXPECT_EQ(stored_bindings[0].target_qword_index, 0);
  EXPECT_EQ(stored_bindings[1].target_qword_index, 1);

  const iree_hal_amdgpu_kernarg_constant_span_t* stored_constants =
      iree_hal_amdgpu_kernarg_layout_constant_spans(layout);
  EXPECT_EQ(stored_constants[0].target_byte_offset, 16);
  EXPECT_EQ(stored_constants[0].source_byte_offset, 0);
  EXPECT_EQ(stored_constants[0].byte_length, 8);
}

TEST(KernargLayoutTest, MarksSparseInterleavedLayoutForZeroFill) {
  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/1},
  };
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/0,
          /*.source_byte_offset=*/0,
          /*.byte_length=*/3,
      },
      {
          /*.target_byte_offset=*/20,
          /*.source_byte_offset=*/3,
          /*.byte_length=*/2,
      },
  };
  std::vector<uint8_t> storage = AllocateStorage(
      IREE_ARRAYSIZE(binding_slots), IREE_ARRAYSIZE(constant_spans));
  iree_hal_amdgpu_kernarg_layout_t* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());

  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/24,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/5,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  EXPECT_FALSE(iree_any_bit_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX));
  EXPECT_TRUE(iree_any_bit_set(
      layout->flags, IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL));
  EXPECT_TRUE(iree_any_bit_set(
      layout->flags, IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_CONTIGUOUS_CONSTANTS));
}

TEST(KernargLayoutTest, MarksImplicitArgsLayoutForZeroFill) {
  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/0},
  };
  std::vector<uint8_t> storage = AllocateStorage(IREE_ARRAYSIZE(binding_slots),
                                                 /*constant_span_count=*/0);
  iree_hal_amdgpu_kernarg_layout_t* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());

  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/272,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/16,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  EXPECT_EQ(layout->implicit_args_byte_offset, 16);
  EXPECT_TRUE(iree_all_bits_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS |
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL));
}

TEST(KernargLayoutTest, EmplacesPackedBindingPrefixLayout) {
  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/0},
      {/*.target_qword_index=*/1},
  };
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/16,
          /*.source_byte_offset=*/0,
          /*.byte_length=*/4,
      },
  };
  std::vector<uint8_t> storage = AllocateStorage(
      IREE_ARRAYSIZE(binding_slots), IREE_ARRAYSIZE(constant_spans));
  iree_hal_amdgpu_kernarg_layout_t* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/20,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/4,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  const uint64_t binding_ptrs[] = {0xA0A1A2A3A4A5A6A7ull,
                                   0xB0B1B2B3B4B5B6B7ull};
  const uint8_t constants[] = {0xC0, 0xC1, 0xC2, 0xC3};
  uint8_t kernargs[20] = {};
  memset(kernargs, 0xCD, sizeof(kernargs));
  iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
      layout, binding_ptrs,
      iree_make_const_byte_span(constants, sizeof(constants)), kernargs);

  uint64_t stored_binding0 = 0;
  uint64_t stored_binding1 = 0;
  memcpy(&stored_binding0, kernargs, sizeof(stored_binding0));
  memcpy(&stored_binding1, kernargs + 8, sizeof(stored_binding1));
  EXPECT_EQ(stored_binding0, binding_ptrs[0]);
  EXPECT_EQ(stored_binding1, binding_ptrs[1]);
  EXPECT_EQ(memcmp(kernargs + 16, constants, sizeof(constants)), 0);
}

TEST(KernargLayoutTest, EmplacesSparseInterleavedLayoutWithZeroFill) {
  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/1},
  };
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/0,
          /*.source_byte_offset=*/0,
          /*.byte_length=*/3,
      },
      {
          /*.target_byte_offset=*/20,
          /*.source_byte_offset=*/3,
          /*.byte_length=*/2,
      },
  };
  std::vector<uint8_t> storage = AllocateStorage(
      IREE_ARRAYSIZE(binding_slots), IREE_ARRAYSIZE(constant_spans));
  iree_hal_amdgpu_kernarg_layout_t* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/24,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/5,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  const uint64_t binding_ptrs[] = {0xA0A1A2A3A4A5A6A7ull};
  const uint8_t constants[] = {0xC0, 0xC1, 0xC2, 0xC3, 0xC4};
  uint8_t kernargs[24] = {};
  memset(kernargs, 0xCD, sizeof(kernargs));
  iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
      layout, binding_ptrs,
      iree_make_const_byte_span(constants, sizeof(constants)), kernargs);

  uint8_t expected[24] = {};
  memcpy(expected, constants, 3);
  memcpy(expected + 8, binding_ptrs, sizeof(binding_ptrs));
  memcpy(expected + 20, constants + 3, 2);
  EXPECT_EQ(memcmp(kernargs, expected, sizeof(expected)), 0);
}

TEST(KernargLayoutTest, RejectsKernargPacketsBeyondLayoutLimit) {
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_MAX_BYTE_LENGTH + 1u,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/{}, /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
  };
  std::vector<uint8_t> storage = AllocateStorage(/*binding_count=*/0,
                                                 /*constant_span_count=*/0);

  EXPECT_THAT(
      Status(iree_hal_amdgpu_kernarg_layout_initialize(
          &params, storage.size(),
          reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data()))),
      StatusIs(StatusCode::kOutOfRange));
}

TEST(KernargLayoutTest, RejectsOverlappingTargetRanges) {
  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/0},
  };
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/4,
          /*.source_byte_offset=*/0,
          /*.byte_length=*/4,
      },
  };
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/16,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/4,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  std::vector<uint8_t> storage = AllocateStorage(
      IREE_ARRAYSIZE(binding_slots), IREE_ARRAYSIZE(constant_spans));

  EXPECT_THAT(
      Status(iree_hal_amdgpu_kernarg_layout_initialize(
          &params, storage.size(),
          reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data()))),
      StatusIs(StatusCode::kInvalidArgument));
}

TEST(KernargLayoutTest, RejectsConstantSourceGaps) {
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/0,
          /*.source_byte_offset=*/2,
          /*.byte_length=*/2,
      },
  };
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/8,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/4,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/{},
      /*.binding_slots=*/{},
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  std::vector<uint8_t> storage = AllocateStorage(
      /*binding_count=*/0, IREE_ARRAYSIZE(constant_spans));

  EXPECT_THAT(
      Status(iree_hal_amdgpu_kernarg_layout_initialize(
          &params, storage.size(),
          reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data()))),
      StatusIs(StatusCode::kInvalidArgument));
}

TEST(KernargLayoutTest, RejectsTooLittleStorage) {
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/0,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/{}, /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
  };
  uint8_t storage[sizeof(iree_hal_amdgpu_kernarg_layout_t) - 1] = {0};

  EXPECT_THAT(
      Status(iree_hal_amdgpu_kernarg_layout_initialize(
          &params, sizeof(storage),
          reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage))),
      StatusIs(StatusCode::kResourceExhausted));
}

}  // namespace
}  // namespace iree::hal::amdgpu
