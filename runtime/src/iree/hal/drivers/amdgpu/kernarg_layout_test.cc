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

  const uint16_t kNone = IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE;
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/272,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/16,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/0,
      /*.constant_spans=*/nullptr,
      /*.implicit_args=*/
      {
          /*.block_count_offset=*/{16, 20, 24},
          /*.group_size_offset=*/{kNone, kNone, kNone},
          /*.grid_dims_offset=*/kNone,
          /*.dynamic_lds_size_offset=*/kNone,
      },
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

// Regression gate: a standard kernel declaring the complete implicit block at
// the fixed AMD ABI offsets must have block_count/group_size/grid_dims/
// dynamic_lds written at those offsets and every other implicit byte (global
// offset, printf/hostcall, remainder, padding) left zero -- matching the
// previous fixed-struct splat.
TEST(KernargLayoutTest, EmplacesFullImplicitArgsAtAbiOffsets) {
  std::vector<uint8_t> storage = AllocateStorage(0, 0);
  auto* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/128,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/0,
      /*.binding_count=*/0,
      /*.binding_slots=*/nullptr,
      /*.constant_span_count=*/0,
      /*.constant_spans=*/nullptr,
      /*.implicit_args=*/
      {
          /*.block_count_offset=*/{0, 4, 8},
          /*.group_size_offset=*/{12, 14, 16},
          /*.grid_dims_offset=*/64,
          /*.dynamic_lds_size_offset=*/120,
      },
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  uint8_t kernargs[128];
  memset(kernargs, 0xCD, sizeof(kernargs));
  const uint32_t workgroup_count[3] = {11u, 22u, 33u};
  const uint16_t workgroup_size[3] = {4u, 5u, 6u};
  iree_hal_amdgpu_kernarg_layout_emplace_implicit_args(
      layout, workgroup_count, workgroup_size,
      /*dynamic_workgroup_local_memory=*/777u, kernargs);

  uint8_t expected[128] = {0};
  memcpy(expected + 0, &workgroup_count[0], sizeof(uint32_t));
  memcpy(expected + 4, &workgroup_count[1], sizeof(uint32_t));
  memcpy(expected + 8, &workgroup_count[2], sizeof(uint32_t));
  memcpy(expected + 12, &workgroup_size[0], sizeof(uint16_t));
  memcpy(expected + 14, &workgroup_size[1], sizeof(uint16_t));
  memcpy(expected + 16, &workgroup_size[2], sizeof(uint16_t));
  const uint16_t grid_dims = 3u;
  memcpy(expected + 64, &grid_dims, sizeof(uint16_t));
  const uint32_t dynamic_lds = 777u;
  memcpy(expected + 120, &dynamic_lds, sizeof(uint32_t));
  EXPECT_EQ(memcmp(kernargs, expected, sizeof(expected)), 0);

  // Global offset (bytes 40..64) must stay zero, not garbage.
  for (int i = 40; i < 64; ++i) EXPECT_EQ(kernargs[i], 0) << "byte " << i;
}

// A hand-written assembly kernel (e.g. MIOpen Winograd) declares only a partial
// hidden block, so the region begins at an unaligned offset and none of the
// synthesized fields are present. Initialization must accept the unaligned
// region (no 8-byte-alignment rejection) and emplacement must zero the region
// (leaving global offset 0) without writing any field that would corrupt the
// kernel or the explicit arguments below the region.
TEST(KernargLayoutTest, EmplacesPartialImplicitArgsAndAcceptsUnalignedRegion) {
  std::vector<uint8_t> storage = AllocateStorage(0, 0);
  auto* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  const uint16_t kNone = IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE;
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/248,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/201,
      /*.binding_count=*/0,
      /*.binding_slots=*/nullptr,
      /*.constant_span_count=*/0,
      /*.constant_spans=*/nullptr,
      /*.implicit_args=*/
      {
          /*.block_count_offset=*/{kNone, kNone, kNone},
          /*.group_size_offset=*/{kNone, kNone, kNone},
          /*.grid_dims_offset=*/kNone,
          /*.dynamic_lds_size_offset=*/kNone,
      },
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, storage.size(), layout));

  uint8_t kernargs[248];
  memset(kernargs, 0xCD, sizeof(kernargs));
  const uint32_t workgroup_count[3] = {7u, 8u, 9u};
  const uint16_t workgroup_size[3] = {1u, 1u, 1u};
  iree_hal_amdgpu_kernarg_layout_emplace_implicit_args(
      layout, workgroup_count, workgroup_size,
      /*dynamic_workgroup_local_memory=*/42u, kernargs);

  for (int i = 0; i < 201; ++i) {
    EXPECT_EQ(kernargs[i], 0xCD) << "explicit byte " << i;
  }
  for (int i = 201; i < 248; ++i) {
    EXPECT_EQ(kernargs[i], 0) << "implicit byte " << i;
  }
}

// A declared implicit field whose [offset, offset+size) exceeds the kernarg
// segment must be rejected (the base-alignment check was removed in favor of
// per-field bounds validation).
TEST(KernargLayoutTest, RejectsImplicitFieldBeyondKernargSegment) {
  std::vector<uint8_t> storage = AllocateStorage(0, 0);
  auto* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  const uint16_t kNone = IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE;
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/128,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/64,
      /*.binding_count=*/0,
      /*.binding_slots=*/nullptr,
      /*.constant_span_count=*/0,
      /*.constant_spans=*/nullptr,
      /*.implicit_args=*/
      {
          /*.block_count_offset=*/{126, kNone, kNone},  // 126 + 4 > 128
          /*.group_size_offset=*/{kNone, kNone, kNone},
          /*.grid_dims_offset=*/kNone,
          /*.dynamic_lds_size_offset=*/kNone,
      },
  };
  EXPECT_THAT(Status(iree_hal_amdgpu_kernarg_layout_initialize(
                  &params, storage.size(), layout)),
              StatusIs(StatusCode::kOutOfRange));
}

// The device indirect-parameter patch writes hidden_block_count_x/y/z as a
// contiguous uint32[3] anchored at the x field, so a kernel declaring the three
// block_count fields at non-contiguous offsets (even when each is individually
// in bounds) must be rejected at layout construction rather than silently
// mispatched on device.
TEST(KernargLayoutTest, RejectsNonContiguousBlockCount) {
  std::vector<uint8_t> storage = AllocateStorage(0, 0);
  auto* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  const uint16_t kNone = IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE;
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/128,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/0,
      /*.binding_count=*/0,
      /*.binding_slots=*/nullptr,
      /*.constant_span_count=*/0,
      /*.constant_spans=*/nullptr,
      /*.implicit_args=*/
      {
          /*.block_count_offset=*/{0, 4, 16},  // z not contiguous with x/y
          /*.group_size_offset=*/{kNone, kNone, kNone},
          /*.grid_dims_offset=*/kNone,
          /*.dynamic_lds_size_offset=*/kNone,
      },
  };
  EXPECT_THAT(Status(iree_hal_amdgpu_kernarg_layout_initialize(
                  &params, storage.size(), layout)),
              StatusIs(StatusCode::kInvalidArgument));
}

// The device patch stores block_count as uint32 dwords, so block_count_x must
// be 4-byte aligned. A kernel declaring contiguous, in-bounds block_count
// fields anchored at a non-dword-aligned offset must be rejected at layout
// construction (this is the alignment guarantee the removed 8-byte base-align
// reject used to provide, now scoped to where the device actually stores).
TEST(KernargLayoutTest, RejectsMisalignedBlockCount) {
  std::vector<uint8_t> storage = AllocateStorage(0, 0);
  auto* layout =
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(storage.data());
  const uint16_t kNone = IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE;
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/128,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/0,
      /*.implicit_args_byte_offset=*/2,
      /*.binding_count=*/0,
      /*.binding_slots=*/nullptr,
      /*.constant_span_count=*/0,
      /*.constant_spans=*/nullptr,
      /*.implicit_args=*/
      {
          // Contiguous and in bounds, but x is not 4-byte aligned.
          /*.block_count_offset=*/{2, 6, 10},
          /*.group_size_offset=*/{kNone, kNone, kNone},
          /*.grid_dims_offset=*/kNone,
          /*.dynamic_lds_size_offset=*/kNone,
      },
  };
  EXPECT_THAT(Status(iree_hal_amdgpu_kernarg_layout_initialize(
                  &params, storage.size(), layout)),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace iree::hal::amdgpu
