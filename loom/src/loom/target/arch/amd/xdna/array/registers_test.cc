// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/registers.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

loom_xdna_register_field_id_t ResolveField(iree_string_view_t key) {
  loom_xdna_register_field_id_t field_id = 0;
  IREE_EXPECT_OK(loom_xdna_register_field_lookup(key, &field_id));
  return field_id;
}

TEST(XdnaRegisterFactsTest, ExposesCrossVerifiedSemanticCorpus) {
  EXPECT_EQ(loom_xdna_register_field_count(), 173u);
  const loom_xdna_register_field_id_t field_id =
      ResolveField(IREE_SV("compute_memory.dma.bd.word5.lock_acquire_value"));
  loom_xdna_register_field_info_t info = {};
  IREE_ASSERT_OK(loom_xdna_register_field_info(field_id, &info));
  EXPECT_TRUE(iree_string_view_equal(
      info.key, IREE_SV("compute_memory.dma.bd.word5.lock_acquire_value")));
  EXPECT_EQ(info.module, LOOM_XDNA_REGISTER_MODULE_COMPUTE_MEMORY);
  EXPECT_EQ(info.least_significant_bit, 5u);
  EXPECT_EQ(info.bit_width, 7u);
  EXPECT_TRUE(info.is_signed);
  EXPECT_EQ(info.dimension_count, 1u);
  EXPECT_EQ(info.provenance_bits, LOOM_XDNA_PROVENANCE_AIE_RT |
                                      LOOM_XDNA_PROVENANCE_REGISTER_DATABASE);

  loom_xdna_register_dimension_info_t dimension = {};
  IREE_ASSERT_OK(loom_xdna_register_field_dimension(field_id, 0, &dimension));
  EXPECT_TRUE(
      iree_string_view_equal(dimension.name, IREE_SV("buffer_descriptor")));
  EXPECT_EQ(dimension.count, 16u);
  EXPECT_EQ(dimension.stride, 0x20u);
}

TEST(XdnaRegisterFactsTest, FormsIndexedAbsoluteAddresses) {
  const loom_xdna_register_field_id_t field_id =
      ResolveField(IREE_SV("shim_noc.dma.bd.word7.valid_bd"));
  const uint16_t indices[] = {3};
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(), field_id, {2, 0}, IREE_ARRAYSIZE(indices),
      indices, &address));
  EXPECT_EQ(address, (UINT64_C(2) << 25) | 0x1D07C);

  const uint16_t invalid_indices[] = {16};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_address(loom_xdna_npu2_array_family(), field_id,
                                       {2, 0}, IREE_ARRAYSIZE(invalid_indices),
                                       invalid_indices, &address));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_xdna_register_field_address(
                            loom_xdna_npu2_array_family(), field_id, {2, 2},
                            IREE_ARRAYSIZE(indices), indices, &address));
}

TEST(XdnaRegisterFactsTest, EncodesExactSignedAndUnsignedDomains) {
  const loom_xdna_register_field_id_t signed_field =
      ResolveField(IREE_SV("compute_memory.dma.bd.word5.lock_acquire_value"));
  uint32_t bits = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_encode(signed_field, -1, &bits));
  EXPECT_EQ(bits, UINT32_C(0x00000FE0));
  IREE_ASSERT_OK(loom_xdna_register_field_encode(signed_field, -64, &bits));
  EXPECT_EQ(bits, UINT32_C(0x00000800));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_encode(signed_field, -65, &bits));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_encode(signed_field, 64, &bits));

  const loom_xdna_register_field_id_t unsigned_field =
      ResolveField(IREE_SV("core.control.enable"));
  IREE_ASSERT_OK(loom_xdna_register_field_encode(unsigned_field, 1, &bits));
  EXPECT_EQ(bits, 1u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_encode(unsigned_field, 2, &bits));
}

TEST(XdnaRegisterFactsTest, ResolvesTwoDimensionalStreamSlotPattern) {
  const loom_xdna_register_field_id_t field_id =
      ResolveField(IREE_SV("memory_tile.stream.slave_slot.packet_id"));
  const uint16_t indices[] = {13, 2};
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(), field_id, {0, 1}, IREE_ARRAYSIZE(indices),
      indices, &address));
  EXPECT_EQ(address, (UINT64_C(1) << 20) | 0xB02D8);
}

}  // namespace
