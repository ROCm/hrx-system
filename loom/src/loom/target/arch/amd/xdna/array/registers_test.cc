// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/registers.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(XdnaRegisterFactsTest, ExposesCrossVerifiedSemanticCorpus) {
  EXPECT_EQ(loom_xdna_register_field_count(), 410u);
  const loom_xdna_register_field_id_t field_id =
      LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_VALUE;
  loom_xdna_register_field_info_t info = {};
  IREE_ASSERT_OK(loom_xdna_register_field_info(field_id, &info));
  EXPECT_TRUE(iree_string_view_equal(
      info.key, IREE_SV("compute_memory.dma.bd.word5.lock_acquire_value")));
  EXPECT_EQ(info.module, LOOM_XDNA_REGISTER_MODULE_COMPUTE_MEMORY);
  EXPECT_EQ(info.access, LOOM_XDNA_REGISTER_ACCESS_READ_WRITE);
  EXPECT_EQ(info.least_significant_bit, 5u);
  EXPECT_EQ(info.bit_width, 7u);
  EXPECT_TRUE(info.is_signed);
  EXPECT_EQ(info.dimension_count, 1u);
  EXPECT_EQ(info.provenance_bits, LOOM_XDNA_PROVENANCE_AIE_RT |
                                      LOOM_XDNA_PROVENANCE_REGISTER_DATABASE);

  const loom_xdna_register_dimension_info_t dimension =
      loom_xdna_register_field_dimension(field_id, 0);
  EXPECT_TRUE(
      iree_string_view_equal(dimension.name, IREE_SV("buffer_descriptor")));
  EXPECT_EQ(dimension.count, 16u);
  EXPECT_EQ(dimension.stride, 0x20u);
}

TEST(XdnaRegisterFactsTest, FormsIndexedAbsoluteAddresses) {
  const loom_xdna_register_field_id_t field_id =
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_VALID_BD;
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
      LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_VALUE;
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
      LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_ENABLE;
  IREE_ASSERT_OK(loom_xdna_register_field_encode(unsigned_field, 1, &bits));
  EXPECT_EQ(bits, 1u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_encode(unsigned_field, 2, &bits));

  EXPECT_EQ(loom_xdna_register_field_encode_admitted(signed_field, -1),
            UINT32_C(0x00000FE0));
  EXPECT_EQ(loom_xdna_register_field_encode_admitted(unsigned_field, 1), 1u);
}

TEST(XdnaRegisterFactsTest, ProjectsAdmittedIndexedAddress) {
  const uint16_t indices[] = {3};
  EXPECT_EQ(loom_xdna_register_field_address_admitted(
                loom_xdna_npu2_array_family(),
                LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_VALID_BD, {2, 0},
                indices),
            (UINT64_C(2) << 25) | 0x1D07C);
}

TEST(XdnaRegisterFactsTest, ResolvesTwoDimensionalStreamSlotPattern) {
  const loom_xdna_register_field_id_t field_id =
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_SLAVE_SLOT_PACKET_ID;
  const uint16_t indices[] = {13, 2};
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(), field_id, {0, 1}, IREE_ARRAYSIZE(indices),
      indices, &address));
  EXPECT_EQ(address, (UINT64_C(1) << 20) | 0xB02D8);
}

TEST(XdnaRegisterFactsTest, DescribesCompleteStreamRegisterBanks) {
  const struct {
    // Representative tile of each switch kind at a nonzero column.
    loom_xdna_tile_coordinate_t coordinate;
    // Generated fields selecting master, slave and filter register banks.
    loom_xdna_register_field_id_t fields[3];
    // Port count of each bank's first indexed dimension.
    uint16_t port_counts[3];
    // Absolute address of the last valid word in each bank.
    uint32_t last_addresses[3];
  } cases[] = {
      {{2, 0},
       {LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_STREAM_MASTER_CONFIG_ENABLE,
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_STREAM_SLAVE_CONFIG_ENABLE,
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_STREAM_SLAVE_SLOT_ENABLE},
       {22, 23, 23},
       {0x0403F054, 0x0403F158, 0x0403F36C}},
      {{2, 1},
       {LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_MASTER_CONFIG_ENABLE,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_SLAVE_CONFIG_ENABLE,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_SLAVE_SLOT_ENABLE},
       {17, 18, 18},
       {0x041B0040, 0x041B0144, 0x041B031C}},
      {{2, 2},
       {LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_MASTER_CONFIG_ENABLE,
        LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_CONFIG_ENABLE,
        LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_ENABLE},
       {23, 25, 25},
       {0x0423F058, 0x0423F160, 0x0423F38C}},
  };
  for (const auto& test_case : cases) {
    const auto slots =
        loom_xdna_register_field_dimension(test_case.fields[2], 1);
    EXPECT_EQ(slots.count, 4u);
    EXPECT_EQ(slots.stride, 4u);
    for (size_t bank = 0; bank < IREE_ARRAYSIZE(test_case.fields); ++bank) {
      SCOPED_TRACE(test_case.fields[bank]);
      const auto ports =
          loom_xdna_register_field_dimension(test_case.fields[bank], 0);
      EXPECT_EQ(ports.count, test_case.port_counts[bank]);
      EXPECT_EQ(ports.stride, bank == 2 ? 16u : 4u);
      const uint16_t indices[] = {static_cast<uint16_t>(ports.count - 1),
                                  static_cast<uint16_t>(slots.count - 1)};
      EXPECT_EQ(loom_xdna_register_field_address_admitted(
                    loom_xdna_npu2_array_family(), test_case.fields[bank],
                    test_case.coordinate, indices),
                test_case.last_addresses[bank]);
    }
  }
}

TEST(XdnaRegisterFactsTest, EncodesCoreTraceFieldsAtExactHardwareAddresses) {
  // XAIE2PGBL_CORE_MODULE_TRACE_* and AM025 define these complete field masks.
  const struct {
    // Field selected through the generated identifier corpus.
    loom_xdna_register_field_id_t field;
    // Largest raw value representable by this field.
    int64_t maximum;
    // Expected positioned bits at the largest value.
    uint32_t mask;
    // Tile-relative register address from the hardware oracles.
    uint32_t offset;
  } cases[] = {
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_CONTROL0_STOP_EVENT, 127, 0x7F000000,
       0x340D0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_CONTROL0_START_EVENT, 127,
       0x007F0000, 0x340D0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_CONTROL0_MODE, 3, 0x00000003,
       0x340D0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_CONTROL1_PACKET_TYPE, 7, 0x00007000,
       0x340D4},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_CONTROL1_PACKET_ID, 31, 0x0000001F,
       0x340D4},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT0_EVENT0, 127, 0x0000007F,
       0x340E0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT0_EVENT1, 127, 0x00007F00,
       0x340E0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT0_EVENT2, 127, 0x007F0000,
       0x340E0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT0_EVENT3, 127, 0x7F000000,
       0x340E0},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT1_EVENT4, 127, 0x0000007F,
       0x340E4},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT1_EVENT5, 127, 0x00007F00,
       0x340E4},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT1_EVENT6, 127, 0x007F0000,
       0x340E4},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT1_EVENT7, 127, 0x7F000000,
       0x340E4},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.field);
    loom_xdna_register_field_info_t info = {};
    IREE_ASSERT_OK(loom_xdna_register_field_info(test_case.field, &info));
    EXPECT_EQ(info.module, LOOM_XDNA_REGISTER_MODULE_CORE);
    EXPECT_EQ(info.access, LOOM_XDNA_REGISTER_ACCESS_READ_WRITE);
    EXPECT_FALSE(info.is_signed);
    EXPECT_EQ(info.dimension_count, 0u);
    uint64_t address = 0;
    IREE_ASSERT_OK(loom_xdna_register_field_address(
        loom_xdna_npu2_array_family(), test_case.field, {7, 5}, 0, nullptr,
        &address));
    EXPECT_EQ(address, UINT64_C(0x0E500000) | test_case.offset);
    uint32_t bits = 0;
    IREE_ASSERT_OK(loom_xdna_register_field_encode(test_case.field, 0, &bits));
    EXPECT_EQ(bits, 0u);
    IREE_ASSERT_OK(loom_xdna_register_field_encode(test_case.field,
                                                   test_case.maximum, &bits));
    EXPECT_EQ(bits, test_case.mask);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_xdna_register_field_encode(test_case.field, -1, &bits));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          loom_xdna_register_field_encode(
                              test_case.field, test_case.maximum + 1, &bits));
  }
}

TEST(XdnaRegisterFactsTest, ResolvesBroadcastTimerSynchronizationSequence) {
  // The pinned mlir-aie AIEInsertTraceFlows sequence resets timers on the
  // start broadcast, then generates shim USER_EVENT_1. aie-rt event IDs are
  // 127 for shim USER_EVENT_1, 122 for core/memory BROADCAST_15, and 157 for
  // memory-tile BROADCAST_15. Shim USER_EVENT_0 (126) supplies the stop event.
  const struct {
    // Field programmed by the upstream synchronization sequence.
    loom_xdna_register_field_id_t field;
    // Concrete source or destination tile.
    loom_xdna_tile_coordinate_t coordinate;
    // Whether the field selects an indexed broadcast channel.
    iree_host_size_t index_count;
    // Broadcast channel when index_count is one.
    uint16_t index;
    // Raw event identifier in this module's event domain.
    int64_t value;
    // Expected absolute register address.
    uint64_t address;
    // Expected positioned event bits.
    uint32_t bits;
  } cases[] = {
      {LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_CONTROL_RESET_EVENT,
       {2, 2},
       0,
       0,
       122,
       0x04234000,
       0x7A00},
      {LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_TIMER_CONTROL_RESET_EVENT,
       {2, 2},
       0,
       0,
       122,
       0x04214000,
       0x7A00},
      {LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_TIMER_CONTROL_RESET_EVENT,
       {2, 1},
       0,
       0,
       157,
       0x04194000,
       0x9D00},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_TIMER_CONTROL_RESET_EVENT,
       {2, 0},
       0,
       0,
       127,
       0x04034000,
       0x7F00},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_BROADCAST_EVENT,
       {2, 0},
       1,
       15,
       127,
       0x0403404C,
       0x7F},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_GENERATE_EVENT,
       {2, 0},
       0,
       0,
       127,
       0x04034008,
       0x7F},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_BROADCAST_EVENT,
       {2, 0},
       1,
       14,
       126,
       0x04034048,
       0x7E},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_GENERATE_EVENT,
       {2, 0},
       0,
       0,
       126,
       0x04034008,
       0x7E},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.field);
    uint64_t address = 0;
    IREE_ASSERT_OK(loom_xdna_register_field_address(
        loom_xdna_npu2_array_family(), test_case.field, test_case.coordinate,
        test_case.index_count,
        test_case.index_count ? &test_case.index : nullptr, &address));
    EXPECT_EQ(address, test_case.address);
    uint32_t bits = 0;
    IREE_ASSERT_OK(loom_xdna_register_field_encode(test_case.field,
                                                   test_case.value, &bits));
    EXPECT_EQ(bits, test_case.bits);
  }
}

TEST(XdnaRegisterFactsTest, PreservesMixedAccessAndStatusComparisonBits) {
  const struct {
    // Field whose access and raw bit domain are being queried.
    loom_xdna_register_field_id_t field;
    // Software-visible access from the register database.
    loom_xdna_register_access_t access;
    // Raw field value, including full-width timer words.
    int64_t value;
    // Expected register bits, also usable for status comparisons.
    uint32_t bits;
  } cases[] = {
      {LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_CONTROL_RESET,
       LOOM_XDNA_REGISTER_ACCESS_WRITE_ONLY, 1, 0x80000000},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_CONTROL_RESET_EVENT,
       LOOM_XDNA_REGISTER_ACCESS_READ_WRITE, 122, 0x7A00},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_STATUS_STATE,
       LOOM_XDNA_REGISTER_ACCESS_READ_ONLY, 3, 0x300},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_STATUS_MODE,
       LOOM_XDNA_REGISTER_ACCESS_READ_ONLY, 2, 0x2},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_LOW_VALUE,
       LOOM_XDNA_REGISTER_ACCESS_READ_ONLY, UINT32_MAX, UINT32_MAX},
      {LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_HIGH_VALUE,
       LOOM_XDNA_REGISTER_ACCESS_READ_ONLY, 0x12345678, 0x12345678},
      {LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_TIMER_TRIGGER_LOW_VALUE,
       LOOM_XDNA_REGISTER_ACCESS_READ_WRITE, UINT32_MAX, UINT32_MAX},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_TIMER_TRIGGER_HIGH_VALUE,
       LOOM_XDNA_REGISTER_ACCESS_READ_WRITE, 0x12345678, 0x12345678},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_GENERATE_EVENT,
       LOOM_XDNA_REGISTER_ACCESS_WRITE_ONLY, 127, 127},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID,
       LOOM_XDNA_REGISTER_ACCESS_WRITE_ONLY, 15, 15},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.field);
    loom_xdna_register_field_info_t info = {};
    IREE_ASSERT_OK(loom_xdna_register_field_info(test_case.field, &info));
    EXPECT_EQ(info.access, test_case.access);
    EXPECT_FALSE(info.is_signed);
    uint32_t bits = 0;
    IREE_ASSERT_OK(loom_xdna_register_field_encode(test_case.field,
                                                   test_case.value, &bits));
    EXPECT_EQ(bits, test_case.bits);
  }
  uint32_t bits = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_xdna_register_field_encode(
                            LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_LOW_VALUE,
                            INT64_C(0x100000000), &bits));
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(),
      LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_STATUS_STATE, {2, 2}, 0, nullptr,
      &address));
  EXPECT_EQ(address, UINT64_C(0x042340D8));
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(),
      LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_HIGH_VALUE, {2, 2}, 0, nullptr,
      &address));
  EXPECT_EQ(address, UINT64_C(0x042340FC));
}

TEST(XdnaRegisterFactsTest, ResolvesBroadcastSwitchesAndRejectsInvalidDomains) {
  const uint16_t switch_b[] = {1};
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(),
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_EVENT_BROADCAST_EAST_VALUE_CHANNELS,
      {2, 1}, 1, switch_b, &address));
  EXPECT_EQ(address, UINT64_C(0x041940C8));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_address(
          loom_xdna_npu2_array_family(),
          LOOM_XDNA_REGISTER_FIELD_CORE_EVENT_BROADCAST_EAST_VALUE_CHANNELS,
          {2, 2}, 1, switch_b, &address));
  const uint16_t invalid_channel[] = {16};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_address(
          loom_xdna_npu2_array_family(),
          LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_BROADCAST_EVENT, {2, 0}, 1,
          invalid_channel, &address));
  for (const auto coordinate :
       {loom_xdna_tile_coordinate_t{2, 0}, loom_xdna_tile_coordinate_t{2, 1}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_xdna_register_field_address(
                              loom_xdna_npu2_array_family(),
                              LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_STATUS_STATE,
                              coordinate, 0, nullptr, &address));
  }
}

TEST(XdnaRegisterFactsTest, EncodesMemoryTileEventsAndStreamObservation) {
  uint32_t bits = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_TRACE_EVENT1_EVENT7, 160, &bits));
  EXPECT_EQ(bits, UINT32_C(0xA0000000));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_EVENT1_EVENT7, 160, &bits));
  IREE_ASSERT_OK(loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_COMBO_EVENT_INPUTS_EVENT_D, 160,
      &bits));
  EXPECT_EQ(bits, UINT32_C(0xA0000000));
  IREE_ASSERT_OK(loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_CORE_COMBO_EVENT_CONTROL_COMBO2, 2, &bits));
  EXPECT_EQ(bits, UINT32_C(0x00020000));
  IREE_ASSERT_OK(loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_EDGE_EVENT_CONTROL_EVENT1, 160,
      &bits));
  EXPECT_EQ(bits, UINT32_C(0x00A00000));
  IREE_ASSERT_OK(loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_EDGE_EVENT_CONTROL_RISING1, 1,
      &bits));
  EXPECT_EQ(bits, UINT32_C(0x02000000));

  // Core trace is slave port 23, represented by port slot 7 in this example.
  IREE_ASSERT_OK(loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_EVENT_PORT_SELECTION1_PORT7_ID, 23,
      &bits));
  EXPECT_EQ(bits, UINT32_C(0x17000000));
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(),
      LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_EVENT_PORT_SELECTION1_PORT7_ID,
      {2, 2}, 0, nullptr, &address));
  EXPECT_EQ(address, UINT64_C(0x0423FF04));
  const uint16_t trace_slot[] = {23, 0};
  IREE_ASSERT_OK(loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(),
      LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_PACKET_ID, {2, 2},
      IREE_ARRAYSIZE(trace_slot), trace_slot, &address));
  EXPECT_EQ(address, UINT64_C(0x0423F370));
}

}  // namespace
