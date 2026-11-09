// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/registers.h"

#include "iree/testing/gtest.h"

namespace {

TEST(XdnaRegisterFactsTest, AddressesAtomicSramCreditRequests) {
  // AM025 and the NPU2 driver encode lock, operation, and signed delta in
  // address bits 15:10, 9, and 8:2. The read result is the success bit.
  const loom_xdna_register_field_id_t fields[] = {
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_LOCK_RELEASE_SUCCESS,
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_LOCK_ACQUIRE_SUCCESS,
  };
  for (uint16_t operation = 0; operation < IREE_ARRAYSIZE(fields);
       ++operation) {
    const auto info = loom_xdna_register_field_info(fields[operation]);
    EXPECT_EQ(info.access, LOOM_XDNA_REGISTER_ACCESS_READ_ONLY);
    EXPECT_EQ(info.least_significant_bit, 0u);
    EXPECT_EQ(info.bit_width, 1u);
    for (uint16_t lock = 0; lock < 64; ++lock) {
      for (uint16_t delta = 0; delta < 128; ++delta) {
        const uint16_t indices[] = {lock, delta};
        EXPECT_EQ(loom_xdna_register_field_address(
                      loom_xdna_npu2_array_family(), fields[operation], {7, 1},
                      indices),
                  UINT64_C(0xE1D0000) | (lock << 10) | (operation << 9) |
                      (delta << 2));
      }
    }
  }
}

TEST(XdnaRegisterFactsTest, ExposesCrossVerifiedSemanticCorpus) {
  const loom_xdna_register_field_id_t field_id =
      LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_VALUE;
  const loom_xdna_register_field_info_t info =
      loom_xdna_register_field_info(field_id);
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

TEST(XdnaRegisterFactsTest, ResolvesDmaIdleStatusAcrossMemoryModules) {
  const struct {
    // Queue-size, running, and four stalled-state register fields.
    loom_xdna_register_field_id_t fields[6];
    // Last physical tile exposing this DMA module.
    loom_xdna_tile_coordinate_t coordinate;
    // Number of independently addressable DMA channels.
    uint16_t channel_count;
    // Absolute channel-zero status address from the hardware oracles.
    uint64_t address;
  } cases[] = {
      {{
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_TASK_QUEUE_SIZE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_CHANNEL_RUNNING,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_ACQUIRE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_RELEASE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_STREAM_STARVATION,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_TOKEN_OR_COUNT_FIFO,
       },
       {7, 5},
       2,
       0xE51DF00},
      {{
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_TASK_QUEUE_SIZE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_CHANNEL_RUNNING,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_ACQUIRE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_RELEASE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_STREAM_BACKPRESSURE,
           LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_COMPLETION_TOKEN,
       },
       {7, 5},
       2,
       0xE51DF10},
      {{
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_TASK_QUEUE_SIZE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_CHANNEL_RUNNING,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_ACQUIRE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_RELEASE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_STREAM_STARVATION,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_TOKEN_OR_COUNT_FIFO,
       },
       {7, 1},
       6,
       0xE1A0660},
      {{
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_TASK_QUEUE_SIZE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_CHANNEL_RUNNING,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_ACQUIRE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_RELEASE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_STREAM_BACKPRESSURE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_COMPLETION_TOKEN,
       },
       {7, 1},
       6,
       0xE1A0680},
      {{
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_TASK_QUEUE_SIZE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_CHANNEL_RUNNING,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_ACQUIRE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_RELEASE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_STREAM_STARVATION,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_TOKEN_OR_COUNT_FIFO,
       },
       {7, 0},
       2,
       0xE01D220},
      {{
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_TASK_QUEUE_SIZE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_CHANNEL_RUNNING,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_ACQUIRE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_RELEASE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_STREAM_BACKPRESSURE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_COMPLETION_TOKEN,
       },
       {7, 0},
       2,
       0xE01D228},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.address);
    uint32_t idle_mask = 0;
    for (size_t i = 0; i < IREE_ARRAYSIZE(test_case.fields); ++i) {
      const auto field = test_case.fields[i];
      const loom_xdna_register_field_info_t info =
          loom_xdna_register_field_info(field);
      EXPECT_EQ(info.access, LOOM_XDNA_REGISTER_ACCESS_READ_ONLY);
      const auto dimension = loom_xdna_register_field_dimension(field, 0);
      EXPECT_EQ(dimension.count, test_case.channel_count);
      EXPECT_EQ(dimension.stride, 4u);
      for (uint16_t channel = 0; channel < test_case.channel_count; ++channel) {
        EXPECT_EQ(loom_xdna_register_field_address(
                      loom_xdna_npu2_array_family(), field,
                      test_case.coordinate, &channel),
                  test_case.address + channel * 4);
      }
      idle_mask |= loom_xdna_register_field_encode(field, i ? 1 : 7);
    }
    EXPECT_EQ(idle_mask, 0x0078003Cu);
  }
}

TEST(XdnaRegisterFactsTest, FormsIndexedAbsoluteAddresses) {
  const loom_xdna_register_field_id_t field_id =
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_VALID_BD;
  const uint16_t indices[] = {3};
  const uint64_t address = loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(), field_id, {2, 0}, indices);
  EXPECT_EQ(address, (UINT64_C(2) << 25) | 0x1D07C);
}

TEST(XdnaRegisterFactsTest, EncodesSignedAndUnsignedBits) {
  const loom_xdna_register_field_id_t signed_field =
      LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_VALUE;
  EXPECT_EQ(loom_xdna_register_field_encode(signed_field, -1),
            UINT32_C(0x00000FE0));
  EXPECT_EQ(loom_xdna_register_field_encode(signed_field, -64),
            UINT32_C(0x00000800));

  const loom_xdna_register_field_id_t unsigned_field =
      LOOM_XDNA_REGISTER_FIELD_CORE_CONTROL_ENABLE;
  EXPECT_EQ(loom_xdna_register_field_encode(unsigned_field, 1), 1u);
}

TEST(XdnaRegisterFactsTest, ResolvesTwoDimensionalStreamSlotPattern) {
  const loom_xdna_register_field_id_t field_id =
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_SLAVE_SLOT_PACKET_ID;
  const uint16_t indices[] = {13, 2};
  const uint64_t address = loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(), field_id, {0, 1}, indices);
  EXPECT_EQ(address, (UINT64_C(1) << 20) | 0xB02D8);
}

TEST(XdnaRegisterFactsTest, DescribesCompleteStreamRegisterBanks) {
  const struct {
    // Representative tile of each switch kind at a nonzero column.
    loom_xdna_tile_coordinate_t coordinate;
    // Generated fields selecting master, slave, and filter register banks.
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
      EXPECT_EQ(loom_xdna_register_field_address(loom_xdna_npu2_array_family(),
                                                 test_case.fields[bank],
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
    const loom_xdna_register_field_info_t info =
        loom_xdna_register_field_info(test_case.field);
    EXPECT_EQ(info.module, LOOM_XDNA_REGISTER_MODULE_CORE);
    EXPECT_EQ(info.access, LOOM_XDNA_REGISTER_ACCESS_READ_WRITE);
    EXPECT_FALSE(info.is_signed);
    EXPECT_EQ(info.dimension_count, 0u);
    const uint64_t address = loom_xdna_register_field_address(
        loom_xdna_npu2_array_family(), test_case.field, {7, 5}, nullptr);
    EXPECT_EQ(address, UINT64_C(0x0E500000) | test_case.offset);
    EXPECT_EQ(loom_xdna_register_field_encode(test_case.field, 0), 0u);
    EXPECT_EQ(
        loom_xdna_register_field_encode(test_case.field, test_case.maximum),
        test_case.mask);
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
    bool indexed;
    // Broadcast channel when indexed.
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
       false,
       0,
       122,
       0x04234000,
       0x7A00},
      {LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_TIMER_CONTROL_RESET_EVENT,
       {2, 2},
       false,
       0,
       122,
       0x04214000,
       0x7A00},
      {LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_TIMER_CONTROL_RESET_EVENT,
       {2, 1},
       false,
       0,
       157,
       0x04194000,
       0x9D00},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_TIMER_CONTROL_RESET_EVENT,
       {2, 0},
       false,
       0,
       127,
       0x04034000,
       0x7F00},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_BROADCAST_EVENT,
       {2, 0},
       true,
       15,
       127,
       0x0403404C,
       0x7F},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_GENERATE_EVENT,
       {2, 0},
       false,
       0,
       127,
       0x04034008,
       0x7F},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_BROADCAST_EVENT,
       {2, 0},
       true,
       14,
       126,
       0x04034048,
       0x7E},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_PL_EVENT_GENERATE_EVENT,
       {2, 0},
       false,
       0,
       126,
       0x04034008,
       0x7E},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.field);
    const uint64_t address = loom_xdna_register_field_address(
        loom_xdna_npu2_array_family(), test_case.field, test_case.coordinate,
        test_case.indexed ? &test_case.index : nullptr);
    EXPECT_EQ(address, test_case.address);
    EXPECT_EQ(loom_xdna_register_field_encode(test_case.field, test_case.value),
              test_case.bits);
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
    const loom_xdna_register_field_info_t info =
        loom_xdna_register_field_info(test_case.field);
    EXPECT_EQ(info.access, test_case.access);
    EXPECT_FALSE(info.is_signed);
    EXPECT_EQ(loom_xdna_register_field_encode(test_case.field, test_case.value),
              test_case.bits);
  }
  EXPECT_EQ(
      loom_xdna_register_field_address(
          loom_xdna_npu2_array_family(),
          LOOM_XDNA_REGISTER_FIELD_CORE_TRACE_STATUS_STATE, {2, 2}, nullptr),
      UINT64_C(0x042340D8));
  EXPECT_EQ(
      loom_xdna_register_field_address(
          loom_xdna_npu2_array_family(),
          LOOM_XDNA_REGISTER_FIELD_CORE_TIMER_HIGH_VALUE, {2, 2}, nullptr),
      UINT64_C(0x042340FC));
}

TEST(XdnaRegisterFactsTest, ResolvesBroadcastSwitchAddress) {
  const uint16_t switch_b[] = {1};
  const uint64_t address = loom_xdna_register_field_address(
      loom_xdna_npu2_array_family(),
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_EVENT_BROADCAST_EAST_VALUE_CHANNELS,
      {2, 1}, switch_b);
  EXPECT_EQ(address, UINT64_C(0x041940C8));
}

TEST(XdnaRegisterFactsTest, EncodesMemoryTileEventsAndStreamObservation) {
  EXPECT_EQ(loom_xdna_register_field_encode(
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_TRACE_EVENT1_EVENT7, 160),
            UINT32_C(0xA0000000));
  EXPECT_EQ(
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_COMBO_EVENT_INPUTS_EVENT_D, 160),
      UINT32_C(0xA0000000));
  EXPECT_EQ(loom_xdna_register_field_encode(
                LOOM_XDNA_REGISTER_FIELD_CORE_COMBO_EVENT_CONTROL_COMBO2, 2),
            UINT32_C(0x00020000));
  EXPECT_EQ(
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_EDGE_EVENT_CONTROL_EVENT1, 160),
      UINT32_C(0x00A00000));
  EXPECT_EQ(
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_EDGE_EVENT_CONTROL_RISING1, 1),
      UINT32_C(0x02000000));

  // Core trace is slave port 23, represented by port slot 7 in this example.
  EXPECT_EQ(
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_EVENT_PORT_SELECTION1_PORT7_ID,
          23),
      UINT32_C(0x17000000));
  EXPECT_EQ(
      loom_xdna_register_field_address(
          loom_xdna_npu2_array_family(),
          LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_EVENT_PORT_SELECTION1_PORT7_ID,
          {2, 2}, nullptr),
      UINT64_C(0x0423FF04));
  const uint16_t trace_slot[] = {23, 0};
  EXPECT_EQ(loom_xdna_register_field_address(
                loom_xdna_npu2_array_family(),
                LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_PACKET_ID,
                {2, 2}, trace_slot),
            UINT64_C(0x0423F370));
}

}  // namespace
