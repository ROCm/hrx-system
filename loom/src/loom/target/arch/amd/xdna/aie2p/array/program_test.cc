// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/program.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/function_requirements.h"

namespace loom {
namespace {

static const loom_aie2p_program_register_write32_t* FindRegisterWrite(
    const loom_aie2p_array_program_t& program, uint32_t address) {
  for (iree_host_size_t i = 0; i < program.array_record_count; ++i) {
    const loom_aie2p_program_record_t& record = program.array_records[i];
    if (record.type == LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32 &&
        record.value.register_write32.address == address) {
      return &record.value.register_write32;
    }
  }
  return nullptr;
}

static void ExpectRegisterMaskWrite(const loom_aie2p_program_record_t& record,
                                    uint32_t address, uint32_t mask,
                                    uint32_t value) {
  ASSERT_EQ(record.type, LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32);
  EXPECT_EQ(record.value.register_mask_write32.address, address);
  EXPECT_EQ(record.value.register_mask_write32.mask, mask);
  EXPECT_EQ(record.value.register_mask_write32.value, value);
}

TEST(Aie2pArrayProgramTest, ResetsComputeDmaBeforePlannedQueueStarts) {
  const loom_low_function_requirements_t requirements = {};
  const loom_aie2p_array_worker_plan_t worker_plan = {
      /*.worker_index=*/0,
      /*.coordinate=*/{0, 2},
      /*.requirements=*/&requirements,
  };
  const loom_aie2p_array_channel_t channel = {
      /*.value_id=*/0,
      /*.sender_endpoint_index=*/0,
      /*.receiver_endpoint_index=*/0,
      /*.source_channel_index=*/0,
      /*.first_channel_slot=*/0,
      /*.sender_dma_index=*/UINT32_MAX,
      /*.capacity=*/1,
      /*.record_count=*/1,
      /*.record_byte_length=*/64,
      /*.transport=*/LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA,
  };
  const loom_aie2p_array_channel_slot_t channel_slot = {
      /*.channel_index=*/0,
      /*.slot=*/0,
      /*.byte_length=*/64,
      /*.sender_storage=*/{},
      /*.receiver_storage=*/
      {
          /*.owner=*/{0, 2},
          /*.owner_offset=*/0,
          /*.load_address=*/0x00070000,
      },
  };
  const loom_aie2p_array_lock_plan_t locks[] = {
      {
          /*.channel_index=*/0,
          /*.coordinate=*/{0, 2},
          /*.lock_id=*/0,
          /*.initial_value=*/1,
          /*.ring_endpoint_direction=*/
          LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
          /*.consumer_ready=*/0,
      },
      {
          /*.channel_index=*/0,
          /*.coordinate=*/{0, 2},
          /*.lock_id=*/1,
          /*.initial_value=*/0,
          /*.ring_endpoint_direction=*/
          LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
          /*.consumer_ready=*/1,
      },
  };
  const loom_aie2p_array_dma_plan_t dma = {
      /*.channel_index=*/0,
      /*.coordinate=*/{0, 2},
      /*.direction=*/LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.flags=*/0,
      /*.buffer_descriptor_start=*/0,
      /*.buffer_descriptor_count=*/1,
      /*.credit_lock_index=*/0,
  };
  loom_aie2p_array_plan_t plan = {};
  plan.family = loom_xdna_npu2_array_family();
  plan.channels = &channel;
  plan.channel_count = 1;
  plan.worker_plans = &worker_plan;
  plan.worker_plan_count = 1;
  plan.channel_slots = &channel_slot;
  plan.channel_slot_count = 1;
  plan.locks = locks;
  plan.lock_count = IREE_ARRAYSIZE(locks);
  plan.dma_channels = &dma;
  plan.dma_channel_count = 1;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_array_program_t program = {};
  IREE_ASSERT_OK(loom_aie2p_array_program_build(&plan, &arena, &program));

  ASSERT_EQ(program.array_record_count, 16u);
  ExpectRegisterMaskWrite(program.array_records[0], 0x00232000, 0x00000003,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[1], 0x0021DE00, 0x00000002,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[2], 0x0021DE08, 0x00000002,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[3], 0x0021DE10, 0x00000002,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[4], 0x0021DE18, 0x00000002,
                          0x00000002);
  ASSERT_EQ(program.array_records[7].type,
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  EXPECT_EQ(program.array_records[7].value.register_block_write32.address,
            0x0021D000u);
  ASSERT_EQ(program.array_records[8].type,
            LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD);
  EXPECT_EQ(program.array_records[8].value.tile_program_load.tile_program_index,
            0u);
  ExpectRegisterMaskWrite(program.array_records[9], 0x0021DE00, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[10], 0x0021DE08, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[11], 0x0021DE10, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[12], 0x0021DE18, 0x00000002,
                          0x00000000);
  ASSERT_EQ(program.array_records[13].type,
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32);
  EXPECT_EQ(program.array_records[13].value.register_write32.address,
            0x0021DE04u);
  EXPECT_EQ(program.array_records[13].value.register_write32.value, 0u);
  ExpectRegisterMaskWrite(program.array_records[14], 0x00232000, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[15], 0x00232000, 0x00000001,
                          0x00000001);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pArrayProgramTest, KeepsDmaServiceCoreReset) {
  const loom_low_function_requirements_t requirements = {};
  const loom_aie2p_array_worker_plan_t worker_plan = {
      /*.worker_index=*/0,
      /*.coordinate=*/{0, 2},
      /*.requirements=*/&requirements,
  };
  const loom_aie2p_array_channel_t channel = {
      /*.value_id=*/0,
      /*.sender_endpoint_index=*/0,
      /*.receiver_endpoint_index=*/0,
      /*.source_channel_index=*/0,
      /*.first_channel_slot=*/0,
      /*.sender_dma_index=*/UINT32_MAX,
      /*.capacity=*/1,
      /*.record_count=*/1,
      /*.record_byte_length=*/64,
      /*.transport=*/LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA,
  };
  const loom_aie2p_array_channel_slot_t channel_slot = {
      /*.channel_index=*/0,
      /*.slot=*/0,
      /*.byte_length=*/64,
      /*.sender_storage=*/{},
      /*.receiver_storage=*/
      {
          /*.owner=*/{0, 3},
          /*.owner_offset=*/0,
          /*.load_address=*/0x00060000,
      },
  };
  const loom_aie2p_array_lock_plan_t locks[] = {
      {
          /*.channel_index=*/0,
          /*.coordinate=*/{0, 3},
          /*.lock_id=*/0,
          /*.initial_value=*/1,
          /*.ring_endpoint_direction=*/
          LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
          /*.consumer_ready=*/0,
      },
      {
          /*.channel_index=*/0,
          /*.coordinate=*/{0, 3},
          /*.lock_id=*/1,
          /*.initial_value=*/0,
          /*.ring_endpoint_direction=*/
          LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
          /*.consumer_ready=*/1,
      },
  };
  const loom_aie2p_array_dma_plan_t dma = {
      /*.channel_index=*/0,
      /*.coordinate=*/{0, 3},
      /*.direction=*/LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.flags=*/LOOM_AIE2P_ARRAY_DMA_FLAG_SERVICE_TILE_LIFECYCLE,
      /*.buffer_descriptor_start=*/0,
      /*.buffer_descriptor_count=*/1,
      /*.credit_lock_index=*/0,
  };
  loom_aie2p_array_plan_t plan = {};
  plan.family = loom_xdna_npu2_array_family();
  plan.channels = &channel;
  plan.channel_count = 1;
  plan.worker_plans = &worker_plan;
  plan.worker_plan_count = 1;
  plan.channel_slots = &channel_slot;
  plan.channel_slot_count = 1;
  plan.locks = locks;
  plan.lock_count = IREE_ARRAYSIZE(locks);
  plan.dma_channels = &dma;
  plan.dma_channel_count = 1;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_array_program_t program = {};
  IREE_ASSERT_OK(loom_aie2p_array_program_build(&plan, &arena, &program));

  ASSERT_EQ(program.array_record_count, 25u);
  ExpectRegisterMaskWrite(program.array_records[0], 0x00232000, 0x00000003,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[1], 0x00332000, 0x00000003,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[6], 0x0031DE00, 0x00000002,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[7], 0x0031DE08, 0x00000002,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[8], 0x0031DE10, 0x00000002,
                          0x00000002);
  ExpectRegisterMaskWrite(program.array_records[9], 0x0031DE18, 0x00000002,
                          0x00000002);
  ASSERT_EQ(program.array_records[12].type,
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  EXPECT_EQ(program.array_records[12].value.register_block_write32.address,
            0x0031D000u);
  ASSERT_EQ(program.array_records[13].type,
            LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD);
  EXPECT_EQ(
      program.array_records[13].value.tile_program_load.tile_program_index, 0u);
  ExpectRegisterMaskWrite(program.array_records[18], 0x0031DE00, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[19], 0x0031DE08, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[20], 0x0031DE10, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[21], 0x0031DE18, 0x00000002,
                          0x00000000);
  ASSERT_EQ(program.array_records[22].type,
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32);
  EXPECT_EQ(program.array_records[22].value.register_write32.address,
            0x0031DE04u);
  EXPECT_EQ(program.array_records[22].value.register_write32.value, 0u);
  ExpectRegisterMaskWrite(program.array_records[23], 0x00232000, 0x00000002,
                          0x00000000);
  ExpectRegisterMaskWrite(program.array_records[24], 0x00232000, 0x00000001,
                          0x00000001);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pArrayProgramTest, RoutesShimCompletionTokensToFirmware) {
  const loom_aie2p_array_binding_t binding = {
      /*.value_id=*/0,
      /*.ordinal=*/0,
      /*.access=*/LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE,
  };
  const loom_aie2p_array_binding_plan_t binding_plan = {
      /*.binding_index=*/0,
      /*.channel_index=*/0,
      /*.dma_index=*/0,
      /*.partition_lane=*/0,
      /*.partition_lane_count=*/1,
      /*.binding_byte_offset=*/0,
      /*.binding_span_byte_length=*/64,
      /*.transfer_byte_length=*/64,
      /*.task_repeat_count=*/1,
  };
  const loom_aie2p_array_dma_plan_t dma = {
      /*.channel_index=*/0,
      /*.coordinate=*/{0, 0},
      /*.direction=*/LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.flags=*/LOOM_AIE2P_ARRAY_DMA_FLAG_SHIM,
      /*.buffer_descriptor_start=*/0,
      /*.buffer_descriptor_count=*/1,
      /*.credit_lock_index=*/UINT32_MAX,
  };
  loom_aie2p_array_plan_t plan = {};
  plan.family = loom_xdna_npu2_array_family();
  plan.bindings = &binding;
  plan.binding_count = 1;
  plan.dma_channels = &dma;
  plan.dma_channel_count = 1;
  plan.binding_plans = &binding_plan;
  plan.binding_plan_count = 1;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_array_program_t program = {};
  IREE_ASSERT_OK(loom_aie2p_array_program_build(&plan, &arena, &program));

  ASSERT_EQ(program.array_record_count, 3u);
  const loom_aie2p_program_register_write32_t* master =
      FindRegisterWrite(program, 0x0003F008);
  const loom_aie2p_program_register_write32_t* slave =
      FindRegisterWrite(program, 0x0003F100);
  const loom_aie2p_program_register_write32_t* slot =
      FindRegisterWrite(program, 0x0003F200);
  ASSERT_NE(master, nullptr);
  ASSERT_NE(slave, nullptr);
  ASSERT_NE(slot, nullptr);
  EXPECT_EQ(master->value, 0xC0000045u);
  EXPECT_EQ(slave->value, 0xC0000000u);
  EXPECT_EQ(slot->value, 0x0F1F0135u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
