// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/types.h"

namespace {

TEST(ChannelCursorTest, RetainsEveryPreviousWorkersBalance) {
  for (bool first_exact : {false, true}) {
    for (uint32_t reads : {0u, 1u}) {
      loom_pipeline_invocation_t owners[3] = {};
      loom_pipeline_invocation_worker_t workers[] = {
          {.invocation = &owners[0]},
          {.invocation = &owners[1]},
          {.invocation = &owners[2]},
      };
      loom_pipeline_invocations_t invocations = {
          .workers = workers,
          .worker_count = 3,
      };
      loom_aie2p_native_channel_t channel = {
          .balance = {.worker = UINT32_MAX, .exact = true, .balanced = true},
          .cursor =
              {
                  .reader = {.worker = UINT32_MAX},
                  .writer = {.worker = UINT32_MAX},
              },
      };
      loom_aie2p_native_begin_cursor_owner(&channel, &invocations, 0);
      loom_aie2p_native_cursor_bind(&channel.cursor.writer, 2, 0, 1,
                                    first_exact, false);
      if (reads) {
        loom_aie2p_native_begin_cursor_owner(&channel, &invocations, 0);
        loom_aie2p_native_cursor_bind(&channel.cursor.reader, 2, 0, reads,
                                      first_exact, false);
      }
      // Each later worker is balanced. Replacing the representative must
      // retain any uncertainty or unread tail from the first worker.
      for (uint32_t worker : {1u, 2u}) {
        loom_aie2p_native_begin_cursor_owner(&channel, &invocations, worker);
        loom_aie2p_native_cursor_bind(&channel.cursor.reader, 2, worker, 2,
                                      true, true);
        loom_aie2p_native_begin_cursor_owner(&channel, &invocations, worker);
        loom_aie2p_native_cursor_bind(&channel.cursor.writer, 2, worker, 2,
                                      true, true);
      }
      loom_aie2p_native_finish_cursor_owners(&channel);
      EXPECT_EQ(channel.balance.balanced, first_exact && reads == 1);
    }
  }
}

TEST(ChannelCursorTest, ConstructionBalanceIncludesIndependentWorkers) {
  struct Case {
    // Read count for the first construction, whose writer publishes two.
    uint32_t first_reads;
    // Whether that read count is exact on all paths.
    bool first_exact;
    // Whether the first reader belongs to the same construction as its writer.
    bool same_construction;
    // Final proof after an independently balanced second construction.
    bool balanced;
  };
  const Case cases[] = {{2, true, true, true},
                        {1, true, true, false},
                        {2, false, true, false},
                        {2, true, false, false}};
  for (const Case& test : cases) {
    SCOPED_TRACE(test.first_reads);
    SCOPED_TRACE(test.first_exact);
    SCOPED_TRACE(test.same_construction);
    loom_pipeline_invocation_t owners[3] = {};
    loom_pipeline_invocation_worker_t workers[] = {
        {.invocation = &owners[0]},
        {.invocation = &owners[test.same_construction ? 0 : 1]},
        {.invocation = &owners[2]},
        {.invocation = &owners[2]},
    };
    loom_pipeline_invocations_t invocations = {
        .workers = workers,
        .worker_count = 4,
    };
    loom_aie2p_native_channel_t channel = {
        .balance = {.worker = UINT32_MAX, .exact = true, .balanced = true},
        .cursor =
            {
                .reader = {.worker = UINT32_MAX},
                .writer = {.worker = UINT32_MAX},
            },
    };
    for (uint32_t worker = 0; worker < 4; ++worker) {
      loom_aie2p_native_begin_cursor_owner(&channel, &invocations, worker);
      auto* cursor =
          worker % 2 ? &channel.cursor.reader : &channel.cursor.writer;
      const uint32_t count = worker == 1  ? test.first_reads
                             : worker < 2 ? 2
                                          : 7;
      loom_aie2p_native_cursor_bind(cursor, 2, worker, count,
                                    worker != 1 || test.first_exact, true);
      // Visiting another admission site retains the producer's aggregate.
      // It must not contribute the worker's total a second time.
      loom_aie2p_native_begin_cursor_owner(&channel, &invocations, worker);
      loom_aie2p_native_cursor_bind(cursor, 2, worker, count, true, true);
    }
    loom_aie2p_native_finish_cursor_owners(&channel);
    EXPECT_EQ(channel.balance.balanced, test.balanced);
  }
}

TEST(ChannelCursorTest, AdmissionUsesConstructionLifetime) {
  for (uint32_t capacity : {1u, 3u}) {
    for (bool imported : {false, true}) {
      loom_aie2p_native_cursor_t cursor = {.worker = UINT32_MAX};
      loom_aie2p_native_cursor_bind(&cursor, capacity, 0, 1, true, imported);
      EXPECT_EQ(cursor.worker, 0u);
      EXPECT_EQ(cursor.maximum_admissions, 1u);
      EXPECT_FALSE(cursor.shared);
      EXPECT_EQ(cursor.advances, capacity > 1 && imported);
      // Repeated sites of one worker retain its bound, rather than summing it.
      loom_aie2p_native_cursor_bind(&cursor, capacity, 0, 4, true, imported);
      loom_aie2p_native_cursor_bind(&cursor, capacity, 0, 2, true, imported);
      EXPECT_EQ(cursor.maximum_admissions, 4u);
      EXPECT_FALSE(cursor.shared);
      EXPECT_EQ(cursor.advances, capacity > 1);
      // Another serial worker shares the cursor even with one local admission.
      loom_aie2p_native_cursor_bind(&cursor, capacity, 1, 1, true, imported);
      EXPECT_EQ(cursor.worker, 1u);
      EXPECT_EQ(cursor.maximum_admissions, 1u);
      EXPECT_TRUE(cursor.shared);
      EXPECT_EQ(cursor.advances, capacity > 1);
    }
  }
}

TEST(ChannelCursorTest, ExactCountRequiresEveryAdmissionSite) {
  for (bool first_exact : {false, true}) {
    for (bool second_exact : {false, true}) {
      loom_aie2p_native_cursor_t cursor = {.worker = UINT32_MAX};
      EXPECT_FALSE(loom_aie2p_native_cursor_has_exact_admissions(&cursor, 64));
      // Completion analysis supplies the endpoint total to both sites. Binding
      // the second site must neither add that total again nor erase
      // uncertainty.
      loom_aie2p_native_cursor_bind(&cursor, 2, 0, 64, first_exact, false);
      loom_aie2p_native_cursor_bind(&cursor, 2, 0, 64, second_exact, false);
      EXPECT_EQ(cursor.maximum_admissions, 64u);
      EXPECT_EQ(loom_aie2p_native_cursor_has_exact_admissions(&cursor, 64),
                first_exact && second_exact);
      EXPECT_FALSE(loom_aie2p_native_cursor_has_exact_admissions(&cursor, 63));
      EXPECT_FALSE(loom_aie2p_native_cursor_has_exact_admissions(&cursor, 65));
    }
  }
}

TEST(ChannelCursorTest, SaturatedAndSharedBoundsCannotProveCompletion) {
  for (uint32_t count : {UINT32_MAX - 1, UINT32_MAX}) {
    loom_aie2p_native_cursor_t cursor = {.worker = UINT32_MAX};
    loom_aie2p_native_cursor_bind(&cursor, 2, 0, count, true, false);
    EXPECT_EQ(loom_aie2p_native_cursor_has_exact_admissions(&cursor, count),
              count != UINT32_MAX);
    // Binding a later exact site does not repair a saturated aggregate.
    loom_aie2p_native_cursor_bind(&cursor, 2, 0, count, true, false);
    EXPECT_EQ(loom_aie2p_native_cursor_has_exact_admissions(&cursor, count),
              count != UINT32_MAX);
    // The new representative's exact local count is not the shared total.
    loom_aie2p_native_cursor_bind(&cursor, 2, 1, 64, true, true);
    EXPECT_TRUE(cursor.exact_admissions);
    EXPECT_TRUE(cursor.shared);
    EXPECT_FALSE(loom_aie2p_native_cursor_has_exact_admissions(&cursor, 64));
  }
}

class ChannelStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &registry_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&registry_));
    Register(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    Register(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    Register(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
    Register(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&registry_));
    IREE_ASSERT_OK(loom_module_allocate(&registry_, IREE_SV("channel_state"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &context_.code));
    context_.code.realization = &realization_;
    context_.family = loom_xdna_npu2_array_family();
    context_.pass = &pass_;
    pass_.arena = &arena_;
    context_.channels = channels_;
    context_.workers = workers_;
    context_.tiles = tiles_;
    context_.core_engines = engines_;
    context_.execution.engine_count = 2;
    context_.execution.workers = representatives_;
    realization_.worker_versions = versions_;
    auto& inventory = realization_.invocations;
    inventory = {
        .entries = owners_,
        .count = 3,
        .workers = occurrences_,
        .worker_count = 3,
        .channels = protocols_,
        .channel_count = 4,
    };
    const size_t first[] = {0, 1, 3};
    const size_t count[] = {1, 2, 1};
    for (size_t i = 0; i < 3; ++i) {
      owners_[i] = &invocations_[i];
      invocations_[i].ordinal = i;
      invocations_[i].first_channel = first[i];
      invocations_[i].resources = &resources_[i];
      invocations_[i].parent = i ? &invocations_[i - 1] : nullptr;
      resources_[i].channel_count = count[i];
      occurrences_[i].invocation = &invocations_[i == 1 ? 1 : 2];
    }
    for (size_t i = 0; i < 4; ++i) {
      protocols_[i].invocation = &invocations_[i == 0 ? 0 : i == 3 ? 2 : 1];
      channels_[i].cursor.reader.worker = UINT32_MAX;
      channels_[i].cursor.writer.worker = UINT32_MAX;
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&registry_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* tables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&registry_, dialect, tables, count));
  }

  std::vector<int64_t> ResetOffsets(size_t owner, size_t engine) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_symbol_ref_t symbol;
    IREE_CHECK_OK(loom_aie2p_worker_symbol(&context_.code, &symbol));
    const loom_type_t type = loom_type_buffer();
    loom_op_t* function;
    IREE_CHECK_OK(loom_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, &type, 1,
        nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    loom_builder_enter_region(&builder, function, loom_func_def_body(function));
    const loom_value_id_t frame =
        loom_region_entry_block(loom_func_def_body(function))->arg_ids[0];
    IREE_CHECK_OK(loom_aie2p_native_initialize_cursors(
        &context_, &builder, frame, engine, owners_[owner]));
    std::vector<int64_t> offsets;
    auto* block = loom_region_entry_block(loom_func_def_body(function));
    for (iree_host_size_t i = 0; i < block->op_count; ++i) {
      const auto* op = loom_block_op(block, i);
      if (!loom_view_store_isa(op)) {
        continue;
      }
      const auto* view = loom_value_def_op(
          loom_module_value(module_, loom_view_store_view(op)));
      EXPECT_EQ(loom_buffer_view_buffer(view), frame);
      const auto* offset = loom_value_def_op(
          loom_module_value(module_, loom_buffer_view_byte_offset(view)));
      offsets.push_back(loom_index_constant_value(offset).i64);
      const auto* zero = loom_value_def_op(
          loom_module_value(module_, loom_view_store_value(op)));
      EXPECT_EQ(loom_scalar_constant_value(zero).i64, 0);
    }
    return offsets;
  }

  void SelectInitializers() {
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_select_initializers(
        &context_, &realization_, &valid));
    ASSERT_TRUE(valid);
  }

  // Module and retained-plan allocation backing.
  iree_arena_block_pool_t blocks_;
  // Per-test retained state.
  iree_arena_allocator_t arena_;
  // Registered ordinary and Low IR used by state emission.
  loom_context_t registry_;
  // Generated initialization functions.
  loom_module_t* module_ = nullptr;
  // Shared production selection and emission context.
  loom_aie2p_native_context_t context_ = {};
  // Plan allocation owner.
  loom_pass_t pass_ = {};
  // Retained root, middle, and leaf invocation inventory.
  loom_pipeline_realization_t realization_ = {};
  // Channel-construction facts for each owner.
  loom_pipeline_resources_t resources_[3] = {};
  // Parent-before-child occurrences.
  loom_pipeline_invocation_t invocations_[3] = {};
  // Stable occurrence index.
  const loom_pipeline_invocation_t* owners_[3] = {};
  // Root, two middle, and leaf-local channel identities.
  loom_pipeline_invocation_channel_t protocols_[4] = {};
  // Physical channel cursor state.
  loom_aie2p_native_channel_t channels_[4] = {};
  // Two leaf roles share engine 1; the middle role uses engine 0.
  loom_pipeline_invocation_worker_t occurrences_[3] = {};
  // Admitted physical core assignment for those roles.
  iree_host_size_t engines_[3] = {1, 0, 1};
  // One representative for each physical core.
  iree_host_size_t representatives_[2] = {1, 0};
  // Selected per-role native resources.
  loom_aie2p_native_worker_t workers_[3] = {};
  // Optional registered function versions.
  loom_target_function_version_t* versions_[3] = {};
  // Physical endpoint tiles used by iterator emission.
  loom_aie2p_native_tile_t tiles_[2] = {};
};

TEST_F(ChannelStateTest, EntryResetsOnlyOwnedFrameCellsOnTheirEngine) {
  loom_aie2p_native_cursor_bind(&channels_[0].cursor.reader, 3, 0, 1, true,
                                true);
  loom_aie2p_native_cursor_bind(&channels_[1].cursor.reader, 3, 0, 1, true,
                                true);
  loom_aie2p_native_cursor_bind(&channels_[1].cursor.writer, 3, 1, 2, true,
                                false);
  loom_aie2p_native_cursor_bind(&channels_[2].cursor.reader, 3, 1, 2, true,
                                false);
  loom_aie2p_native_cursor_bind(&channels_[2].cursor.writer, 3, 0, 1, true,
                                true);
  loom_aie2p_native_cursor_bind(&channels_[3].cursor.reader, 3, 0, 2, true,
                                false);
  EXPECT_EQ(ResetOffsets(0, 1), (std::vector<int64_t>{0}));
  EXPECT_EQ(ResetOffsets(1, 1), (std::vector<int64_t>{8, 20}));
  EXPECT_TRUE(ResetOffsets(1, 0).empty());
  EXPECT_TRUE(ResetOffsets(2, 1).empty());
  // Reentry initializes the same cells, without touching the borrowed root.
  EXPECT_EQ(ResetOffsets(1, 1), (std::vector<int64_t>{8, 20}));
}

TEST_F(ChannelStateTest, SharedLocalCursorSurvivesWorkerReturns) {
  auto* cursor = &channels_[3].cursor.reader;
  loom_aie2p_native_cursor_bind(cursor, 3, 0, 1, true, false);
  EXPECT_FALSE(loom_aie2p_native_cursor_persistent(&realization_.invocations, 3,
                                                   cursor));
  loom_aie2p_native_cursor_bind(cursor, 3, 2, 1, true, false);
  EXPECT_TRUE(loom_aie2p_native_cursor_persistent(&realization_.invocations, 3,
                                                  cursor));
  EXPECT_EQ(ResetOffsets(2, 1), (std::vector<int64_t>{24}));
}

TEST_F(ChannelStateTest,
       BorrowedCapabilitySelectsCursorRegardlessOfDmaDirection) {
  loom_type_id_t payload;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_,
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(1), 0),
      &payload));
  loom_type_t read, write, mutable_read;
  IREE_ASSERT_OK(loom_read_type_make(
      module_, 0, payload, static_cast<loom_read_type_mode_t>(0), &read));
  IREE_ASSERT_OK(
      loom_read_type_make(module_, LOOM_READ_TYPE_BUILD_FLAG_HAS_MODE, payload,
                          LOOM_READ_TYPE_MODE_MUTABLE, &mutable_read));
  IREE_ASSERT_OK(loom_write_type_make(module_, payload, &write));
  auto* channel = &channels_[1];
  EXPECT_EQ(loom_aie2p_native_record_cursor(channel, read),
            &channel->cursor.reader);
  EXPECT_EQ(loom_aie2p_native_record_cursor(channel, mutable_read),
            &channel->cursor.reader);
  EXPECT_EQ(loom_aie2p_native_record_cursor(channel, write),
            &channel->cursor.writer);
}

TEST_F(ChannelStateTest, DmaIteratorUsesAdmittedControllerNotArithmeticOwner) {
  loom_aie2p_native_distribution_endpoint_t receiver = {
      .channel = &channels_[2],
  };
  receiver.controller_engine = 0;
  receiver.sequence.terminal_slot = 1;
  receiver.sequence.rewind_iteration = true;
  loom_aie2p_native_distribution_t distribution = {.invocation = owners_[2]};
  distribution.sender.channel = &channels_[1];
  distribution.sender.controller_engine = 1;
  distribution.sender.sequence.terminal_slot = 2;
  distribution.sender.sequence.rewind_iteration = true;
  distribution.receivers = &receiver;
  distribution.receiver_count = 1;
  workers_[0].distributions = &distribution;
  // Both arithmetic owners are absent. Child admission selected controller
  // engines for the autonomous endpoints and retained that decision.
  ASSERT_EQ(channels_[1].cursor.writer.worker, UINT32_MAX);
  ASSERT_EQ(channels_[2].cursor.reader.worker, UINT32_MAX);
  SelectInitializers();
  ASSERT_NE(context_.endpoint_initializers, nullptr);
  const auto* source = context_.endpoint_initializers[1 * 2 + 1];
  const auto* destination = context_.endpoint_initializers[1 * 2 + 0];
  ASSERT_NE(source, nullptr);
  ASSERT_NE(destination, nullptr);
  EXPECT_EQ(source->endpoint, &distribution.sender);
  EXPECT_EQ(destination->endpoint, &receiver);
  EXPECT_EQ(source->next, nullptr);
  EXPECT_EQ(destination->next, nullptr);
  EXPECT_EQ(context_.endpoint_initializers[2 * 2], nullptr);
  EXPECT_EQ(context_.endpoint_initializers[2 * 2 + 1], nullptr);
}

TEST_F(ChannelStateTest, RootAndWholeRingDmaNeedNoEntryRestoration) {
  loom_aie2p_native_distribution_endpoint_t receiver = {
      .channel = &channels_[1],
  };
  loom_aie2p_native_distribution_t distribution = {.invocation = owners_[2]};
  distribution.sender.channel = &channels_[0];
  distribution.sender.sequence.terminal_slot = 1;
  distribution.sender.sequence.rewind_iteration = true;
  distribution.receivers = &receiver;
  distribution.receiver_count = 1;
  workers_[0].distributions = &distribution;
  SelectInitializers();
  EXPECT_EQ(context_.endpoint_initializers, nullptr);
}

TEST_F(ChannelStateTest, IteratorRestorationWritesOnlyItsCurrentPositionWord) {
  for (bool memory : {false, true}) {
    const auto kind =
        memory ? LOOM_XDNA_TILE_KIND_MEMORY : LOOM_XDNA_TILE_KIND_COMPUTE;
    auto* tile = &tiles_[0];
    tile->coordinate = {0, static_cast<uint8_t>(memory ? 1 : 2)};
    tile->facts = loom_xdna_array_tile_kind_facts(context_.family, kind);
    tile->memory_layout =
        memory ? &loom_aie2p_sram_memory : &loom_aie2p_compute_memory;
    const auto* layout = tile->memory_layout;
    loom_aie2p_native_distribution_t distribution = {.invocation = owners_[2]};
    auto* endpoint = &distribution.sender;
    endpoint->channel = &channels_[1];
    endpoint->tile = tile;
    endpoint->controller_engine = 1;
    uint32_t descriptor[8] = {};
    descriptor[0] = 1 - tile->facts->dma.transfer_length_offset;
    descriptor[layout->lock_word] =
        loom_xdna_register_field_encode(layout->locks.valid, 1);
    iree_string_view_t reason;
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_build_sequence(
        &tile->facts->dma, layout, 2, tile->facts->dma.buffer_descriptor_count,
        LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM, 2, 3, 128, 4, 1, descriptor,
        &arena_, &endpoint->sequence, &reason, &valid));
    ASSERT_TRUE(valid);
    ASSERT_TRUE(endpoint->sequence.rewind_iteration);
    workers_[0].distributions = &distribution;
    uint8_t packets[] = {7, 0};
    workers_[0].control_packets = packets;
    context_.endpoint_initializers = nullptr;
    SelectInitializers();
    loom_builder_t builder;
    loom_symbol_ref_t caller;
    loom_op_t* function;
    IREE_ASSERT_OK(loom_aie2p_worker_helper(&context_.code, nullptr, 0, nullptr,
                                            0, &builder, &caller, &function));
    IREE_ASSERT_OK(loom_aie2p_native_initialize_endpoints(&context_, &builder,
                                                          1, owners_[1]));
    auto* block = loom_region_entry_block(loom_low_func_def_body(function));
    ASSERT_EQ(block->op_count, 1u);
    const auto* call = block->first_op;
    ASSERT_TRUE(loom_low_invoke_isa(call));
    const auto symbol = loom_low_invoke_callee(call);
    auto* helper = module_->symbols.entries[symbol.symbol_id].defining_op;
    auto* body = loom_region_entry_block(loom_low_func_def_body(helper));
    const auto* descriptors = context_.code.descriptors;
    const uint32_t stream = loom_low_descriptor_set_descriptor_ordinal(
        descriptors,
        &descriptors->descriptors[AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32]);
    const uint32_t last = loom_low_descriptor_set_descriptor_ordinal(
        descriptors,
        &descriptors
             ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32]);
    std::vector<uint32_t> words;
    for (auto* op = body->first_op; op; op = op->next_op) {
      if (!loom_low_op_isa(op) || (loom_low_op_descriptor(op) != stream &&
                                   loom_low_op_descriptor(op) != last)) {
        continue;
      }
      const auto* constant = loom_value_def_op(
          loom_module_value(module_, loom_op_operands(op)[0]));
      const auto attrs = loom_low_const_isa(constant)
                             ? loom_low_const_attrs(constant)
                             : loom_low_op_attrs(constant);
      for (iree_host_size_t i = 0; i < attrs.count; ++i) {
        if (attrs.entries[i].name_id == context_.code.integer_name) {
          words.push_back(static_cast<uint32_t>(attrs.entries[i].value.i64));
        }
      }
      EXPECT_EQ(loom_low_op_descriptor(op) == last, words.size() == 3u);
    }
    const uint16_t ordinal = 2;
    const uint64_t origin = uint64_t(tile->coordinate.row)
                            << context_.family->row_shift;
    const uint32_t address = loom_xdna_register_field_address(
                                 context_.family, layout->iteration.current,
                                 tile->coordinate, &ordinal) -
                             origin;
    const uint32_t iteration =
        loom_xdna_register_field_encode(layout->iteration.step, 0) |
        loom_xdna_register_field_encode(layout->iteration.wrap, 2) |
        loom_xdna_register_field_encode(layout->iteration.current, 0);
    EXPECT_EQ(words, (std::vector<uint32_t>{
                         7, loom_aie2p_worker_control_write_command(address, 1),
                         iteration}));
  }
}

}  // namespace
