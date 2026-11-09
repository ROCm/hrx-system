// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

namespace {

class PhaseRetirementTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &registry_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&registry_));
    iree_host_size_t count;
    const auto* tables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&registry_, LOOM_DIALECT_LOW,
                                                 tables, count));
    IREE_ASSERT_OK(loom_context_finalize(&registry_));
    IREE_ASSERT_OK(loom_module_allocate(&registry_, IREE_SV("retirement"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &context_.code));
    realization_.worker_versions = versions_;
    context_.code.realization = &realization_;
    context_.family = loom_xdna_npu2_array_family();
    context_.workers = &worker_;
    context_.core_engines = &engine_;
    context_.execution.workers = &engine_;
    context_.tiles = &tile_;
    const bool memory = std::get<0>(GetParam());
    tile_.coordinate = {0, static_cast<uint16_t>(memory ? 1 : 2)};
    tile_.facts = loom_xdna_array_tile_facts(context_.family, tile_.coordinate);
    tile_.memory_layout =
        memory ? &loom_aie2p_sram_memory : &loom_aie2p_compute_memory;
    endpoint_.tile = &tile_;
    endpoint_.engine = 1;
    command_.invocation = &invocation_;
    command_.endpoint = &endpoint_;
    command_.direction = std::get<1>(GetParam())
                             ? LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                             : LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY;
    command_.retirement = &retirement_;
    retirement_.source = &command_;
    retirement_.function = loom_symbol_ref_null();
    worker_.distribution_commands = &command_;
    worker_.control_packets = &packet_;
    worker_.control_replies = &reply_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&registry_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  loom_op_t* Boundary(const loom_pipeline_invocation_t* invocation) {
    loom_builder_t builder;
    loom_symbol_ref_t symbol;
    loom_op_t* function;
    IREE_CHECK_OK(loom_aie2p_worker_helper(&context_.code, nullptr, 0, nullptr,
                                           0, &builder, &symbol, &function));
    IREE_CHECK_OK(loom_aie2p_native_distribution_boundary(
        &context_, &builder, 0, invocation, false));
    auto* call =
        loom_region_entry_block(loom_low_func_def_body(function))->first_op;
    EXPECT_TRUE(loom_low_invoke_isa(call));
    return module_->symbols.entries[loom_low_invoke_callee(call).symbol_id]
        .defining_op;
  }

  // Module and plan allocation backing.
  iree_arena_block_pool_t blocks_;
  // Selected native state and helper construction storage.
  iree_arena_allocator_t arena_;
  // Ordinary Low operation registry.
  loom_context_t registry_;
  // Generated retirement helpers and their direct callers.
  loom_module_t* module_ = nullptr;
  // Production emission context.
  loom_aie2p_native_context_t context_ = {};
  // Physical worker without optional function-version registration.
  loom_pipeline_realization_t realization_ = {};
  // Absent version owner for standalone native emission.
  loom_target_function_version_t* versions_[1] = {};
  // The single admitted controller and representative.
  iree_host_size_t engine_ = 0;
  // Control packet destination for this endpoint.
  uint8_t packet_ = 3;
  // Private response route already admitted for the controller.
  uint8_t reply_ = 5;
  // Physical SRAM or compute DMA endpoint.
  loom_aie2p_native_tile_t tile_ = {};
  // Canonical child occurrence owning the retirement boundary.
  loom_pipeline_invocation_t invocation_ = {};
  // Physical endpoint state consumed by the controller.
  loom_aie2p_native_distribution_endpoint_t endpoint_ = {};
  // Selected per-occurrence endpoint command.
  loom_aie2p_native_distribution_command_t command_ = {};
  // Static use inventory, independent of runtime loop visits.
  loom_aie2p_native_retirement_t retirement_ = {};
  // Controller carrying private request and response routes.
  loom_aie2p_native_worker_t worker_ = {};
};

TEST_P(PhaseRetirementTest, OneStaticSiteKeepsItsPollingBodyInline) {
  retirement_.use_count = 1;
  auto* boundary = Boundary(&invocation_);
  EXPECT_FALSE(loom_symbol_ref_is_valid(retirement_.function));
  // The poll has a CFG backedge; it is not an empty helper or an outlined call.
  auto* body = loom_low_func_def_body(boundary);
  EXPECT_GT(body->block_count, 1u);
  for (iree_host_size_t b = 0; b < body->block_count; ++b) {
    for (auto* op = body->blocks[b]->first_op; op; op = op->next_op) {
      EXPECT_FALSE(loom_low_func_call_isa(op));
    }
  }
}

TEST_P(PhaseRetirementTest, DistinctStaticSitesCallOneRetirementBody) {
  retirement_.use_count = 2;
  for (size_t site = 0; site < 2; ++site) {
    auto* boundary = Boundary(&invocation_);
    ASSERT_TRUE(loom_symbol_ref_is_valid(retirement_.function));
    auto* block = loom_region_entry_block(loom_low_func_def_body(boundary));
    ASSERT_EQ(block->op_count, 2u);
    ASSERT_TRUE(loom_low_func_call_isa(block->first_op));
    EXPECT_EQ(loom_low_func_call_callee(block->first_op).symbol_id,
              retirement_.function.symbol_id);
    EXPECT_TRUE(loom_low_return_isa(block->last_op));
  }
  auto* helper =
      module_->symbols.entries[retirement_.function.symbol_id].defining_op;
  const auto callable = loom_func_like_cast(module_, helper);
  EXPECT_EQ(loom_func_like_inline_policy(callable),
            LOOM_INLINE_POLICY_NOINLINE);
  EXPECT_GT(loom_low_func_def_body(helper)->block_count, 1u);
}

TEST_P(PhaseRetirementTest, AnotherChildDoesNotRetireThisEndpoint) {
  retirement_.use_count = 2;
  const loom_pipeline_invocation_t other = {};
  auto* boundary = Boundary(&other);
  auto* body = loom_low_func_def_body(boundary);
  ASSERT_EQ(body->block_count, 1u);
  ASSERT_EQ(body->blocks[0]->op_count, 1u);
  EXPECT_TRUE(loom_low_return_isa(body->blocks[0]->first_op));
  EXPECT_FALSE(loom_symbol_ref_is_valid(retirement_.function));
}

INSTANTIATE_TEST_SUITE_P(MemoryAndDirection, PhaseRetirementTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
