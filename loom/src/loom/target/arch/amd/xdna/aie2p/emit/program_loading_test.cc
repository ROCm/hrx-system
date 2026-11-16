// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_loading.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/callable_compile.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"
#include "loom/target/profile.h"

namespace loom {
namespace {

class Aie2pProgramLoadingTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_aie2p_target_provider.register_context(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("loading"), &blocks_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // Shared backing for the generated helper and its detached native result.
  iree_arena_block_pool_t blocks_;
  // Native compilation result lifetime.
  iree_arena_allocator_t arena_;
  // Source operation and target descriptor vocabulary.
  loom_context_t context_ = {};
  // Private workspace populated by the production loading builder.
  loom_module_t* module_ = nullptr;
};

// Inspect the final scheduled instructions, not the builder's insertion order:
// the old endpoint must be inert before replacement starts, and callers must
// observe its clobber through the ordinary native callable contract.
TEST_P(Aie2pProgramLoadingTest, DisablesLoopEndpointBeforeIssuingCodeTransfer) {
  loom_symbol_ref_t helper;
  IREE_ASSERT_OK(loom_aie2p_program_loading_build(module_, GetParam(),
                                                  0x800A0644u, &helper));
  loom_target_low_descriptor_registry_t registry;
  loom_aie2p_low_descriptor_registry_initialize(&registry);
  const loom_aie2p_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_aie2p_target_profile_select(IREE_SV("core"), &profile));
  loom_target_facts_t* facts = nullptr;
  IREE_ASSERT_OK(
      loom_target_profile_project_facts(&profile->base, &arena_, &facts));
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &registry.registry,
      .function_target_facts = facts,
  };
  bool compiled = false;
  loom_aie2p_callable_t callable;
  IREE_ASSERT_OK(loom_aie2p_callable_compile(
      module_, module_->symbols.entries[helper.symbol_id].defining_op, &options,
      {}, &arena_, &compiled, &callable));
  ASSERT_TRUE(compiled);

  const auto endpoint =
      loom_aie2p_machine_find_physical_register(IREE_SV("le"));
  EXPECT_TRUE(loom_aie2p_leaf_may_write_register(
      &callable.contribution.realization, endpoint));
  bool clobbers_endpoint = false;
  for (iree_host_size_t i = 0; i < callable.call_contract.clobbers.count; ++i) {
    clobbers_endpoint |=
        callable.call_contract.clobbers.values[i].location == endpoint;
  }
  EXPECT_TRUE(clobbers_endpoint);

  const int64_t disabled_address = 0xFFFFF;
  const auto reset = loom_aie2p_descriptor_encode(
      loom_aie2p_core_descriptor_set(),
      AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_END_ADDRESS, nullptr,
      &disabled_address);
  const auto stream_write = loom_aie2p_encoding_find_instruction(
      IREE_SV("MOV_st_mMStream_tlast_imm"));
  ASSERT_NE(stream_write, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  const auto& object = callable.contribution.object;
  const auto& entry =
      object.symbols[callable.contribution.realization.entry_symbol_index];
  const auto code = object.sections[entry.section_contribution_index].contents;
  iree_host_size_t reset_packet = IREE_HOST_SIZE_MAX;
  iree_host_size_t first_write_packet = IREE_HOST_SIZE_MAX;
  iree_host_size_t packet = 0;
  for (iree_host_size_t offset = 0; offset < code.data_length; ++packet) {
    loom_aie2p_decoded_bundle_t bundle;
    iree_host_size_t length = 0;
    IREE_ASSERT_OK(loom_aie2p_encoding_decode_bundle_prefix(
        iree_make_const_byte_span(code.data + offset,
                                  iree_min(code.data_length - offset, 16u)),
        &bundle, &length));
    for (uint8_t i = 0; i < bundle.slot_count; ++i) {
      const auto slot = bundle.slots[i];
      if (slot.slot == reset.slot && slot.value == reset.value) {
        EXPECT_EQ(reset_packet, IREE_HOST_SIZE_MAX);
        reset_packet = packet;
      }
      loom_aie2p_instruction_id_t candidates[16];
      const auto count = loom_aie2p_encoding_query_instruction_candidates(
          slot.slot, slot.value, IREE_ARRAYSIZE(candidates), candidates);
      ASSERT_LE(count, IREE_ARRAYSIZE(candidates));
      for (iree_host_size_t j = 0; j < count; ++j) {
        if (candidates[j] == stream_write) {
          first_write_packet = iree_min(first_write_packet, packet);
        }
      }
    }
    offset += length;
  }
  ASSERT_NE(first_write_packet, IREE_HOST_SIZE_MAX);
  EXPECT_LT(reset_packet, first_write_packet);
}

INSTANTIATE_TEST_SUITE_P(QueueHeaderEncodings, Aie2pProgramLoadingTest,
                         ::testing::Values(2u, 0x80000003u));

}  // namespace
}  // namespace loom
