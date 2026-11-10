// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_submission.h"

#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/types.h"

namespace {

class DmaSubmissionTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &registry_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&registry_));
    iree_host_size_t count;
    const auto* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&registry_, LOOM_DIALECT_LOW,
                                                 vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&registry_));
    IREE_ASSERT_OK(loom_module_allocate(&registry_, IREE_SV("submission"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &context_.code));
    realization_.worker_versions = versions_;
    context_.code.realization = &realization_;
    context_.family = loom_xdna_npu2_array_family();
    context_.pass = &pass_;
    pass_.arena = &arena_;
    const bool memory = std::get<0>(GetParam());
    local_.coordinate = {0, static_cast<uint8_t>(memory ? 1 : 2)};
    local_.memory_layout =
        memory ? &loom_aie2p_sram_memory : &loom_aie2p_compute_memory;
    shim_.coordinate = {0, 0};
    path_.local = &local_;
    path_.shim = &shim_;
    path_.local_engine = memory ? 1 : 0;
    path_.ingress = std::get<1>(GetParam());
    path_.completion_tokens = memory ? 2 : path_.ingress ? 0 : 1;
    path_.control.local = 1;
    path_.control.shim = 2;
    group_.use_count = 3;
    group_.function = loom_symbol_ref_null();
    task_.path = &path_;
    task_.first_transfer = transfers_;
    task_.last_transfer = &transfers_[1];
    task_.transfer_count = 2;
    task_.local_descriptor = &local_descriptor_;
    task_.shim_descriptor = &shim_descriptor_;
    local_descriptor_.path = shim_descriptor_.path = &path_;
    local_descriptor_.sequence.descriptor_count = 2;
    shim_descriptor_.sequence.descriptor_count = 2;
    local_descriptor_.sequence.descriptor = 2;
    shim_descriptor_.sequence.descriptor = 8;
    for (unsigned i = 0; i < 2; ++i) {
      transfers_[i].task = &task_;
      transfers_[i].descriptor_ordinal = i;
      transfers_[i].relocation = &relocations_[i];
      relocations_[i].load_address = 0x70000 + i * 8;
    }
    transfers_[0].task_next = &transfers_[1];
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&registry_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  loom_block_t* Body(loom_symbol_ref_t symbol) {
    return loom_region_entry_block(loom_low_func_def_body(
        module_->symbols.entries[symbol.symbol_id].defining_op));
  }

  uint32_t Immediate(loom_value_id_t value, loom_string_id_t name) {
    const auto* op = loom_value_def_op(loom_module_value(module_, value));
    if (!op || (!loom_low_const_isa(op) && !loom_low_op_isa(op))) {
      ADD_FAILURE() << "expected a materialized integer";
      return 0;
    }
    const auto attrs = loom_low_const_isa(op) ? loom_low_const_attrs(op)
                                              : loom_low_op_attrs(op);
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (attrs.entries[i].name_id == name) {
        return static_cast<uint32_t>(attrs.entries[i].value.i64);
      }
    }
    ADD_FAILURE() << "missing integer attribute";
    return 0;
  }

  uint32_t Integer(loom_value_id_t value) {
    return Immediate(value, context_.code.integer_name);
  }

  void CheckOperation(const loom_op_t* op, uint32_t descriptor,
                      iree_host_size_t operand_count) {
    ASSERT_NE(op, nullptr);
    ASSERT_TRUE(loom_low_op_isa(op));
    const auto* descriptors = context_.code.descriptors;
    EXPECT_EQ(loom_low_op_descriptor(op),
              loom_low_descriptor_set_descriptor_ordinal(
                  descriptors, &descriptors->descriptors[descriptor]));
    ASSERT_EQ(op->operand_count, operand_count);
  }

  // The shim receives base + offset as two words. The high word must include
  // carry from the low word, including when one patch body serves many sites.
  void CheckAddress(loom_value_id_t low, loom_value_id_t high,
                    loom_value_id_t offset, loom_value_id_t base,
                    uint32_t load_address) {
    const auto definition = [&](loom_value_id_t value) {
      return loom_value_def_op(loom_module_value(module_, value));
    };
    const auto* low_sum = definition(low);
    const auto* high_sum = definition(high);
    ASSERT_NO_FATAL_FAILURE(
        CheckOperation(low_sum, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, 2));
    ASSERT_NO_FATAL_FAILURE(
        CheckOperation(high_sum, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, 2));
    const auto original_low = loom_op_operands(low_sum)[0];
    EXPECT_EQ(loom_op_operands(low_sum)[1], offset);
    const auto* carry = definition(loom_op_operands(high_sum)[1]);
    ASSERT_NO_FATAL_FAILURE(
        CheckOperation(carry, AIE2P_CORE_DESCRIPTOR_REF_CMP_ULT_I32, 2));
    EXPECT_EQ(loom_op_operands(carry)[0], low);
    EXPECT_EQ(loom_op_operands(carry)[1], original_low);
    const loom_value_id_t words[] = {original_low,
                                     loom_op_operands(high_sum)[0]};
    loom_value_id_t loaded_base = LOOM_VALUE_ID_INVALID;
    for (unsigned i = 0; i < 2; ++i) {
      const auto* load = definition(words[i]);
      ASSERT_NO_FATAL_FAILURE(CheckOperation(
          load, AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_I32_INDEXED_IMMEDIATE,
          1));
      EXPECT_EQ(Immediate(words[i], context_.code.displacement_name), i * 4);
      if (i == 0) {
        loaded_base = loom_op_operands(load)[0];
      } else {
        EXPECT_EQ(loom_op_operands(load)[0], loaded_base);
      }
    }
    if (base != LOOM_VALUE_ID_INVALID) {
      EXPECT_EQ(loaded_base, base);
    } else {
      ASSERT_NO_FATAL_FAILURE(CheckOperation(
          definition(loaded_base),
          AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32, 0));
      EXPECT_EQ(Integer(loaded_base), load_address);
    }
  }

  struct Word {
    // Value written to the control stream.
    loom_value_id_t value;
    // This word closes the current control packet.
    bool last;
  };

  std::vector<Word> Writes(loom_block_t* body) {
    const auto* descriptors = context_.code.descriptors;
    const uint32_t write = loom_low_descriptor_set_descriptor_ordinal(
        descriptors,
        &descriptors->descriptors[AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32]);
    const uint32_t last = loom_low_descriptor_set_descriptor_ordinal(
        descriptors,
        &descriptors
             ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32]);
    std::vector<Word> words;
    for (iree_host_size_t i = 0; i < body->op_count; ++i) {
      const auto* op = loom_block_op(body, i);
      if (!loom_low_op_isa(op)) {
        continue;
      }
      const uint32_t descriptor = loom_low_op_descriptor(op);
      if (descriptor == write || descriptor == last) {
        words.push_back({loom_op_operands(op)[0], descriptor == last});
      }
    }
    return words;
  }

  uint32_t Command(bool shim, loom_xdna_register_field_id_t field,
                   uint16_t index, unsigned count = 1) {
    const auto coordinate = shim ? shim_.coordinate : local_.coordinate;
    const uint64_t origin =
        (uint64_t)coordinate.column << context_.family->column_shift |
        (uint64_t)coordinate.row << context_.family->row_shift;
    return loom_aie2p_worker_control_write_command(
        loom_xdna_register_field_address(context_.family, field, coordinate,
                                         &index) -
            origin,
        count);
  }

  // Each endpoint's queue payload is the selected first BD and its token bit.
  uint32_t QueueWord(bool shim) {
    const bool token =
        path_.completion_tokens && (shim || path_.completion_tokens == 2);
    return (shim ? shim_descriptor_ : local_descriptor_).sequence.descriptor |
           (token ? UINT32_C(0x80000000) : 0);
  }

  void CheckQueues(const std::vector<Word>& words, loom_value_id_t local,
                   loom_value_id_t shim) {
    ASSERT_GE(words.size(), 6u);
    const auto local_queue =
        local_.memory_layout->directions[path_.ingress ? 1 : 0].start;
    const auto shim_queue =
        path_.ingress
            ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
            : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID;
    // The receiver must be queued first in either transfer direction.
    for (unsigned i = 0; i < 2; ++i) {
      const bool is_shim = path_.ingress ? i == 1 : i == 0;
      const size_t position = words.size() - 6 + i * 3;
      EXPECT_EQ(Integer(words[position].value), is_shim ? 2u : 1u);
      EXPECT_EQ(Integer(words[position + 1].value),
                Command(is_shim, is_shim ? shim_queue : local_queue,
                        is_shim ? path_.shim_engine : path_.local_engine));
      if ((is_shim ? shim : local) != LOOM_VALUE_ID_INVALID) {
        EXPECT_EQ(words[position + 2].value, is_shim ? shim : local);
      } else {
        EXPECT_EQ(Integer(words[position + 2].value), QueueWord(is_shim));
      }
      EXPECT_FALSE(words[position].last);
      EXPECT_FALSE(words[position + 1].last);
      EXPECT_TRUE(words[position + 2].last);
    }
  }

  void CheckShared(bool local_dynamic) {
    group_.local_queue_argument = !local_dynamic;
    task_.submit_group = &group_;
    local_descriptor_.address_dynamic = local_dynamic;
    shim_descriptor_.address_dynamic = true;
    for (unsigned site = 0; site < 3; ++site) {
      local_descriptor_.sequence.descriptor = local_dynamic ? 2 : 2 + site * 2;
      shim_descriptor_.sequence.descriptor = 8 + site * 2;
      IREE_ASSERT_OK(loom_aie2p_native_emit_submission(&context_, &task_));
      auto* adapter = Body(task_.submit);
      ASSERT_EQ(adapter->arg_count, local_dynamic ? 4u : 2u);
      const loom_op_t* call = nullptr;
      for (iree_host_size_t i = 0; i < adapter->op_count; ++i) {
        const auto* op = loom_block_op(adapter, i);
        if (loom_low_func_call_isa(op)) {
          ASSERT_EQ(call, nullptr);
          call = op;
        }
      }
      ASSERT_NE(call, nullptr);
      EXPECT_EQ(loom_low_func_call_callee(call).symbol_id,
                group_.function.symbol_id);
      ASSERT_EQ(call->operand_count, local_dynamic ? 9u : 8u);
      const auto* operands = loom_op_operands(call);
      for (unsigned i = 0; i < 2; ++i) {
        const unsigned position = i * (local_dynamic ? 4 : 3);
        EXPECT_EQ(operands[position],
                  adapter->arg_ids[i * (local_dynamic ? 2 : 1)]);
        EXPECT_EQ(Integer(operands[position + 1]),
                  relocations_[i].load_address);
        EXPECT_EQ(
            Integer(operands[position + 2]),
            Command(
                true,
                LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD1_BASE_ADDRESS_LOW,
                shim_descriptor_.sequence.descriptor + i, 2));
        if (local_dynamic) {
          EXPECT_EQ(operands[position + 3], adapter->arg_ids[i * 2 + 1]);
        }
      }
      EXPECT_EQ(Integer(operands[local_dynamic ? 8 : 6]), QueueWord(true));
      if (!local_dynamic) {
        EXPECT_EQ(Integer(operands[7]), QueueWord(false));
      }
    }
    // Three adapters reference one body, with per-site queue/patch selection.
    EXPECT_EQ(loom_module_block(module_)->op_count, 4u);
    auto* shared = Body(group_.function);
    ASSERT_EQ(shared->arg_count, local_dynamic ? 9u : 8u);
    const auto words = Writes(shared);
    ASSERT_EQ(words.size(), local_dynamic ? 20u : 14u);
    for (unsigned i = 0; i < 2; ++i) {
      size_t position = i * (local_dynamic ? 7 : 4);
      const unsigned argument = i * (local_dynamic ? 4 : 3);
      if (local_dynamic) {
        EXPECT_EQ(Integer(words[position].value), 1u);
        EXPECT_EQ(Integer(words[position + 1].value),
                  Command(false, local_.memory_layout->address, 2 + i));
        EXPECT_EQ(words[position + 2].value, shared->arg_ids[argument + 3]);
        EXPECT_FALSE(words[position].last);
        EXPECT_FALSE(words[position + 1].last);
        EXPECT_TRUE(words[position + 2].last);
        position += 3;
      }
      EXPECT_EQ(Integer(words[position].value), 2u);
      EXPECT_EQ(words[position + 1].value, shared->arg_ids[argument + 2]);
      EXPECT_FALSE(words[position].last);
      EXPECT_FALSE(words[position + 1].last);
      EXPECT_FALSE(words[position + 2].last);
      EXPECT_TRUE(words[position + 3].last);
      CheckAddress(words[position + 2].value, words[position + 3].value,
                   shared->arg_ids[argument], shared->arg_ids[argument + 1], 0);
    }
    CheckQueues(words,
                local_dynamic ? LOOM_VALUE_ID_INVALID : shared->arg_ids[7],
                shared->arg_ids[local_dynamic ? 8 : 6]);
  }

  // Module and transient code-generation allocation lifetime.
  iree_arena_block_pool_t blocks_ = {};
  // Submission planning scratch.
  iree_arena_allocator_t arena_ = {};
  // Registered IR vocabulary used by the real worker builder.
  loom_context_t registry_ = {};
  // Destination of generated functions and adapters.
  loom_module_t* module_ = nullptr;
  // Pass-owned allocator consumed by submission emission.
  loom_pass_t pass_ = {};
  // Physical target and worker emission state.
  loom_aie2p_native_context_t context_ = {};
  // Realization with no function-version owner, as in standalone lowering.
  loom_pipeline_realization_t realization_ = {};
  // One standalone worker without version bookkeeping.
  loom_target_function_version_t* versions_[1] = {};
  // On-chip endpoint, either compute or SRAM.
  loom_aie2p_native_tile_t local_ = {};
  // External endpoint in the shim tile.
  loom_aie2p_native_tile_t shim_ = {};
  // Admitted engine pair, routes, and completion policy.
  loom_aie2p_native_dma_path_t path_ = {};
  // Shared patch body selected for three uses.
  loom_aie2p_native_dma_submit_t group_ = {};
  // Task containing two independently addressed transfer segments.
  loom_aie2p_native_transfer_task_t task_ = {};
  // Selected local descriptor chain.
  loom_aie2p_native_dma_descriptor_t local_descriptor_ = {};
  // Selected external descriptor chain.
  loom_aie2p_native_dma_descriptor_t shim_descriptor_ = {};
  // Finalized segment facts passed to the emitter.
  loom_aie2p_native_transfer_t transfers_[2] = {};
  // Worker-local base address slots for each segment's external binding.
  loom_aie2p_native_relocation_t relocations_[2] = {};
};

TEST_P(DmaSubmissionTest, FixedVariantsBindBothQueuesToOnePatchBody) {
  CheckShared(false);
}

TEST_P(DmaSubmissionTest, DynamicLocalAddressesBindEachSegmentToItsPatch) {
  CheckShared(true);
}

TEST_P(DmaSubmissionTest, FixedAddressesOnlyEnqueueReceiverThenSender) {
  IREE_ASSERT_OK(loom_aie2p_native_emit_submission(&context_, &task_));
  auto* body = Body(task_.submit);
  EXPECT_EQ(body->arg_count, 0u);
  const auto words = Writes(body);
  ASSERT_EQ(words.size(), 6u);
  CheckQueues(words, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID);
}

TEST_P(DmaSubmissionTest, DynamicExternalAddressesPatchBothWordsBeforeEnqueue) {
  shim_descriptor_.address_dynamic = true;
  IREE_ASSERT_OK(loom_aie2p_native_emit_submission(&context_, &task_));
  auto* body = Body(task_.submit);
  ASSERT_EQ(body->arg_count, 2u);
  const auto words = Writes(body);
  ASSERT_EQ(words.size(), 14u);
  for (unsigned i = 0; i < 2; ++i) {
    const unsigned position = i * 4;
    EXPECT_EQ(Integer(words[position].value), 2u);
    EXPECT_EQ(
        Integer(words[position + 1].value),
        Command(true,
                LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD1_BASE_ADDRESS_LOW,
                shim_descriptor_.sequence.descriptor + i, 2));
    CheckAddress(words[position + 2].value, words[position + 3].value,
                 body->arg_ids[i], LOOM_VALUE_ID_INVALID,
                 relocations_[i].load_address);
    EXPECT_FALSE(words[position].last);
    EXPECT_FALSE(words[position + 1].last);
    EXPECT_FALSE(words[position + 2].last);
    EXPECT_TRUE(words[position + 3].last);
  }
  CheckQueues(words, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID);
}

INSTANTIATE_TEST_SUITE_P(MemoryAndDirection, DmaSubmissionTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
