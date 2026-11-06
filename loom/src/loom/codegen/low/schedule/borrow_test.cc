// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/borrow.h"

#include <algorithm>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

// A typed, single-block snapshot isolates the lifetime decision from scheduler
// policy. Authored CFG, calls, and physical execution are covered separately.
class BorrowPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, uint16_t(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("borrow"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_string_id_t namespace_id, payload_id, contract_id, name_id;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("media"), &namespace_id));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("frame"), &payload_id));
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("test.low.core"),
                                             &contract_id));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("project"), &name_id));
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol));
    const auto* descriptors = loom_test_low_core_descriptor_set();
    uint16_t register_class = 0;
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptors, IREE_SV("test.ptr"), &register_class, nullptr));
    const uint64_t carrier =
        loom_low_register_type_pack_payload1(register_class, 1);
    IREE_ASSERT_OK(loom_module_intern_register_type(
        module_, descriptors->stable_id, carrier,
        loom_type_reference(LOOM_REFERENCE_OWNERSHIP_OWNED, namespace_id,
                            payload_id, 0),
        &owner_type_));
    IREE_ASSERT_OK(loom_module_intern_register_type(
        module_, descriptors->stable_id, carrier,
        loom_type_reference(LOOM_REFERENCE_OWNERSHIP_BORROWED, namespace_id,
                            payload_id, 0),
        &pointer_type_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, 0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*effect_contract=*/0, /*temperature=*/0, /*inline_policy=*/0,
        /*allocation=*/0, /*schedule=*/0, contract_id, /*target=*/{}, /*abi=*/0,
        {}, {}, LOOM_STRING_ID_INVALID, {}, {0, symbol}, &owner_type_, 1,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicate_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function));
    auto* region = loom_low_func_def_body(function);
    loom_builder_enter_region(&builder_, function, region);
    owner_ = loom_region_entry_block(region)->arg_ids[0];
    values_.push_back(owner_);
    schedule_.module = module_;
    schedule_.function_op = function;
    schedule_.target.descriptor_set = descriptors;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  uint32_t Append(loom_op_t* op) {
    const uint32_t index = uint32_t(nodes_.size());
    loom_low_schedule_node_t node = {
        .op = op,
        .scheduled_ordinal = index,
        .operand_count = op->operand_count,
        .result_count = op->result_count,
    };
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      const auto value = loom_op_operands(op)[i];
      const auto found = std::find(values_.begin(), values_.end(), value);
      EXPECT_NE(found, values_.end());
      loom_low_schedule_node_operand_ordinals(&node)[i] =
          loom_value_ordinal_t(found - values_.begin());
    }
    for (uint16_t i = 0; i < op->result_count; ++i) {
      loom_low_schedule_node_result_ordinals(&node)[i] =
          loom_value_ordinal_t(values_.size());
      values_.push_back(loom_op_results(op)[i]);
    }
    nodes_.push_back(node);
    return index;
  }

  loom_value_id_t Borrow() {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_borrow_build(&builder_, owner_, pointer_type_,
                                        LOOM_LOCATION_UNKNOWN, &op));
    Append(op);
    ++schedule_.requirements.borrow_count;
    return loom_low_borrow_pointer(op);
  }

  loom_value_id_t Copy(loom_value_id_t pointer) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_copy_build(&builder_, pointer, false, pointer_type_,
                                      LOOM_LOCATION_UNKNOWN, &op));
    Append(op);
    return loom_low_copy_result(op);
  }

  uint32_t Release(loom_value_id_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(
        loom_low_release_build(&builder_, value, LOOM_LOCATION_UNKNOWN, &op));
    return Append(op);
  }

  void BuildPlan() {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &op));
    Append(op);
    schedule_.nodes = nodes_.data();
    schedule_.node_count = uint32_t(nodes_.size());
    schedule_.value_ids = values_.data();
    schedule_.value_count = loom_value_ordinal_t(values_.size());
    IREE_ASSERT_OK(loom_low_schedule_borrow_plan_build(
        &schedule_, nullptr, &scratch_, &arena_, &schedule_.borrows));
  }

  // Shared backing for IR and independent planning arenas.
  iree_arena_block_pool_t pool_ = {};
  // Retained plan output.
  iree_arena_allocator_t arena_ = {};
  // Disposable indexed proof workspace.
  iree_arena_allocator_t scratch_ = {};
  // Registered structural operation semantics.
  loom_context_t context_ = {};
  // Owns the typed function and its operations.
  loom_module_t* module_ = nullptr;
  // Appends instructions to the function's single block.
  loom_builder_t builder_ = {};
  // Owning register type with an external nominal identity.
  loom_type_t owner_type_ = {};
  // Borrowed register type with the same nominal identity.
  loom_type_t pointer_type_ = {};
  // Function's explicitly consumed owner argument.
  loom_value_id_t owner_ = LOOM_VALUE_ID_INVALID;
  // Retained source-order nodes; scheduled order is explicit on each node.
  std::vector<loom_low_schedule_node_t> nodes_;
  // Function-local dense value domain.
  std::vector<loom_value_id_t> values_;
  // Snapshot supplied to the proof and its diagnostic consumer.
  loom_low_schedule_table_t schedule_ = {};
};

TEST_F(BorrowPlanTest, NoBorrowAllocatesNoProofState) {
  Release(owner_);
  const auto retained_before = arena_.used_allocation_size;
  const auto scratch_before = scratch_.used_allocation_size;
  BuildPlan();
  EXPECT_EQ(schedule_.borrows.count, 0u);
  EXPECT_EQ(schedule_.borrows.values, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, retained_before);
  EXPECT_EQ(scratch_.used_allocation_size, scratch_before);
}

TEST_F(BorrowPlanTest, CoverageIncludesEveryCopyButNotDeadCarrierCleanup) {
  const auto pointer = Borrow();
  const auto copy = Copy(pointer);
  const auto last_copy = Copy(copy);
  Release(owner_);
  Release(pointer);
  Release(copy);
  Release(last_copy);
  BuildPlan();
  ASSERT_EQ(schedule_.borrows.count, 1u);
  const auto& decision = schedule_.borrows.decisions[0];
  EXPECT_EQ(decision.protection, LOOM_LOW_BORROW_COVERED);
  EXPECT_EQ(decision.last_use_node, 2u);
  EXPECT_EQ(decision.reason_node, UINT32_MAX);
  EXPECT_EQ(schedule_.borrows.values[1], LOOM_LOW_BORROW_COVERED);
  EXPECT_EQ(schedule_.borrows.values[2], LOOM_LOW_BORROW_NONE);
  EXPECT_EQ(schedule_.borrows.values[3], LOOM_LOW_BORROW_NONE);
}

TEST_F(BorrowPlanTest, ProtectionAndDiagnosticUseScheduledLifetime) {
  const auto pointer = Borrow();
  const uint32_t owner_end = Release(owner_);
  const auto copy = Copy(pointer);
  Release(pointer);
  Release(copy);
  BuildPlan();
  ASSERT_EQ(schedule_.borrows.count, 1u);
  const auto& decision = schedule_.borrows.decisions[0];
  EXPECT_EQ(decision.protection, LOOM_LOW_BORROW_OWNER_RETIRED);
  EXPECT_EQ(decision.reason_node, owner_end);
  EXPECT_EQ(decision.last_use_node, 2u);
  struct Capture {
    // Counts emitted protection remarks.
    uint32_t count = 0;
    // Borrow node expected as the protection's acquisition point.
    const loom_op_t* borrow;
    // Consuming boundary that prevents coverage.
    const loom_op_t* release;
    // Final use in the pointer-copy chain.
    const loom_op_t* last_use;
  } capture = {0, nodes_[0].op, nodes_[owner_end].op, nodes_[2].op};
  const iree_diagnostic_emitter_t emitter = {
      .fn =
          [](void* user_data, const loom_diagnostic_emission_t* emission) {
            auto& capture = *static_cast<Capture*>(user_data);
            ++capture.count;
            EXPECT_EQ(emission->error, LOOM_ERR_DOMINANCE_021);
            EXPECT_EQ(emission->op, capture.borrow);
            EXPECT_EQ(emission->related_op_count, 2u);
            if (emission->related_op_count == 2) {
              EXPECT_EQ(emission->related_ops[0].op, capture.release);
              EXPECT_EQ(emission->related_ops[1].op, capture.last_use);
            }
            return iree_ok_status();
          },
      .user_data = &capture,
  };
  IREE_ASSERT_OK(
      loom_low_schedule_borrow_diagnostics_emit(&schedule_, emitter));
  EXPECT_EQ(capture.count, 1u);
}

TEST_F(BorrowPlanTest, OwningMoveEndsTheOriginalAnchor) {
  const auto pointer = Borrow();
  loom_op_t* move = nullptr;
  IREE_ASSERT_OK(loom_low_move_build(&builder_, owner_, false, owner_type_,
                                     LOOM_LOCATION_UNKNOWN, &move));
  const auto end = Append(move);
  const auto copy = Copy(pointer);
  Release(pointer);
  Release(copy);
  Release(loom_low_move_result(move));
  BuildPlan();
  ASSERT_EQ(schedule_.borrows.count, 1u);
  EXPECT_EQ(schedule_.borrows.decisions[0].protection,
            LOOM_LOW_BORROW_OWNER_RETIRED);
  EXPECT_EQ(schedule_.borrows.decisions[0].reason_node, end);
}

}  // namespace
}  // namespace loom
