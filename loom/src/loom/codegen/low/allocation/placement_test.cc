// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/placement.h"

#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

TEST(LowPlacementTest, DefiningTransferPrecedesEarlierCollectedUses) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  iree_host_size_t vtable_count = 0;
  const auto* vtables = loom_low_dialect_vtables(&vtable_count);
  IREE_ASSERT_OK(loom_context_register_dialect(
      &context, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_low_reg_class_t classes[2] = {};
  loom_low_physical_register_t registers[2] = {};
  const uint16_t candidates[] = {0, 1};
  const uint16_t candidate_ordinals[] = {0, 0};
  const uint16_t atomic_units[] = {0, 1};
  const loom_low_reg_class_alt_t alternatives[] = {
      {0, LOOM_LOW_REGISTER_PART_NONE, LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED,
       0},
      {1, LOOM_LOW_REGISTER_PART_NONE, LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED,
       0},
  };
  loom_low_operand_t operands[2] = {};
  for (uint16_t i = 0; i < 2; ++i) {
    classes[i].target_bank_id = i;
    classes[i].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                       LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
    classes[i].alloc_unit_bits = 32;
    classes[i].allocatable_count = 1;
    classes[i].physical_register_candidate_start = i;
    classes[i].candidate_lookup = {i, i, 1};
    classes[i].physical_atomic_unit_count = 1;
    registers[i].atomic_unit_start = i;
    registers[i].atomic_unit_count = 1;
    operands[i].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
    operands[i].source_value_index = i;
    operands[i].reg_class_alt_start = i;
    operands[i].reg_class_alt_count = 1;
    operands[i].unit_count = 1;
  }
  const loom_low_constraint_t constraint = {
      LOOM_LOW_CONSTRAINT_KIND_SAME_REGISTER_ORDINAL, 0, 1, 0};
  loom_low_descriptor_t descriptor = {};
  descriptor.operand_count = 2;
  descriptor.minimum_packet_operand_count = 2;
  descriptor.constraint_count = 1;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = 1;
  descriptor_set.reg_classes = classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(classes);
  descriptor_set.physical_registers = registers;
  descriptor_set.physical_register_count = IREE_ARRAYSIZE(registers);
  descriptor_set.physical_register_candidate_ids = candidates;
  descriptor_set.physical_register_candidate_count = IREE_ARRAYSIZE(candidates);
  descriptor_set.physical_register_candidate_ordinals = candidate_ordinals;
  descriptor_set.physical_register_atomic_units = atomic_units;
  descriptor_set.physical_register_atomic_unit_count =
      IREE_ARRAYSIZE(atomic_units);
  descriptor_set.reg_class_alts = alternatives;
  descriptor_set.reg_class_alt_count = IREE_ARRAYSIZE(alternatives);
  descriptor_set.operands = operands;
  descriptor_set.operand_count = IREE_ARRAYSIZE(operands);
  descriptor_set.descriptors = &descriptor;
  descriptor_set.descriptor_count = 1;
  descriptor_set.constraints = &constraint;
  descriptor_set.constraint_count = 1;

  const loom_low_placement_value_ref_t values[] = {
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 1}};
  const loom_low_placement_predicate_t predicate = {
      0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 1};
  const loom_low_placement_clause_t clause = {0, 1, 1,
                                              LOOM_LOW_PLACEMENT_CLAUSE_ANY};
  const loom_low_placement_preference_t preference = {values, &predicate,
                                                      &clause, 2, 1};
  const uint16_t preference_indices[] = {1};

  for (bool is_move : {false, true}) {
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("transfer"), &pool,
                                        nullptr, iree_allocator_system(),
                                        &module));
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("transfer"), &name));
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name, &symbol));
    const loom_type_t types[] = {loom_low_register_type(1, 0, 1),
                                 loom_low_register_type(1, 1, 1)};
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, name, {}, 0, {}, {},
        LOOM_STRING_ID_INVALID, {}, loom_symbol_ref_t{0, symbol}, types, 2,
        nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    loom_region_t* body = loom_low_func_def_body(function);
    loom_builder_enter_region(&builder, function, body);
    const loom_value_id_t source = loom_region_entry_arg_id(body, 0);
    const loom_value_id_t peer = loom_region_entry_arg_id(body, 1);

    // Execution is entry -> producer -> consumer. Region layout visits the
    // consumer first, so its input constraint precedes the defining transfer
    // in the collector. Both branch payloads exercise the reverse edge index.
    loom_block_t* consumer = nullptr;
    loom_block_t* producer = nullptr;
    IREE_ASSERT_OK(loom_region_append_block(module, body, &consumer));
    IREE_ASSERT_OK(loom_region_append_block(module, body, &producer));
    loom_value_id_t forwarded = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(module, types[1], &forwarded));
    IREE_ASSERT_OK(loom_block_add_arg(module, producer, forwarded));
    loom_value_id_t received = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(module, types[1], &received));
    IREE_ASSERT_OK(loom_block_add_arg(module, consumer, received));
    loom_op_t* entry_branch = nullptr;
    IREE_ASSERT_OK(loom_low_br_build(&builder, producer, &peer, 1,
                                     LOOM_LOCATION_UNKNOWN, &entry_branch));
    loom_builder_set_block(&builder, producer);
    loom_op_t* transfer = nullptr;
    IREE_ASSERT_OK(is_move
                       ? loom_low_move_build(&builder, source, false, types[0],
                                             LOOM_LOCATION_UNKNOWN, &transfer)
                       : loom_low_copy_build(&builder, source, false, types[0],
                                             LOOM_LOCATION_UNKNOWN, &transfer));
    const loom_value_id_t result = loom_op_results(transfer)[0];
    loom_op_t* producer_branch = nullptr;
    IREE_ASSERT_OK(loom_low_br_build(&builder, consumer, &forwarded, 1,
                                     LOOM_LOCATION_UNKNOWN, &producer_branch));
    loom_builder_set_block(&builder, consumer);
    const loom_value_id_t inputs[] = {result, received};
    loom_op_t* use = nullptr;
    IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
        &builder, &descriptor_set, &descriptor, 0, inputs, 2, {}, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &use));
    loom_op_t* terminator = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));

    loom_local_value_domain_t domain = {};
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module, body, &module->arena, &domain));
    loom_liveness_analysis_t liveness = {};
    IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
        &domain, loom_liveness_order_empty(), &module->arena, &liveness));
    for (iree_host_size_t i = 0; i < liveness.operation_count; ++i) {
      if (loom_liveness_operation_at(&liveness, i)->op == use) {
        break;
      }
      ASSERT_NE(loom_liveness_operation_at(&liveness, i)->op, transfer);
    }
    loom_low_placement_table_t placement = {};
    loom_low_placement_preference_index_t preferences = {};
    loom_low_resolved_target_t target = {};
    target.descriptor_set = &descriptor_set;
    loom_low_allocation_target_constraints_t constraints = {};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
        module, function, &target, nullptr, 0, nullptr, 0, {}, &module->arena,
        &constraints));
    IREE_ASSERT_OK(loom_low_allocation_placement_build(
        &constraints, body, &domain, &liveness, nullptr, 0, {},
        {preference_indices, &preference}, &module->arena, &module->arena,
        &placement, &preferences));
    // Explicit physical IDs are not linear bank coordinates. This target
    // preference is inapplicable even though both inputs are registers.
    EXPECT_EQ(preferences.use_count, 0u);
    EXPECT_EQ(preferences.offsets_by_origin, nullptr);
    const auto ordinal = loom_local_value_domain_try_ordinal(&domain, result);
    const auto range = placement.ranges_by_result_ordinal[ordinal];
    ASSERT_EQ(range.count, 2u);
    const auto* defining =
        loom_low_placement_defining_transfer_for_value_ordinal(&placement,
                                                               ordinal);
    ASSERT_EQ(defining, &placement.relations[range.start]);
    EXPECT_EQ(defining->op, transfer);
    EXPECT_EQ(placement.relations[range.start + 1].op, use);
    for (auto value : {source, peer, forwarded, received}) {
      EXPECT_EQ(
          loom_low_placement_defining_transfer_for_value_ordinal(
              &placement, loom_local_value_domain_try_ordinal(&domain, value)),
          nullptr);
    }
    for (loom_value_ordinal_t i = 0; i < placement.value_count; ++i) {
      const auto outgoing = placement.ranges_by_source_ordinal[i];
      for (uint32_t j = 0; j < outgoing.count; ++j) {
        const auto index =
            placement.relation_indices_by_source_ordinal[outgoing.start + j];
        EXPECT_EQ(placement.relations[index].source_ordinal, i);
      }
    }
    ASSERT_EQ(placement.edge_relation_count, 2u);
    EXPECT_EQ(placement.relations[placement.edge_relation_indices[0]].op,
              entry_branch);
    EXPECT_EQ(placement.relations[placement.edge_relation_indices[1]].op,
              producer_branch);
    loom_local_value_domain_release(&domain);
    loom_module_free(module);
  }
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(LowPlacementTest, RetainsOperandConstraintsAcrossExactTiesOnly) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  iree_host_size_t vtable_count = 0;
  const auto* vtables = loom_low_dialect_vtables(&vtable_count);
  IREE_ASSERT_OK(loom_context_register_dialect(
      &context, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  // The two classes are alternatives of the same instruction field, with
  // different hardware alignment. An earlier tied chain has no requirement
  // of its own; its final consumer determines the whole component's base.
  loom_low_reg_class_t classes[2] = {};
  for (auto& reg_class : classes) {
    reg_class.alloc_unit_bits = 32;
  }
  classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  classes[0].allocatable_count = 7;
  const loom_low_reg_class_alt_t alternatives[] = {
      {0, LOOM_LOW_REGISTER_PART_NONE, LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED,
       0},
      {1, LOOM_LOW_REGISTER_PART_NONE, 0, 0},
      {0, LOOM_LOW_REGISTER_PART_NONE, LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED,
       3},
      {1, LOOM_LOW_REGISTER_PART_NONE, 0, 1},
  };
  loom_low_operand_t operands[3] = {};
  for (auto& operand : operands) {
    operand.role = LOOM_LOW_OPERAND_ROLE_OPERAND;
    operand.reg_class_alt_count = 2;
    operand.unit_count = 1;
  }
  operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  operands[0].address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET;
  operands[0].addressable_unit_count = 16;
  operands[1].address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE;
  operands[1].addressable_unit_count = 16;
  operands[1].address_state_slot = 1;
  operands[2].reg_class_alt_start = 2;
  operands[2].address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET;
  operands[2].addressable_unit_count = 8;
  const loom_low_constraint_t tie_constraint = {LOOM_LOW_CONSTRAINT_KIND_TIED,
                                                0, 1, 0};
  loom_low_descriptor_t descriptors[2] = {};
  descriptors[0].operand_count = 2;
  descriptors[0].result_count = 1;
  descriptors[0].minimum_packet_operand_count = 1;
  descriptors[0].constraint_count = 1;
  descriptors[1].operand_start = 2;
  descriptors[1].operand_count = 1;
  descriptors[1].minimum_packet_operand_count = 1;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = 1;
  descriptor_set.reg_classes = classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(classes);
  descriptor_set.reg_class_alts = alternatives;
  descriptor_set.reg_class_alt_count = IREE_ARRAYSIZE(alternatives);
  descriptor_set.operands = operands;
  descriptor_set.operand_count = IREE_ARRAYSIZE(operands);
  descriptor_set.descriptors = descriptors;
  descriptor_set.descriptor_count = IREE_ARRAYSIZE(descriptors);
  descriptor_set.constraints = &tie_constraint;
  descriptor_set.constraint_count = 1;

  const loom_low_placement_value_ref_t values[] = {
      {0, LOOM_LOW_PLACEMENT_VALUE_RESULT, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0}};
  loom_low_placement_predicate_t predicate = {
      0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 1};
  const loom_low_placement_clause_t clause = {0, 1, 1,
                                              LOOM_LOW_PLACEMENT_CLAUSE_ANY};
  const loom_low_placement_preference_t instruction_preferences[] = {
      {values, &predicate, &clause, 3, 1},
      {values + 1, &predicate, &clause, 2, 1}};
  const uint16_t preference_indices[] = {1, 2};

  struct PreferenceCase {
    // Selected descriptor class: finite physical zero or unbounded virtual one.
    uint16_t class_id;
    // Actual predicate semantics, independent of allocation strategy.
    loom_low_placement_relation_kind_t kind;
    // Original predicate mask, potentially noncontiguous or full-width.
    uint32_t location_mask;
    // Expected carry-closed dependency mask.
    uint32_t dependency_mask;
    // Expected power-of-two domain cap, or zero for direct evaluation.
    uint32_t entry_count;
  };
  const PreferenceCase cases[] = {
      {0, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 1, 1, 2},
      {0, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 5, 7, 4},
      {0, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, UINT32_MAX,
       UINT32_MAX, 4},
      {1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 5, 7, 0},
      {0, LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE, 0, 0, 0},
  };
  for (const auto& test_case : cases) {
    const uint16_t class_id = test_case.class_id;
    predicate.kind = test_case.kind;
    predicate.location_mask = test_case.location_mask;
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("alignment"), &pool,
                                        nullptr, iree_allocator_system(),
                                        &module));
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("alignment"), &name));
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name, &symbol));
    const loom_type_t type = loom_low_register_type(1, class_id, 1);
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, name, {}, 0, {}, {},
        LOOM_STRING_ID_INVALID, {}, loom_symbol_ref_t{0, symbol}, &type, 1,
        nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    loom_region_t* body = loom_low_func_def_body(function);
    loom_builder_enter_region(&builder, function, body);
    const loom_value_id_t source = loom_region_entry_arg_id(body, 0);
    loom_op_t* copy = nullptr;
    IREE_ASSERT_OK(loom_low_copy_build(&builder, source, false, type,
                                       LOOM_LOCATION_UNKNOWN, &copy));
    loom_value_id_t chain[] = {loom_low_copy_result(copy), 0, 0};
    const loom_tied_result_t tie = {0, 0, false};
    for (unsigned i = 1; i < IREE_ARRAYSIZE(chain); ++i) {
      loom_op_t* op = nullptr;
      IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
          &builder, &descriptor_set, &descriptors[0], 0, &chain[i - 1], 1, {},
          &type, 1, &tie, 1, LOOM_LOCATION_UNKNOWN, &op));
      chain[i] = loom_op_results(op)[0];
    }
    loom_op_t* consumer = nullptr;
    IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
        &builder, &descriptor_set, &descriptors[1], 0, &chain[2], 1, {},
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &consumer));
    loom_op_t* terminator = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));
    loom_local_value_domain_t domain = {};
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module, body, &module->arena, &domain));
    loom_liveness_analysis_t liveness = {};
    IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
        &domain, loom_liveness_order_empty(), &module->arena, &liveness));
    loom_low_placement_table_t placement = {};
    loom_low_placement_preference_index_t preferences = {};
    loom_low_resolved_target_t target = {};
    target.descriptor_set = &descriptor_set;
    loom_low_allocation_target_constraints_t constraints = {};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
        module, function, &target, nullptr, 0, nullptr, 0, {}, &module->arena,
        &constraints));
    IREE_ASSERT_OK(loom_low_allocation_placement_build(
        &constraints, body, &domain, &liveness, nullptr, 0, {},
        {preference_indices, instruction_preferences}, &module->arena,
        &module->arena, &placement, &preferences));
    ASSERT_EQ(preferences.use_count, 3u);
    ASSERT_EQ(preferences.binding_count, 8u);
    EXPECT_EQ(preferences.instruction_use_count, 3u);
    EXPECT_EQ(preferences.max_incident_use_count, 0u);
    EXPECT_EQ(preferences.max_incident_binding_count, 0u);
    EXPECT_EQ(preferences.max_memo_entry_count, 0u);
    EXPECT_EQ(preferences.offsets_by_origin, nullptr);
    EXPECT_EQ(preferences.use_indices, nullptr);
    const auto origin = loom_local_value_domain_ordinal(&domain, chain[0]);
    for (uint32_t i = 0; i < preferences.use_count; ++i) {
      const auto& use = preferences.uses[i];
      EXPECT_EQ(use.memo.location_bit_count == 0
                    ? 0
                    : UINT32_MAX >> (32 - use.memo.location_bit_count),
                test_case.dependency_mask);
      if (test_case.entry_count == 0) {
        EXPECT_EQ(use.memo.index_bit_count_plus_one, 0u);
      } else {
        EXPECT_EQ(UINT32_C(1) << (use.memo.index_bit_count_plus_one - 1),
                  test_case.entry_count);
      }
      for (uint16_t j = 0; j < use.preference->value_count; ++j) {
        const auto& binding = preferences.bindings[use.binding_start + j];
        EXPECT_EQ(binding.representative, j);
        EXPECT_EQ(
            placement
                .tied_storage_origins_by_value_ordinal[binding.value_ordinal],
            origin);
      }
    }
    // Deferred instruction bindings keep actual values without an origin CSR.
    EXPECT_EQ(preferences.bindings[0].value_ordinal,
              loom_local_value_domain_ordinal(&domain, chain[1]));
    EXPECT_EQ(preferences.bindings[1].value_ordinal, origin);
    ASSERT_NE(placement.operand_constraints_by_interval, nullptr);
    ASSERT_NE(placement.tied_storage_origins_by_value_ordinal, nullptr);
    const auto chain_origin =
        loom_local_value_domain_try_ordinal(&domain, chain[0]);
    for (auto value : chain) {
      const auto ordinal = loom_local_value_domain_try_ordinal(&domain, value);
      EXPECT_EQ(placement.tied_storage_origins_by_value_ordinal[ordinal],
                chain_origin);
      EXPECT_EQ(placement
                    .operand_constraints_by_interval
                        [liveness.value_interval_indices[ordinal]]
                    .unit_alignment_log2,
                class_id == 0 ? 3 : 1);
      EXPECT_EQ(placement
                    .operand_constraints_by_interval
                        [liveness.value_interval_indices[ordinal]]
                    .addressable_unit_count,
                8u);
      EXPECT_TRUE(placement
                      .operand_constraints_by_interval
                          [liveness.value_interval_indices[ordinal]]
                      .has_target_address_state);
    }
    const auto source_ordinal =
        loom_local_value_domain_try_ordinal(&domain, source);
    EXPECT_EQ(placement.tied_storage_origins_by_value_ordinal[source_ordinal],
              source_ordinal);
    EXPECT_EQ(placement
                  .operand_constraints_by_interval
                      [liveness.value_interval_indices[source_ordinal]]
                  .unit_alignment_log2,
              0);
    EXPECT_EQ(placement
                  .operand_constraints_by_interval
                      [liveness.value_interval_indices[source_ordinal]]
                  .addressable_unit_count,
              0u);
    EXPECT_FALSE(placement
                     .operand_constraints_by_interval
                         [liveness.value_interval_indices[source_ordinal]]
                     .has_target_address_state);
    loom_local_value_domain_release(&domain);
    loom_module_free(module);
  }
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&pool);
}

class LowPlacementStorageTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<LowPlacementStorageTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected placement allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializeStorage(iree_host_size_t block_size = 128 * 1024) {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate},
                                     &result_pool_);
    iree_arena_initialize(&result_pool_, &result_arena_);
  }

  void DeinitializeStorage() {
    iree_arena_deinitialize(&result_arena_);
    iree_arena_block_pool_deinitialize(&result_pool_);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &module_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const auto* vtables = loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    InitializeStorage();
  }

  void TearDown() override {
    DeinitializeStorage();
    loom_local_value_domain_release(&domain_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&module_pool_);
  }

  void BuildChain(uint32_t relation_count, uint32_t first_tie) {
    first_tie_ = first_tie;
    register_class_.alloc_unit_bits = 32;
    for (auto& operand : operands_) {
      operand.reg_class_alt_count = 1;
      operand.unit_count = 1;
    }
    operands_[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
    operands_[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
    descriptor_.operand_count = 2;
    descriptor_.result_count = 1;
    descriptor_.minimum_packet_operand_count = 1;
    descriptor_.constraint_count = 1;
    descriptor_set_.stable_id = 1;
    descriptor_set_.reg_classes = &register_class_;
    descriptor_set_.reg_class_count = 1;
    descriptor_set_.reg_class_alts = &alternative_;
    descriptor_set_.reg_class_alt_count = 1;
    descriptor_set_.operands = operands_;
    descriptor_set_.operand_count = IREE_ARRAYSIZE(operands_);
    descriptor_set_.descriptors = &descriptor_;
    descriptor_set_.descriptor_count = 1;
    descriptor_set_.constraints = &constraint_;
    descriptor_set_.constraint_count = 1;

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("chain"),
                                        &module_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("chain"), &name));
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t type = loom_low_register_type(1, 0, 1);
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, name, {}, 0, {}, {},
        LOOM_STRING_ID_INVALID, {}, loom_symbol_ref_t{0, symbol}, &type, 1,
        &type, 1, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    function_ = function;
    body_ = loom_low_func_def_body(function);
    loom_builder_enter_region(&builder, function, body_);
    values_.push_back(loom_region_entry_arg_id(body_, 0));
    // The last relation is a branch payload, after the copy/tied chain. This
    // exercises both operation-ordered side indexes across chunk boundaries.
    for (uint32_t i = 0; i + 1 < relation_count; ++i) {
      loom_op_t* op = nullptr;
      if (i < first_tie) {
        IREE_ASSERT_OK(loom_low_copy_build(&builder, values_.back(), false,
                                           type, LOOM_LOCATION_UNKNOWN, &op));
      } else {
        const loom_tied_result_t tie = {0, 0, false};
        IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
            &builder, &descriptor_set_, &descriptor_, 0, &values_.back(), 1, {},
            &type, 1, &tie, 1, LOOM_LOCATION_UNKNOWN, &op));
      }
      operations_.push_back(op);
      values_.push_back(loom_op_results(op)[0]);
    }
    if (relation_count != 0) {
      loom_block_t* successor = nullptr;
      IREE_ASSERT_OK(loom_region_append_block(module_, body_, &successor));
      loom_value_id_t received = LOOM_VALUE_ID_INVALID;
      IREE_ASSERT_OK(loom_module_define_value(module_, type, &received));
      IREE_ASSERT_OK(loom_block_add_arg(module_, successor, received));
      loom_op_t* branch = nullptr;
      IREE_ASSERT_OK(loom_low_br_build(&builder, successor, &values_.back(), 1,
                                       LOOM_LOCATION_UNKNOWN, &branch));
      operations_.push_back(branch);
      values_.push_back(received);
      loom_builder_set_block(&builder, successor);
    }
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder, &values_.back(), 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, body_, &module_->arena, &domain_));
    IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
        &domain_, loom_liveness_order_empty(), &module_->arena, &liveness_));
  }

  iree_status_t Analyze(loom_low_placement_table_t* out_table) {
    *out_table = {};
    loom_low_placement_preference_index_t preferences = {};
    loom_low_resolved_target_t target = {};
    target.descriptor_set = &descriptor_set_;
    loom_low_allocation_target_constraints_t constraints = {};
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_initialize(
        module_, function_, &target, nullptr, 0, nullptr, 0, {}, &result_arena_,
        &constraints));
    return loom_low_allocation_placement_build(
        &constraints, body_, &domain_, &liveness_, nullptr, 0, {}, {},
        &result_arena_, &result_arena_, out_table, &preferences);
  }

  void ExpectRelations(const loom_low_placement_table_t& table) {
    ASSERT_EQ(table.relation_count, operations_.size());
    ASSERT_EQ(table.value_count, values_.size());
    const size_t operation_count =
        operations_.empty() ? 0 : operations_.size() - 1;
    const size_t write_count =
        first_tie_ < operation_count ? operation_count - first_tie_ : 0;
    ASSERT_EQ(table.storage.write_relation_count, write_count);
    ASSERT_EQ(table.edge_relation_count, operations_.empty() ? 0u : 1u);
    ASSERT_EQ(table.tied_storage_origins_by_value_ordinal == nullptr,
              write_count == 0);
    for (size_t i = 0; i < values_.size(); ++i) {
      const auto ordinal =
          loom_local_value_domain_ordinal(&domain_, values_[i]);
      const auto outgoing = table.ranges_by_source_ordinal[ordinal];
      ASSERT_EQ(outgoing.count, i < operations_.size() ? 1u : 0u);
      if (i < operations_.size()) {
        const auto index =
            table.relation_indices_by_source_ordinal[outgoing.start];
        EXPECT_EQ(table.relations[index].op, operations_[i]);
      }
      if (write_count != 0) {
        const auto origin =
            i > first_tie_ && i <= operation_count
                ? loom_local_value_domain_ordinal(&domain_, values_[first_tie_])
                : ordinal;
        EXPECT_EQ(table.tied_storage_origins_by_value_ordinal[ordinal], origin);
      }
      const auto range = table.ranges_by_result_ordinal[ordinal];
      ASSERT_EQ(range.count, i == 0 ? 0u : 1u);
      if (i == 0) {
        continue;
      }
      const auto& relation = table.relations[range.start];
      const bool is_edge = i == operations_.size();
      const bool is_tied = !is_edge && i - 1 >= first_tie_;
      if (is_tied) {
        EXPECT_EQ(
            loom_low_placement_tied_source_for_value_ordinal(&table, ordinal),
            loom_local_value_domain_ordinal(&domain_, values_[i - 1]));
      }
      EXPECT_EQ(relation.op, operations_[i - 1]);
      EXPECT_EQ(relation.result_ordinal, ordinal);
      EXPECT_EQ(relation.source_ordinal,
                loom_local_value_domain_ordinal(&domain_, values_[i - 1]));
      EXPECT_EQ(relation.result_unit_offset, 0u);
      EXPECT_EQ(relation.source_unit_offset, 0u);
      EXPECT_EQ(relation.unit_count, 1u);
      EXPECT_EQ(relation.kind, LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE);
      EXPECT_EQ(relation.cause, is_edge   ? LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH
                                : is_tied ? LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT
                                          : LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY);
      EXPECT_EQ(
          relation.flags,
          LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
              (is_tied ? LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                             LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE
                       : LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED));
      EXPECT_EQ(relation.write_point,
                loom_liveness_operation_at(&liveness_, i - 1)->end_point);
      EXPECT_EQ(relation.priority, 1u);
      EXPECT_EQ(relation.source_operand_index, 0u);
      if (is_tied) {
        EXPECT_EQ(table.storage.write_relation_indices[i - 1 - first_tie_],
                  range.start);
      } else if (is_edge) {
        EXPECT_EQ(table.edge_relation_indices[0], range.start);
      }
    }
  }

  // IR and prerequisite analyses outlive all placement result lifetimes.
  iree_arena_block_pool_t module_pool_ = {};
  // Registered Low operation semantics.
  loom_context_t context_ = {};
  // Owned module containing the chain.
  loom_module_t* module_ = nullptr;
  // Function owning the analyzed region and allocation diagnostics.
  loom_op_t* function_ = nullptr;
  // Analyzed function body.
  loom_region_t* body_ = nullptr;
  // Acquired ordinals shared by liveness and placement.
  loom_local_value_domain_t domain_ = {};
  // Real producer result, allocated outside observed placement storage.
  loom_liveness_analysis_t liveness_ = {};
  // Unbounded virtual class used by the descriptor-backed tied operations.
  loom_low_reg_class_t register_class_ = {};
  // Single matching class alternative for both packet fields.
  loom_low_reg_class_alt_t alternative_ = {
      0, LOOM_LOW_REGISTER_PART_NONE, LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED, 0};
  // Result and operand packet fields.
  loom_low_operand_t operands_[2] = {};
  // Verified whole-value result/operand tie.
  loom_low_constraint_t constraint_ = {LOOM_LOW_CONSTRAINT_KIND_TIED, 0, 1, 0};
  // One-result, one-operand destructive packet.
  loom_low_descriptor_t descriptor_ = {};
  // Borrowed target descriptors live throughout the fixture.
  loom_low_descriptor_set_t descriptor_set_ = {};
  // Collected operation sequence excluding the final return.
  std::vector<const loom_op_t*> operations_;
  // Entry argument, packet results and final branch destination.
  std::vector<loom_value_id_t> values_;
  // First destructive packet after the optional copy prefix.
  uint32_t first_tie_ = UINT32_MAX;
  // Observed pool shared by placement results and internal scratch.
  iree_arena_block_pool_t result_pool_ = {};
  // Result lifetime ends between repeated analyses.
  iree_arena_allocator_t result_arena_ = {};
  // Attempted backing allocations, excluding frees.
  iree_host_size_t allocation_count_ = 0;
  // Selected allocation failure, or SIZE_MAX for normal execution.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

class LowPlacementStorageBoundaryTest
    : public LowPlacementStorageTest,
      public ::testing::WithParamInterface<std::pair<uint32_t, uint32_t>> {};

TEST_P(LowPlacementStorageBoundaryTest, PreservesRelationsAndReusesPoolBlocks) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(GetParam().first, GetParam().second));
  loom_low_placement_table_t table = {};
  IREE_ASSERT_OK(Analyze(&table));
  ASSERT_NO_FATAL_FAILURE(ExpectRelations(table));
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&result_pool_, &statistics);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  const auto allocation_count = allocation_count_;
  iree_arena_reset(&result_arena_);
  IREE_ASSERT_OK(Analyze(&table));
  ASSERT_NO_FATAL_FAILURE(ExpectRelations(table));
  EXPECT_EQ(allocation_count_, allocation_count);
}

INSTANTIATE_TEST_SUITE_P(
    RelationChunks, LowPlacementStorageBoundaryTest,
    ::testing::ValuesIn(std::vector<std::pair<uint32_t, uint32_t>>{
        {0, UINT32_MAX},
        {1, UINT32_MAX},
        {63, UINT32_MAX},
        {64, UINT32_MAX},
        {65, UINT32_MAX},
        {127, UINT32_MAX},
        {128, UINT32_MAX},
        {129, UINT32_MAX},
        {2048, UINT32_MAX},
        {2049, UINT32_MAX},
        {64, 0},
        {65, 0},
        {129, 63},
        {129, 64},
        {129, 65},
        {2049, 64}}));

TEST_F(LowPlacementStorageTest, RelationChunksFitSmallPoolBlocks) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(65, 0));
  DeinitializeStorage();
  InitializeStorage(4096);
  loom_low_placement_table_t table = {};
  IREE_ASSERT_OK(Analyze(&table));
  ASSERT_NO_FATAL_FAILURE(ExpectRelations(table));
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&result_pool_, &statistics);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
}

TEST_F(LowPlacementStorageTest, BackingFailureCanBeResetAndRetried) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(2049, 64));
  loom_low_placement_table_t table = {};
  IREE_ASSERT_OK(Analyze(&table));
  ASSERT_NO_FATAL_FAILURE(ExpectRelations(table));
  const auto allocation_count = allocation_count_;
  ASSERT_GT(allocation_count, 1u);
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    SCOPED_TRACE(i);
    DeinitializeStorage();
    InitializeStorage();
    failure_index_ = i;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Analyze(&table));
    EXPECT_EQ(allocation_count_, i + 1);
    EXPECT_EQ(table.relations, nullptr);
    EXPECT_EQ(table.relation_count, 0u);
    EXPECT_EQ(table.tied_storage_origins_by_value_ordinal, nullptr);
    failure_index_ = SIZE_MAX;
    iree_arena_reset(&result_arena_);
    IREE_ASSERT_OK(Analyze(&table));
    ASSERT_NO_FATAL_FAILURE(ExpectRelations(table));
  }
}

}  // namespace
}  // namespace loom
