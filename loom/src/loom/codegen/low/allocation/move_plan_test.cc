// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/target_records.h"

namespace loom {
namespace {

class LowAllocationMovePlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    const auto* target_bundle = loom_target_bundle_table_lookup(
        &loom_test_target_bundles, LOOM_TEST_TARGET_KIND_LOW_CORE);
    ASSERT_NE(target_bundle, nullptr);
    loom_target_facts_builder_initialize(&loom_test_target_fact_type,
                                         target_bundle, &target_facts_);
    target_ = {
        .target_facts = &target_facts_,
        .target_name = target_bundle->name,
        .descriptor_set_key = target_bundle->config->contract_set.spelling,
        .feature_bits = target_bundle->config->contract_feature_bits,
        .descriptor_set = loom_test_low_core_descriptor_set(),
    };
    ASSERT_NE(target_.descriptor_set, nullptr);

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("moves"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &diagnostic_module_));
    loom_builder_t builder;
    loom_builder_initialize(diagnostic_module_, &diagnostic_module_->arena,
                            loom_module_block(diagnostic_module_), &builder);
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("owner"), &name));
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(diagnostic_module_, name, &symbol));
    loom_string_id_t descriptor_set;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder, target_.descriptor_set_key, &descriptor_set));
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, 0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*effect_contract=*/0, /*temperature=*/0, /*inline_policy=*/0,
        /*allocation=*/0,
        /*schedule=*/0, descriptor_set, /*target=*/{}, /*abi=*/0, {}, {},
        LOOM_STRING_ID_INVALID, {}, loom_symbol_ref_t{0, symbol},
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &diagnostic_function_));
    loom_builder_enter_region(&builder, diagnostic_function_,
                              loom_low_func_def_body(diagnostic_function_));
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&arena_);
    loom_module_free(diagnostic_module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Shared blocks for retained output, transient workspace, and IR ownership.
  iree_arena_block_pool_t pool_ = {};
  // Retains move rows and constraints beyond scratch release.
  iree_arena_allocator_t arena_ = {};
  // Transient source indexes and sequencing workspace.
  iree_arena_allocator_t scratch_ = {};
  // Registered Low operations for the owned function.
  loom_context_t context_ = {};
  // Target facts used by constraint resolution.
  loom_target_facts_t target_facts_ = {};
  // Resolved descriptor contract shared by native move groups.
  loom_low_resolved_target_t target_ = {};
  // Owns the empty function anchoring native move diagnostics.
  loom_module_t* diagnostic_module_ = nullptr;
  // Diagnostic owner only; not a program manufacturing solver inputs.
  loom_op_t* diagnostic_function_ = nullptr;
};

TEST_F(LowAllocationMovePlanTest, AppendsOnlyDemandedAssignmentUnits) {
  struct Transport {
    // Descriptor class supplying the unit-to-location mapping.
    iree_string_view_t register_class;
    // Coordinate space for both endpoints.
    loom_low_allocation_location_kind_t location_kind;
    // Source assignment base or aggregate view.
    uint32_t source_base;
    // Destination assignment base or aggregate view.
    uint32_t destination_base;
    // Concrete source locations in logical unit order.
    uint32_t source_units[2];
    // Concrete destination locations in logical unit order.
    uint32_t destination_units[2];
  };
  const Transport transports[] = {
      {IREE_SV("test.i32"),
       LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
       2,
       0,
       {2, 3},
       {0, 1}},
      {IREE_SV("test.phys"),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       2,
       0,
       {2, 3},
       {0, 1}},
      {IREE_SV("test.explicit32"),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       4,
       5,
       {0, 2},
       {1, 3}},
  };
  const auto* descriptors = target_.descriptor_set;
  uint32_t starts[] = {2, 2};
  uint32_t ends[] = {6, 6};
  const loom_low_allocation_unit_liveness_t liveness = {
      .start_points = starts,
      .end_points = ends,
      .point_count = IREE_ARRAYSIZE(ends),
  };
  const loom_low_allocation_move_plan_context_t context = {
      .descriptor_set = descriptors,
      .unit_liveness = &liveness,
  };
  loom_low_allocation_move_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_allocation_move_plan_initialize(
      &context, /*move_input_capacity=*/2, /*raw_group_capacity=*/2, &arena_,
      &scratch_, &plan));
  for (const auto& transport : transports) {
    uint16_t register_class_id = LOOM_LOW_REG_CLASS_NONE;
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptors, transport.register_class, &register_class_id, nullptr));
    SCOPED_TRACE(register_class_id);
    loom_low_allocation_assignment_t source = {
        .descriptor_reg_class_id = register_class_id,
        .flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS,
        .start_point = 2,
        .end_point = 6,
        .unit_count = 2,
        .location_kind = transport.location_kind,
        .location_base = transport.source_base,
        .location_count = 2,
    };
    auto destination = source;
    destination.location_base = transport.destination_base;
    for (uint32_t demanded_units = 1; demanded_units < 4; ++demanded_units) {
      SCOPED_TRACE(demanded_units);
      for (uint32_t unit = 0; unit < 2; ++unit) {
        ends[unit] = demanded_units & (1u << unit) ? 6 : starts[unit];
      }
      // Whole assignments and individual subranges share the same demand.
      for (uint32_t first = 0; first < 2; ++first) {
        for (uint32_t count = 1; count <= 2 - first; ++count) {
          SCOPED_TRACE(first);
          SCOPED_TRACE(count);
          iree_host_size_t move_count = 0;
          loom_low_allocation_move_plan_append_assignment(
              &plan, &source, first, &destination, first, count, &move_count);
          iree_host_size_t expected_count = 0;
          for (uint32_t unit = first; unit < first + count; ++unit) {
            if (!(demanded_units & (1u << unit))) {
              continue;
            }
            ASSERT_LT(expected_count, move_count);
            const auto& move = plan.sequence_scratch.moves[expected_count++];
            EXPECT_EQ(move.source.location_kind, transport.location_kind);
            EXPECT_EQ(move.source.descriptor_reg_class_id, register_class_id);
            EXPECT_EQ(move.source.location, transport.source_units[unit]);
            EXPECT_EQ(move.destination.location_kind, transport.location_kind);
            EXPECT_EQ(move.destination.descriptor_reg_class_id,
                      register_class_id);
            EXPECT_EQ(move.destination.location,
                      transport.destination_units[unit]);
          }
          EXPECT_EQ(move_count, expected_count);
        }
      }
    }
  }
}

TEST_F(LowAllocationMovePlanTest, CycleTemporariesRespectStorageSupport) {
  struct StorageClass {
    // Descriptor class defining the transfer width and spill policy.
    iree_string_view_t name;
    // Coordinate space of the two input locations.
    loom_low_allocation_location_kind_t location_kind;
    // Memory space allowed by this class when registers are saturated.
    loom_storage_space_t space;
    // Bytes and power-of-two alignment of one saved unit for these classes.
    uint32_t byte_length;
    // Whether the class permits saving a unit in memory.
    bool spillable;
  };
  const StorageClass classes[] = {
      {IREE_SV("test.i8"), LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
       LOOM_STORAGE_SPACE_PRIVATE, 1, true},
      {IREE_SV("test.f32"), LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
       LOOM_STORAGE_SPACE_PRIVATE, 4, true},
      {IREE_SV("test.i64"), LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
       LOOM_STORAGE_SPACE_PRIVATE, 8, true},
      {IREE_SV("test.phys"), LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_STORAGE_SPACE_STACK, 64, true},
      {IREE_SV("test.pressure.alias64"),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_STORAGE_SPACE_STACK,
       8, true},
      {IREE_SV("test.spillable.explicit32"),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_STORAGE_SPACE_STACK,
       4, true},
      {IREE_SV("test.explicit32"),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_STORAGE_SPACE_PRIVATE, 4, false},
  };
  for (const auto& storage_class : classes) {
    SCOPED_TRACE(storage_class.name.data);
    uint16_t register_class = LOOM_LOW_REG_CLASS_NONE;
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        target_.descriptor_set, storage_class.name, &register_class, nullptr));
    for (uint32_t capacity : {2u, 3u}) {
      SCOPED_TRACE(capacity);
      for (auto spaces :
           {LOOM_LOW_STORAGE_SPACE_SET_NONE, LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
            LOOM_LOW_STORAGE_SPACE_SET_STACK}) {
        SCOPED_TRACE(spaces);
        const bool has_register = capacity == 3;
        const bool has_storage =
            storage_class.spillable &&
            loom_low_storage_space_set_contains(spaces, storage_class.space);
        const loom_low_allocation_budget_t budget = {storage_class.name,
                                                     capacity};
        loom_low_allocation_target_constraints_t constraints = {};
        IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
            diagnostic_module_, diagnostic_function_, &target_, &budget, 1,
            nullptr, 0, {}, &arena_, &constraints));
        // Only the transfer locations are occupied; no other values are live.
        const loom_low_allocation_unit_liveness_t unit_liveness = {};
        const loom_low_allocation_move_plan_context_t context = {
            .descriptor_set = target_.descriptor_set,
            .target_constraints = &constraints,
            .unit_liveness = &unit_liveness,
            .storage_spaces = spaces,
        };
        const auto checkpoint = iree_arena_checkpoint_save(&scratch_);
        loom_low_allocation_move_plan_t plan = {};
        IREE_ASSERT_OK(loom_low_allocation_move_plan_initialize(
            &context, /*move_input_capacity=*/4, /*raw_group_capacity=*/2,
            &arena_, &scratch_, &plan));
        const loom_low_move_location_t lhs = {
            static_cast<uint16_t>(storage_class.location_kind), register_class,
            0};
        const loom_low_move_location_t rhs = {
            static_cast<uint16_t>(storage_class.location_kind), register_class,
            1};
        loom_low_move_group_t groups[2] = {};
        for (auto& group : groups) {
          plan.sequence_scratch.moves[0] = {.destination = lhs, .source = rhs};
          plan.sequence_scratch.moves[1] = {.destination = rhs, .source = lhs};
          IREE_ASSERT_OK(loom_low_allocation_move_plan_append_group(
              &plan, diagnostic_function_, /*read_point=*/0,
              /*write_point=*/0, /*raw_move_count=*/2, &group, nullptr));
          if (!has_register && !has_storage) {
            ASSERT_EQ(constraints.error_count, 1u);
            EXPECT_TRUE(iree_string_view_equal(
                constraints.failure.failure_code,
                IREE_SV("parallel-move-no-scratch-unit")));
            EXPECT_EQ(constraints.failure.op, diagnostic_function_);
            EXPECT_EQ(constraints.failure.descriptor_reg_class_id,
                      register_class);
            EXPECT_EQ(constraints.failure.budget_units, capacity);
            EXPECT_EQ(constraints.failure.required_unit_count, 1u);
            EXPECT_EQ(group.moves.count, 0u);
            EXPECT_EQ(plan.move_count, 0u);
            EXPECT_EQ(plan.storage_count, 0u);
            EXPECT_EQ(plan.scratch_move_index_count, 0u);
            break;
          }
          ASSERT_EQ(constraints.error_count, 0u);
        }
        iree_arena_checkpoint_restore(&checkpoint);
        iree_arena_block_pool_trim(&pool_);
        if (!has_register && !has_storage) {
          continue;
        }
        ASSERT_EQ(plan.move_count, 6u);
        ASSERT_EQ(plan.scratch_move_index_count, 2u);
        EXPECT_EQ(groups[0].moves.start, 0u);
        EXPECT_EQ(groups[1].moves.start, 3u);
        EXPECT_EQ(groups[0].scratch_move_index_start, 0u);
        EXPECT_EQ(groups[1].scratch_move_index_start, 1u);
        ASSERT_EQ(plan.storage_count, has_register ? 0u : 1u);
        if (!has_register) {
          EXPECT_EQ(plan.storage[0].register_class, register_class);
          EXPECT_EQ(plan.storage[0].space, storage_class.space);
          EXPECT_EQ(plan.storage[0].byte_length, storage_class.byte_length);
          EXPECT_EQ(plan.storage[0].byte_alignment, storage_class.byte_length);
        }
        for (const auto& group : groups) {
          ASSERT_EQ(group.moves.count, 3u);
          ASSERT_EQ(group.scratch_move_index_count, 1u);
          EXPECT_EQ(plan.scratch_move_indices[group.scratch_move_index_start],
                    group.moves.start);
          const auto* moves = plan.moves + group.moves.start;
          EXPECT_EQ(moves[0].destination.location_kind,
                    has_register ? storage_class.location_kind
                                 : LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE);
          EXPECT_EQ(moves[0].destination.location, has_register ? 2u : 0u);
          uint32_t registers[] = {23, 17, 0};
          uint32_t cell = 0;
          for (iree_host_size_t i = 0; i < group.moves.count; ++i) {
            EXPECT_EQ(moves[i].kind, LOOM_LOW_MOVE_KIND_COPY);
            EXPECT_EQ(moves[i].source.descriptor_reg_class_id, register_class);
            EXPECT_EQ(moves[i].destination.descriptor_reg_class_id,
                      register_class);
            const auto& source = moves[i].source;
            const auto& destination = moves[i].destination;
            ASSERT_LT(source.location, IREE_ARRAYSIZE(registers));
            ASSERT_LT(destination.location, IREE_ARRAYSIZE(registers));
            if (source.location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE) {
              EXPECT_EQ(source.location, 0u);
            } else {
              EXPECT_EQ(source.location_kind, storage_class.location_kind);
            }
            const uint32_t value =
                source.location_kind ==
                        LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE
                    ? cell
                    : registers[source.location];
            if (destination.location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE) {
              EXPECT_EQ(destination.location, 0u);
              cell = value;
            } else {
              EXPECT_EQ(destination.location_kind, storage_class.location_kind);
              registers[destination.location] = value;
            }
          }
          EXPECT_EQ(registers[0], 17u);
          EXPECT_EQ(registers[1], 23u);
        }
      }
    }
  }
}

TEST_F(LowAllocationMovePlanTest, CycleScratchHonorsSparseAndReservedStorage) {
  for (const char* class_name : {"test.f32", "test.phys", "test.explicit32"}) {
    SCOPED_TRACE(class_name);
    const auto name = iree_make_cstring_view(class_name);
    uint16_t register_class = LOOM_LOW_REG_CLASS_NONE;
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        target_.descriptor_set, name, &register_class, nullptr));
    const auto location_kind =
        iree_string_view_equal(name, IREE_SV("test.f32"))
            ? LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID
            : LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    const loom_low_allocation_budget_t budget = {name, 3};
    const loom_liveness_segment_t segments[] = {{0, 2}, {4, 6}};
    uint32_t start = 0;
    uint32_t end = 6;
    loom_low_allocation_unit_liveness_t unit_liveness = {
        .start_points = &start,
        .end_points = &end,
        .point_count = 1,
    };
    unit_liveness.storage_segments.entries = segments;
    const loom_low_allocation_assignment_t assignment = {
        .descriptor_reg_class_id = register_class,
        .start_point = 0,
        .end_point = 6,
        .unit_count = 1,
        .location_kind = location_kind,
        .location_base = 2,
        .location_count = 1,
        .unit_point_start = 0,
        .liveness_segments = {0, IREE_ARRAYSIZE(segments)},
    };
    const loom_low_allocation_reserved_range_t reserved = {name, location_kind,
                                                           2, 1};
    for (uint32_t point : {1u, 2u, 4u, 6u}) {
      SCOPED_TRACE(point);
      for (bool reserve : {false, true}) {
        SCOPED_TRACE(reserve);
        loom_low_allocation_target_constraints_t constraints = {};
        IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
            diagnostic_module_, diagnostic_function_, &target_, &budget, 1,
            reserve ? &reserved : nullptr, reserve ? 1 : 0, {}, &arena_,
            &constraints));
        const loom_low_allocation_move_plan_context_t context = {
            .descriptor_set = target_.descriptor_set,
            .target_constraints = &constraints,
            .unit_liveness = &unit_liveness,
            .assignment_map =
                {
                    // A reserved register cannot also hold an allocated value.
                    .assignments = reserve ? nullptr : &assignment,
                    .assignment_count = reserve ? 0u : 1u,
                },
        };
        loom_low_allocation_move_plan_t plan = {};
        IREE_ASSERT_OK(loom_low_allocation_move_plan_initialize(
            &context, /*move_input_capacity=*/2, /*raw_group_capacity=*/2,
            &arena_, &scratch_, &plan));
        const loom_low_move_location_t lhs = {
            static_cast<uint16_t>(location_kind), register_class, 0};
        const loom_low_move_location_t rhs = {
            static_cast<uint16_t>(location_kind), register_class, 1};
        plan.sequence_scratch.moves[0] = {.destination = lhs, .source = rhs};
        plan.sequence_scratch.moves[1] = {.destination = rhs, .source = lhs};
        loom_low_move_group_t group = {};
        IREE_ASSERT_OK(loom_low_allocation_move_plan_append_group(
            &plan, diagnostic_function_, point, point, /*raw_move_count=*/2,
            &group, nullptr));
        if (reserve || point == 1 || point == 4) {
          EXPECT_EQ(constraints.error_count, 1u);
          EXPECT_TRUE(
              iree_string_view_equal(constraints.failure.failure_code,
                                     IREE_SV("parallel-move-no-scratch-unit")));
          EXPECT_EQ(group.moves.count, 0u);
        } else {
          EXPECT_EQ(constraints.error_count, 0u);
          ASSERT_EQ(group.moves.count, 3u);
          EXPECT_EQ(plan.moves[group.moves.start].destination.location, 2u);
        }
        EXPECT_EQ(plan.storage_count, 0u);
      }
    }
  }
}

}  // namespace
}  // namespace loom
