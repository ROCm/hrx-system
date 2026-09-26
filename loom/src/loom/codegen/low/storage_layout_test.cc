// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_layout.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

TEST(LowStorageSpaceSetTest, NamesUseStableDeclarationOrder) {
  iree_string_view_t names[LOOM_STORAGE_SPACE_COUNT_];
  const iree_host_size_t count = loom_low_storage_space_set_names(
      LOOM_LOW_STORAGE_SPACE_SET_WORKGROUP | LOOM_LOW_STORAGE_SPACE_SET_STACK |
          LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
      IREE_ARRAYSIZE(names), names);

  ASSERT_EQ(count, 3u);
  EXPECT_TRUE(iree_string_view_equal(names[0], IREE_SV("stack")));
  EXPECT_TRUE(iree_string_view_equal(names[1], IREE_SV("private")));
  EXPECT_TRUE(iree_string_view_equal(names[2], IREE_SV("workgroup")));
}

class LowStorageLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &layout_arena_);
    loom_low_storage_layout_builder_initialize(nullptr, &layout_builder_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    BuildLowFunction();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&layout_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return loom_symbol_ref_t{
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  void BuildLowFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    const loom_symbol_ref_t target_ref = AddSymbol(IREE_SV("target"));
    const loom_symbol_ref_t callee_ref = AddSymbol(IREE_SV("storage_layout"));
    loom_string_id_t representation_contract = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &module_builder, IREE_SV("test.low.core"), &representation_contract));
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &module_builder, LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET,
        /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0, /*inline_policy=*/0,
        /*allocation=*/0, /*schedule=*/0,
        /*descriptor_set=*/representation_contract, target_ref, /*abi=*/0,
        loom_named_attr_slice_t{}, loom_named_attr_slice_t{},
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_t{}, callee_ref,
        /*arg_types=*/nullptr,
        /*arg_types_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_op));
    function_op_ = function_op;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_low_func_def_body(function_op_)),
        &body_builder_);
    loom_builder_enter_region(&body_builder_, function_op_,
                              loom_low_func_def_body(function_op_));
  }

  loom_value_id_t Reserve(loom_storage_space_t space, int64_t byte_length,
                          int64_t byte_alignment) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_storage_reserve_build(
        &body_builder_, byte_length, byte_alignment, loom_type_storage(space),
        LOOM_LOCATION_UNKNOWN, &op));
    IREE_CHECK_OK(loom_low_storage_layout_builder_append(
        module_, op, &layout_arena_, &layout_builder_));
    return loom_low_storage_reserve_storage(op);
  }

  loom_value_id_t View(loom_value_id_t source, int64_t byte_offset,
                       int64_t byte_length) {
    loom_op_t* op = nullptr;
    const loom_type_t result_type = loom_module_value_type(module_, source);
    IREE_CHECK_OK(loom_low_storage_view_build(
        &body_builder_, source, byte_offset, byte_length, result_type,
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_low_storage_view_result(op);
  }

  void ExpectReservation(const loom_low_storage_layout_t& layout,
                         loom_value_id_t storage_value_id,
                         loom_storage_space_t expected_space,
                         uint64_t expected_byte_offset,
                         uint64_t expected_byte_size,
                         uint64_t expected_byte_alignment) {
    loom_low_storage_layout_reservation_t reservation = {};
    loom_low_storage_layout_lookup_reservation(&layout, storage_value_id,
                                               &reservation);
    EXPECT_EQ(reservation.space, expected_space);
    EXPECT_EQ(reservation.byte_offset, expected_byte_offset);
    EXPECT_EQ(reservation.byte_size, expected_byte_size);
    EXPECT_EQ(reservation.byte_alignment, expected_byte_alignment);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t layout_arena_;
  loom_low_storage_layout_builder_t layout_builder_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* function_op_ = nullptr;
  loom_builder_t body_builder_;
};

TEST_F(LowStorageLayoutTest, PacksReservationsByStorageSpace) {
  const loom_value_id_t stack0 = Reserve(LOOM_STORAGE_SPACE_STACK, 8, 4);
  const loom_value_id_t private0 = Reserve(LOOM_STORAGE_SPACE_PRIVATE, 4, 4);
  const loom_value_id_t scratch0 = Reserve(LOOM_STORAGE_SPACE_SCRATCH, 8, 8);
  const loom_value_id_t stack1 = Reserve(LOOM_STORAGE_SPACE_STACK, 16, 16);
  const loom_value_id_t workgroup0 =
      Reserve(LOOM_STORAGE_SPACE_WORKGROUP, 3, 2);

  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &layout));
  EXPECT_EQ(layout.space_sizes.stack_bytes, 32u);
  EXPECT_EQ(layout.space_sizes.private_bytes, 4u);
  EXPECT_EQ(layout.space_sizes.scratch_bytes, 8u);
  EXPECT_EQ(layout.space_sizes.workgroup_bytes, 3u);
  EXPECT_EQ(layout.record_count, 5u);
  ExpectReservation(layout, stack0, LOOM_STORAGE_SPACE_STACK, 0, 8, 4);
  ExpectReservation(layout, private0, LOOM_STORAGE_SPACE_PRIVATE, 0, 4, 4);
  ExpectReservation(layout, scratch0, LOOM_STORAGE_SPACE_SCRATCH, 0, 8, 8);
  ExpectReservation(layout, stack1, LOOM_STORAGE_SPACE_STACK, 16, 16, 16);
  ExpectReservation(layout, workgroup0, LOOM_STORAGE_SPACE_WORKGROUP, 0, 3, 2);
}

TEST_F(LowStorageLayoutTest, ResolvesNestedStorageViews) {
  const loom_value_id_t stack = Reserve(LOOM_STORAGE_SPACE_STACK, 16, 8);
  const loom_value_id_t view = View(stack, 4, 8);
  const loom_value_id_t nested_view = View(view, 2, 4);

  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &layout));
  loom_low_storage_layout_reference_t reference = {};
  loom_low_storage_layout_lookup_reference(&layout, module_, nested_view,
                                           &reference);
  EXPECT_EQ(reference.reservation.space, LOOM_STORAGE_SPACE_STACK);
  EXPECT_EQ(reference.reservation.byte_offset, 0u);
  EXPECT_EQ(reference.reservation.byte_size, 16u);
  EXPECT_EQ(reference.byte_offset, 6u);
  EXPECT_EQ(reference.byte_length, 4u);
}

TEST_F(LowStorageLayoutTest,
       HoistsReservationsPreservingAbiPreambleLayoutAndViews) {
  loom_region_t* body = loom_low_func_def_body(function_op_);
  loom_block_t* entry = loom_region_entry_block(body);
  const loom_type_t pointer_type =
      loom_low_register_type(loom_test_low_core_descriptor_set()->stable_id,
                             TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR, 1);
  loom_string_id_t live_in_source = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(
      &body_builder_, IREE_SV("test.arg0"), &live_in_source));
  loom_op_t* live_in = nullptr;
  IREE_ASSERT_OK(loom_low_live_in_build(&body_builder_, 0, live_in_source,
                                        loom_named_attr_slice_t{}, pointer_type,
                                        LOOM_LOCATION_UNKNOWN, &live_in));
  loom_type_id_t buffer_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module_, loom_type_buffer(), &buffer_type_id));
  loom_op_t* resource = nullptr;
  IREE_ASSERT_OK(loom_low_resource_build(
      &body_builder_, 0, LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER,
      LOOM_VALUE_ID_INVALID, /*index=*/0, buffer_type_id, /*extent=*/0,
      /*cache_swizzle_stride=*/0, pointer_type, LOOM_LOCATION_UNKNOWN,
      &resource));
  const loom_value_id_t stack = Reserve(LOOM_STORAGE_SPACE_STACK, 16, 8);
  const loom_value_id_t entry_view = View(stack, 4, 8);
  const loom_value_id_t private_storage =
      Reserve(LOOM_STORAGE_SPACE_PRIVATE, 8, 8);
  loom_block_t* tail = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &tail));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_low_br_build(&body_builder_, tail, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&body_builder_, tail);
  const loom_value_id_t scratch = Reserve(LOOM_STORAGE_SPACE_SCRATCH, 32, 16);
  const loom_value_id_t outer = View(scratch, 8, 16);
  const loom_value_id_t nested = View(outer, 4, 8);
  const loom_value_id_t aligned_stack =
      Reserve(LOOM_STORAGE_SPACE_STACK, 8, 32);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_low_return_build(&body_builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &return_op));

  IREE_ASSERT_OK(loom_low_storage_layout_hoist_reservations(module_, body,
                                                            &layout_arena_));
  // Repeated frame preparation leaves the established declaration order alone.
  IREE_ASSERT_OK(loom_low_storage_layout_hoist_reservations(module_, body,
                                                            &layout_arena_));
  const loom_value_id_t expected[] = {stack, private_storage, scratch,
                                      aligned_stack};
  loom_low_storage_layout_builder_t builder;
  loom_low_storage_layout_builder_initialize(nullptr, &builder);
  ASSERT_EQ(entry->first_op, live_in);
  ASSERT_EQ(live_in->next_op, resource);
  loom_op_t* op = resource->next_op;
  for (loom_value_id_t value : expected) {
    ASSERT_NE(op, nullptr);
    ASSERT_TRUE(loom_low_storage_reserve_isa(op));
    EXPECT_EQ(loom_low_storage_reserve_storage(op), value);
    IREE_ASSERT_OK(loom_low_storage_layout_builder_append(
        module_, op, &layout_arena_, &builder));
    op = op->next_op;
  }
  EXPECT_EQ(op, loom_value_def_op(loom_module_value(module_, entry_view)));
  EXPECT_EQ(op->next_op, branch);
  EXPECT_EQ(tail->first_op,
            loom_value_def_op(loom_module_value(module_, outer)));
  EXPECT_EQ(tail->last_op, return_op);

  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(loom_low_storage_layout_builder_finish(&builder, &layout));
  ExpectReservation(layout, stack, LOOM_STORAGE_SPACE_STACK, 0, 16, 8);
  ExpectReservation(layout, private_storage, LOOM_STORAGE_SPACE_PRIVATE, 0, 8,
                    8);
  ExpectReservation(layout, scratch, LOOM_STORAGE_SPACE_SCRATCH, 0, 32, 16);
  ExpectReservation(layout, aligned_stack, LOOM_STORAGE_SPACE_STACK, 32, 8, 32);
  loom_low_storage_layout_reference_t reference = {};
  loom_low_storage_layout_lookup_reference(&layout, module_, nested,
                                           &reference);
  EXPECT_EQ(reference.byte_offset, 12u);
  EXPECT_EQ(reference.byte_length, 8u);
}

TEST_F(LowStorageLayoutTest, PlacementIncludesPaddingAndStrongestAlignment) {
  Reserve(LOOM_STORAGE_SPACE_STACK, 8, 4);
  Reserve(LOOM_STORAGE_SPACE_PRIVATE, 12, 64);
  Reserve(LOOM_STORAGE_SPACE_STACK, 16, 16);
  Reserve(LOOM_STORAGE_SPACE_STACK, 4, 4);
  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &layout));
  const auto stack =
      loom_low_storage_layout_requirement(&layout, LOOM_STORAGE_SPACE_STACK);
  EXPECT_EQ(stack.byte_length, 36u);
  EXPECT_EQ(stack.minimum_alignment, 16u);
  const auto private_storage =
      loom_low_storage_layout_requirement(&layout, LOOM_STORAGE_SPACE_PRIVATE);
  EXPECT_EQ(private_storage.byte_length, 12u);
  EXPECT_EQ(private_storage.minimum_alignment, 64u);
}

TEST_F(LowStorageLayoutTest, EmptySpaceHasNoPlacementRequirement) {
  Reserve(LOOM_STORAGE_SPACE_PRIVATE, 8, 8);
  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &layout));
  const auto requirement =
      loom_low_storage_layout_requirement(&layout, LOOM_STORAGE_SPACE_STACK);
  EXPECT_EQ(requirement.byte_length, 0u);
  EXPECT_EQ(requirement.minimum_alignment, 0u);
}

TEST_F(LowStorageLayoutTest, TailAlignmentAppliesAfterEveryFixedReservation) {
  const auto first = Reserve(LOOM_STORAGE_SPACE_WORKGROUP, 20, 4);
  loom_low_storage_layout_builder_require_workgroup_tail(64, &layout_builder_);
  // A later compiler-generated allocation still belongs before the tail.
  const auto late = Reserve(LOOM_STORAGE_SPACE_WORKGROUP, 64, 4);
  loom_low_storage_layout_builder_require_workgroup_tail(4, &layout_builder_);
  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &layout));
  EXPECT_EQ(layout.workgroup_tail_alignment, 64u);
  EXPECT_EQ(layout.space_sizes.workgroup_bytes, 128u);
  EXPECT_EQ(layout.workgroup_record_count, 2u);
  ExpectReservation(layout, first, LOOM_STORAGE_SPACE_WORKGROUP, 0, 20, 4);
  ExpectReservation(layout, late, LOOM_STORAGE_SPACE_WORKGROUP, 20, 64, 4);
  const auto requirement = loom_low_storage_layout_requirement(
      &layout, LOOM_STORAGE_SPACE_WORKGROUP);
  EXPECT_EQ(requirement.byte_length, 128u);
  EXPECT_EQ(requirement.minimum_alignment, 64u);
}

TEST_F(LowStorageLayoutTest, TailOnlyHasAlignmentWithoutInventedReservation) {
  loom_low_storage_layout_builder_require_workgroup_tail(64, &layout_builder_);
  loom_low_storage_layout_t layout = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &layout));
  EXPECT_EQ(layout.record_count, 0u);
  EXPECT_EQ(layout.space_sizes.workgroup_bytes, 0u);
  const auto requirement = loom_low_storage_layout_requirement(
      &layout, LOOM_STORAGE_SPACE_WORKGROUP);
  EXPECT_EQ(requirement.byte_length, 0u);
  EXPECT_EQ(requirement.minimum_alignment, 64u);
}

TEST_F(LowStorageLayoutTest, ImportsWorkgroupPrefixWhilePrivateStorageChanges) {
  Reserve(LOOM_STORAGE_SPACE_PRIVATE, 16, 4);
  const auto fixed = Reserve(LOOM_STORAGE_SPACE_WORKGROUP, 20, 4);
  loom_low_storage_layout_builder_require_workgroup_tail(64, &layout_builder_);
  loom_low_storage_layout_t source = {};
  IREE_ASSERT_OK(
      loom_low_storage_layout_builder_finish(&layout_builder_, &source));
  loom_low_storage_layout_t workgroup = {};
  IREE_ASSERT_OK(loom_low_storage_layout_project_workgroup(
      &source, &layout_arena_, &workgroup));
  ASSERT_EQ(workgroup.record_count, 1u);
  EXPECT_EQ(workgroup.space_sizes.private_bytes, 0u);
  EXPECT_EQ(workgroup.space_sizes.workgroup_bytes, 64u);

  // Native repair appends private storage while the WG declarations stay put.
  Reserve(LOOM_STORAGE_SPACE_PRIVATE, 32, 32);
  const auto* block =
      loom_region_entry_block(loom_low_func_def_body(function_op_));
  // A speculative frame borrows the full accepted layout; a checkpoint
  // replacement imports the copied workgroup projection instead.
  for (const auto* snapshot : {&source, &workgroup}) {
    loom_low_storage_layout_builder_t rebuilt_builder = {};
    loom_low_storage_layout_builder_initialize(snapshot, &rebuilt_builder);
    for (const loom_op_t* op = block->first_op; op != nullptr;
         op = op->next_op) {
      IREE_ASSERT_OK(loom_low_storage_layout_builder_append(
          module_, op, &layout_arena_, &rebuilt_builder));
    }
    loom_low_storage_layout_t rebuilt = {};
    IREE_ASSERT_OK(
        loom_low_storage_layout_builder_finish(&rebuilt_builder, &rebuilt));
    EXPECT_EQ(rebuilt.space_sizes.private_bytes, 64u);
    EXPECT_EQ(rebuilt.space_sizes.workgroup_bytes, 64u);
    EXPECT_EQ(rebuilt.workgroup_tail_alignment, 64u);
    ExpectReservation(rebuilt, fixed, LOOM_STORAGE_SPACE_WORKGROUP, 0, 20, 4);
    EXPECT_NE(rebuilt.records, snapshot->records);
  }
}

TEST_F(LowStorageLayoutTest, RejectsTailPaddingOverflow) {
  Reserve(LOOM_STORAGE_SPACE_WORKGROUP, INT64_MAX, 1);
  Reserve(LOOM_STORAGE_SPACE_WORKGROUP, INT64_MAX, 1);
  loom_low_storage_layout_builder_require_workgroup_tail(4, &layout_builder_);
  loom_low_storage_layout_t layout = {};
  EXPECT_THAT(iree::Status(loom_low_storage_layout_builder_finish(
                  &layout_builder_, &layout)),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
}

}  // namespace
}  // namespace loom
