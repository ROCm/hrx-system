// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/storage_layout.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

namespace loom {
namespace {

class AmdgpuStorageLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &layout_arena_);
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
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &module_builder, 0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0,
        /*allocation=*/0, /*schedule=*/0, target_ref, /*abi=*/0,
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
  }

  loom_value_id_t Reserve(loom_storage_space_t space, int64_t byte_length,
                          int64_t byte_alignment) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_storage_reserve_build(
        &body_builder_, byte_length, byte_alignment, loom_type_storage(space),
        LOOM_LOCATION_UNKNOWN, &op));
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

  void ExpectReservation(const loom_amdgpu_storage_layout_t& layout,
                         loom_value_id_t storage_value_id,
                         loom_storage_space_t expected_space,
                         uint64_t expected_byte_offset,
                         uint64_t expected_byte_size,
                         uint64_t expected_byte_alignment) {
    loom_amdgpu_storage_layout_reservation_t reservation = {};
    IREE_EXPECT_OK(loom_amdgpu_storage_layout_lookup(&layout, storage_value_id,
                                                     &reservation));
    EXPECT_EQ(reservation.space, expected_space);
    EXPECT_EQ(reservation.byte_offset, expected_byte_offset);
    EXPECT_EQ(reservation.byte_size, expected_byte_size);
    EXPECT_EQ(reservation.byte_alignment, expected_byte_alignment);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t layout_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* function_op_ = nullptr;
  loom_builder_t body_builder_;
};

TEST_F(AmdgpuStorageLayoutTest,
       ProjectsScratchAndPrivateIntoOnePrivateSegment) {
  const loom_value_id_t scratch0 = Reserve(LOOM_STORAGE_SPACE_SCRATCH, 8, 8);
  const loom_value_id_t private0 = Reserve(LOOM_STORAGE_SPACE_PRIVATE, 4, 4);
  const loom_value_id_t scratch1 = Reserve(LOOM_STORAGE_SPACE_SCRATCH, 16, 16);
  const loom_value_id_t workgroup0 =
      Reserve(LOOM_STORAGE_SPACE_WORKGROUP, 12, 4);

  loom_amdgpu_storage_layout_segment_sizes_t sizes = {};
  IREE_EXPECT_OK(loom_amdgpu_storage_layout_collect_segment_sizes(
      module_, function_op_, &sizes));
  EXPECT_EQ(sizes.private_segment_fixed_size, 32u);
  EXPECT_EQ(sizes.group_segment_fixed_size, 12u);

  loom_amdgpu_storage_layout_t layout = {};
  IREE_EXPECT_OK(loom_amdgpu_storage_layout_build(module_, function_op_,
                                                  &layout_arena_, &layout));
  ExpectReservation(layout, scratch0, LOOM_STORAGE_SPACE_SCRATCH, 0, 8, 8);
  ExpectReservation(layout, private0, LOOM_STORAGE_SPACE_PRIVATE, 8, 4, 4);
  ExpectReservation(layout, scratch1, LOOM_STORAGE_SPACE_SCRATCH, 16, 16, 16);
  ExpectReservation(layout, workgroup0, LOOM_STORAGE_SPACE_WORKGROUP, 0, 12, 4);
}

TEST_F(AmdgpuStorageLayoutTest, ResolvesViewsAgainstProjectedOffsets) {
  const loom_value_id_t scratch = Reserve(LOOM_STORAGE_SPACE_SCRATCH, 8, 8);
  const loom_value_id_t private_storage =
      Reserve(LOOM_STORAGE_SPACE_PRIVATE, 8, 4);
  const loom_value_id_t view = View(private_storage, 2, 4);

  loom_amdgpu_storage_layout_t layout = {};
  IREE_EXPECT_OK(loom_amdgpu_storage_layout_build(module_, function_op_,
                                                  &layout_arena_, &layout));
  loom_amdgpu_storage_layout_reference_t reference = {};
  IREE_EXPECT_OK(loom_amdgpu_storage_layout_lookup_reference(&layout, module_,
                                                             view, &reference));
  EXPECT_EQ(reference.reservation.space, LOOM_STORAGE_SPACE_PRIVATE);
  EXPECT_EQ(reference.reservation.byte_offset, 8u);
  EXPECT_EQ(reference.byte_offset, 2u);
  EXPECT_EQ(reference.byte_length, 4u);

  loom_amdgpu_storage_layout_reservation_t scratch_reservation = {};
  IREE_EXPECT_OK(loom_amdgpu_storage_layout_lookup(&layout, scratch,
                                                   &scratch_reservation));
  EXPECT_EQ(scratch_reservation.byte_offset, 0u);
}

TEST_F(AmdgpuStorageLayoutTest, RejectsStackStorage) {
  Reserve(LOOM_STORAGE_SPACE_STACK, 8, 4);

  loom_amdgpu_storage_layout_segment_sizes_t sizes = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_amdgpu_storage_layout_collect_segment_sizes(
                            module_, function_op_, &sizes));
}

}  // namespace
}  // namespace loom
