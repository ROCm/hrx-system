// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_defs.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class OpEraseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_kernel_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_KERNEL,
                                                 vtables, (uint16_t)count));
    vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("erase"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Storage backing the module's invocation-lifetime arena.
  iree_arena_block_pool_t block_pool_ = {};
  // Invocation scratch kept separate from the module's persistent records.
  iree_arena_allocator_t scratch_arena_ = {};
  // Immutable operation metadata shared by the fixture's builders.
  loom_context_t context_ = {};
  // Owned module whose retained reference state is under test.
  loom_module_t* module_ = nullptr;
};

class RegionRemovalAttributeTest
    : public OpEraseTest,
      public ::testing::WithParamInterface<loom_attr_kind_t> {};

TEST_P(RegionRemovalAttributeTest, RejectsExternalReferenceBeforeMutation) {
  loom_region_t* region = module_->body;
  loom_block_t* removed = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &removed));
  loom_block_t* referenced = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &referenced));
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t width = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index_type, &width));
  IREE_ASSERT_OK(loom_block_add_arg(module_, removed, width));
  loom_type_id_t type = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_group_1d(loom_dim_pack_dynamic(width)), &type));
  loom_predicate_t predicate = {LOOM_PREDICATE_EQ,
                                2,
                                {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
                                {},
                                {width, 16}};
  loom_string_id_t key = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("constraint"), &key));
  const loom_attribute_t attribute =
      GetParam() == LOOM_ATTR_TYPE ? loom_attr_type(type)
                                   : loom_attr_predicate_list(&predicate, 1);
  const loom_named_attr_t attributes[] = {{key, {}, attribute}};
  loom_builder_t builder = {};
  loom_builder_initialize(module_, &module_->arena,
                          loom_region_entry_block(region), &builder);
  loom_op_t* input = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(1), index_type, LOOM_LOCATION_UNKNOWN, &input));
  loom_builder_set_block(&builder, referenced);
  loom_op_t* owner = nullptr;
  IREE_ASSERT_OK(
      loom_test_attrs_build(&builder, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT,
                            loom_test_constant_result(input),
                            loom_make_named_attr_slice(attributes, 1),
                            index_type, LOOM_LOCATION_UNKNOWN, &owner));
  ASSERT_TRUE(loom_value_has_attribute_uses(loom_module_value(module_, width)));
  ASSERT_EQ(loom_module_value(module_, width)->use_count, 0u);
  ASSERT_FALSE(loom_module_value_has_type_uses(module_, width));

  // The removal request is not closed: the attribute owner is kept.
  bool remove_blocks[] = {false, true, false};
  uint16_t removed_count = UINT16_MAX;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_region_remove_blocks(module_, region, remove_blocks,
                                IREE_ARRAYSIZE(remove_blocks), &scratch_arena_,
                                &removed_count));
  EXPECT_EQ(removed_count, 0u);
  EXPECT_EQ(region->block_count, 3u);
  EXPECT_EQ(removed->parent_region, region);
  EXPECT_EQ(removed->arg_count, 1u);
  ASSERT_TRUE(loom_value_is_block_arg(loom_module_value(module_, width)));
  EXPECT_EQ(loom_value_def_block(loom_module_value(module_, width)), removed);
  EXPECT_TRUE(loom_value_has_attribute_uses(loom_module_value(module_, width)));
  EXPECT_EQ(owner->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);

  // Nested kept and removed owners exercise both outcomes of the block index.
  IREE_ASSERT_OK(loom_op_erase(module_, owner));
  loom_op_t* kept_scope = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(
      &builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &kept_scope));
  loom_builder_set_block(
      &builder, loom_region_entry_block(loom_test_block_args_body(kept_scope)));
  IREE_ASSERT_OK(
      loom_test_attrs_build(&builder, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT,
                            loom_test_constant_result(input),
                            loom_make_named_attr_slice(attributes, 1),
                            index_type, LOOM_LOCATION_UNKNOWN, &owner));
  loom_builder_set_block(&builder, removed);
  loom_op_t* removed_scope = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(
      &builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &removed_scope));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_region_remove_blocks(module_, region, remove_blocks,
                                IREE_ARRAYSIZE(remove_blocks), &scratch_arena_,
                                &removed_count));
  EXPECT_EQ(removed_count, 0u);
  EXPECT_EQ(region->block_count, 3u);
  EXPECT_EQ(removed_scope->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_EQ(owner->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_TRUE(loom_value_has_attribute_uses(loom_module_value(module_, width)));

  // Including the attribute's entire subtree makes the same request closed.
  remove_blocks[2] = true;
  IREE_ASSERT_OK(loom_region_remove_blocks(module_, region, remove_blocks,
                                           IREE_ARRAYSIZE(remove_blocks),
                                           &scratch_arena_, &removed_count));
  EXPECT_EQ(removed_count, 2u);
  EXPECT_EQ(region->block_count, 1u);
  EXPECT_NE(removed_scope->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_NE(kept_scope->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_NE(owner->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(module_, width)));
  EXPECT_FALSE(loom_value_is_block_arg(loom_module_value(module_, width)));
  EXPECT_EQ(
      loom_module_value(module_, loom_test_constant_result(input))->use_count,
      0u);
}

INSTANTIATE_TEST_SUITE_P(EmbeddedReferences, RegionRemovalAttributeTest,
                         ::testing::Values(LOOM_ATTR_TYPE,
                                           LOOM_ATTR_PREDICATE_LIST));

TEST_F(OpEraseTest, RegionRemovalAllocationFailureLeavesIRUnchanged) {
  loom_block_t* removed = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, module_->body, &removed));
  loom_builder_t builder = {};
  loom_builder_initialize(module_, &module_->arena, removed, &builder);
  loom_op_t* nested = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(&builder, nullptr, 0,
                                            LOOM_LOCATION_UNKNOWN, &nested));
  iree_arena_block_pool_t empty_pool = {};
  iree_arena_block_pool_initialize(4096, iree_allocator_null(), &empty_pool);
  iree_arena_allocator_t failing_arena = {};
  iree_arena_initialize(&empty_pool, &failing_arena);
  const bool remove_blocks[] = {false, true};
  uint16_t removed_count = UINT16_MAX;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_region_remove_blocks(module_, module_->body, remove_blocks,
                                IREE_ARRAYSIZE(remove_blocks), &failing_arena,
                                &removed_count));
  EXPECT_EQ(removed_count, 0u);
  EXPECT_EQ(module_->body->block_count, 2u);
  EXPECT_EQ(removed->parent_region, module_->body);
  EXPECT_EQ(removed->first_op, nested);
  EXPECT_EQ(nested->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_EQ(failing_arena.used_allocation_size, 0u);
  iree_arena_deinitialize(&failing_arena);
  iree_arena_block_pool_deinitialize(&empty_pool);
}

TEST_F(OpEraseTest, KernelDeclarationDropsBothOwnedSignatures) {
  loom_builder_t builder = {};
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("dispatch"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  const loom_symbol_ref_t callee = {0, symbol};
  const loom_type_t argument_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_group_1d(loom_dim_pack_static(4)),
  };
  loom_op_t* declaration = nullptr;
  IREE_ASSERT_OK(loom_kernel_decl_build(
      &builder, /*build_flags=*/0, /*retain=*/0, loom_symbol_ref_null(),
      LOOM_STRING_ID_INVALID, /*export_linkage=*/0, callee, argument_types,
      IREE_ARRAYSIZE(argument_types), argument_types,
      IREE_ARRAYSIZE(argument_types), /*predicates=*/nullptr,
      /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &declaration));
  const loom_value_slice_t signatures[] = {
      loom_kernel_decl_workloads(declaration),
      loom_kernel_decl_args(declaration),
  };
  for (const loom_value_slice_t signature : signatures) {
    for (uint16_t i = 0; i < signature.count; ++i) {
      EXPECT_EQ(
          loom_value_owner_op(loom_module_value(module_, signature.values[i])),
          declaration);
    }
    IREE_ASSERT_OK(loom_module_set_value_type(
        module_, signature.values[1],
        loom_type_group_1d(loom_dim_pack_dynamic(signature.values[0]))));
  }
  ASSERT_EQ(module_->type_uses.active_carrier_count, 2u);
  const iree_host_size_t arena_bytes = module_->arena.used_allocation_size;
  std::vector<uint32_t> visits(module_->values.count, 0);
  IREE_ASSERT_OK(loom_op_walk_subtree_value_refs(
      module_, declaration,
      [](loom_value_id_t value, void* user_data) {
        ++(*static_cast<std::vector<uint32_t>*>(user_data))[value];
        return iree_ok_status();
      },
      &visits));
  for (const loom_value_slice_t signature : signatures) {
    EXPECT_EQ(visits[signature.values[0]], 1u);
    EXPECT_EQ(visits[signature.values[1]], 0u);
  }
  IREE_ASSERT_OK(loom_op_erase(module_, declaration));
  EXPECT_FALSE(loom_module_has_active_type_uses(module_));
  EXPECT_EQ(module_->arena.used_allocation_size, arena_bytes);
  for (const loom_value_slice_t signature : signatures) {
    EXPECT_EQ(
        loom_value_owner_op(loom_module_value(module_, signature.values[1])),
        nullptr);
    EXPECT_EQ(loom_module_value(module_, signature.values[1])->use_count, 0u);
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(module_, signature.values[1],
                                        &dependencies);
    EXPECT_EQ(loom_type_dependencies_next(&dependencies),
              LOOM_VALUE_ID_INVALID);
    EXPECT_FALSE(loom_module_value_has_type_uses(module_, signature.values[0]));
  }
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  EXPECT_FALSE(loom_module_has_active_type_uses(module_));
}

TEST(DialectTableHelpers, ReturnVtableArraysAndCounts) {
  const loom_op_vtable_t vtable = {};
  const loom_op_vtable_t* const vtables[] = {
      &vtable,
  };

  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* result =
      loom_dialect_vtable_array(vtables, IREE_ARRAYSIZE(vtables), &count);
  EXPECT_EQ(result, vtables);
  EXPECT_EQ(count, 1u);

  EXPECT_EQ(loom_dialect_vtable_array(vtables, IREE_ARRAYSIZE(vtables),
                                      /*out_count=*/nullptr),
            vtables);
}

TEST(DialectTableHelpers, ReturnSemanticArraysAndCounts) {
  loom_op_semantics_t semantics[] = {
      loom_op_semantics_empty(),
      loom_op_semantics_empty(),
  };
  semantics[1].phase = LOOM_OP_PHASE_EXECUTABLE;
  semantics[1].contract_families = LOOM_CONTRACT_VECTOR_COORDINATE;

  iree_host_size_t count = 0;
  const loom_op_semantics_t* result = loom_dialect_semantics_array(
      semantics, IREE_ARRAYSIZE(semantics), &count);
  EXPECT_EQ(result, semantics);
  EXPECT_EQ(count, 2u);

  EXPECT_EQ(loom_dialect_semantics_array(semantics, IREE_ARRAYSIZE(semantics),
                                         /*out_count=*/nullptr),
            semantics);
}

TEST(DialectTableHelpers, LookupSemanticsByDialectAndIndex) {
  loom_op_semantics_t semantics[] = {
      loom_op_semantics_empty(),
      loom_op_semantics_empty(),
  };
  semantics[1].phase = LOOM_OP_PHASE_EXECUTABLE;
  semantics[1].contract_families = LOOM_CONTRACT_VECTOR_COORDINATE;

  loom_op_semantics_t found = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_TEST, 1), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(found.phase, LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_EQ(found.contract_families, LOOM_CONTRACT_VECTOR_COORDINATE);

  loom_op_semantics_t wrong_dialect = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 1), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(wrong_dialect.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(wrong_dialect.contract_families, 0u);

  loom_op_semantics_t out_of_range = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_TEST, 2), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(out_of_range.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(out_of_range.contract_families, 0u);
}

TEST(MemoryAccessHelpers, OperandIndexIsPayload) {
  loom_op_t op = {.operand_count = 5};

  loom_memory_access_vtable_t memory_access = {
      .operation_kind = LOOM_MEMORY_ACCESS_OPERATION_STORE,
      .value_operand_index = 3,
      .expected_operand_index = LOOM_OPERAND_INDEX_NONE,
      .replacement_operand_index = LOOM_OPERAND_INDEX_NONE,
  };

  loom_op_vtable_t op_vtable = {};
  op_vtable.fixed_operand_count = op.operand_count;
  op_vtable.memory_access = &memory_access;

  loom_memory_access_t access = {.op = &op, .op_vtable = &op_vtable};

  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 0));
  EXPECT_TRUE(loom_memory_access_operand_index_is_payload(access, 3));
  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 5));

  memory_access.operation_kind = LOOM_MEMORY_ACCESS_OPERATION_LOAD;
  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 3));

  memory_access.operation_kind = LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_CMPXCHG;
  memory_access.value_operand_index = LOOM_OPERAND_INDEX_NONE;
  memory_access.expected_operand_index = 1;
  memory_access.replacement_operand_index = 2;
  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 0));
  EXPECT_TRUE(loom_memory_access_operand_index_is_payload(access, 1));
  EXPECT_TRUE(loom_memory_access_operand_index_is_payload(access, 2));
}

}  // namespace
}  // namespace loom
