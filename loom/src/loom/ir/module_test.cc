// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.name=*/IREE_SV("q8_0"),
};

static const loom_encoding_vtable_t kQ6KEncodingVtable = {
    /*.name=*/IREE_SV("q6_k"),
};

static const loom_encoding_vtable_t kDenseEncodingVtable = {
    /*.name=*/IREE_SV("dense"),
    /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};

class ModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ6KEncodingVtable));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kDenseEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

//===----------------------------------------------------------------------===//
// Module lifecycle
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, AllocateAndFree) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_NE(module->body, nullptr);
  EXPECT_EQ(module->body->block_count, 1);
  loom_module_free(module);
}

TEST_F(ModuleTest, FreeNull) { loom_module_free(NULL); }

TEST_F(ModuleTest, ModuleName) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("my_module"),
                                      &block_pool_, NULL,
                                      iree_allocator_system(), &module));
  iree_string_view_t name = module->strings.entries[module->name_id];
  EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("my_module")));
  loom_module_free(module);
}

TEST_F(ModuleTest, RegisterSourceDeduplicatesBySpelling) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_source_id_t first_id = LOOM_SOURCE_ID_INVALID;
  loom_source_id_t second_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("model.loom"), &first_id));
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("model.loom"), &second_id));

  EXPECT_EQ(first_id, 0u);
  EXPECT_EQ(second_id, first_id);
  ASSERT_EQ(module->sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module->sources.entries[0],
                                     IREE_SV("model.loom")));
  loom_module_free(module);
}

TEST_F(ModuleTest, RegisterEmptySource) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_register_source(module, iree_string_view_empty(),
                                             &source_id));

  EXPECT_EQ(source_id, 0u);
  ASSERT_EQ(module->sources.count, 1u);
  EXPECT_EQ(module->sources.entries[0].size, 0u);
  loom_module_free(module);
}

TEST_F(ModuleTest, RegisterSourceRejectsInvalidSentinelId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  iree_string_view_t* entries = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_array(&module->arena,
                                           LOOM_SOURCE_ID_INVALID,
                                           sizeof(*entries), (void**)&entries));
  module->sources.entries = entries;
  module->sources.capacity = LOOM_SOURCE_ID_INVALID;
  module->sources.count = LOOM_SOURCE_ID_INVALID;

  loom_source_id_t source_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_register_source(module, IREE_SV("overflow"), &source_id));
  loom_module_free(module);
}

TEST_F(ModuleTest, ValueNameHelpersSkipAnonymousSource) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &source));
  loom_value_id_t target = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &target));

  const iree_host_size_t string_count = module->strings.count;
  IREE_ASSERT_OK(loom_module_copy_value_name(module, source, target));
  EXPECT_EQ(loom_module_value(module, target)->name_id, LOOM_STRING_ID_INVALID);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  IREE_ASSERT_OK(loom_module_try_set_derived_value_name(
      module, source, target, IREE_SV("bounded"), &scratch_arena));
  iree_arena_deinitialize(&scratch_arena);
  EXPECT_EQ(loom_module_value(module, target)->name_id, LOOM_STRING_ID_INVALID);
  EXPECT_EQ(module->strings.count, string_count);
  loom_module_free(module);
}

TEST_F(ModuleTest, ValueNameHelpersCopyMoveAndDerive) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &source));
  loom_value_id_t copied = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &copied));
  loom_value_id_t derived = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &derived));
  loom_value_id_t overwritten = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &overwritten));
  loom_value_id_t moved = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &moved));

  loom_string_id_t source_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("head"), &source_name));
  IREE_ASSERT_OK(loom_module_set_value_name(module, source, source_name));

  IREE_ASSERT_OK(loom_module_copy_value_name(module, source, copied));
  EXPECT_EQ(loom_module_value(module, copied)->name_id, source_name);

  loom_string_id_t temporary_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("temporary"), &temporary_name));
  IREE_ASSERT_OK(
      loom_module_set_value_name(module, overwritten, temporary_name));
  IREE_ASSERT_OK(loom_module_overwrite_value_name(module, source, overwritten));
  EXPECT_EQ(loom_module_value(module, overwritten)->name_id, source_name);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  IREE_ASSERT_OK(loom_module_try_set_derived_value_name(
      module, source, derived, IREE_SV("bounded"), &scratch_arena));
  iree_arena_deinitialize(&scratch_arena);
  loom_string_id_t derived_name = loom_module_value(module, derived)->name_id;
  ASSERT_NE(derived_name, LOOM_STRING_ID_INVALID);
  ASSERT_LT(derived_name, module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(module->strings.entries[derived_name],
                                     IREE_SV("head_bounded")));

  IREE_ASSERT_OK(loom_module_move_value_name(module, source, moved));
  EXPECT_EQ(loom_module_value(module, source)->name_id, LOOM_STRING_ID_INVALID);
  EXPECT_EQ(loom_module_value(module, moved)->name_id, source_name);

  IREE_ASSERT_OK(loom_module_clear_value_name(module, moved));
  EXPECT_EQ(loom_module_value(module, moved)->name_id, LOOM_STRING_ID_INVALID);
  loom_module_free(module);
}

TEST_F(ModuleTest, BodyBlock) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(block->op_count, 0);
  EXPECT_EQ(block->first_op, nullptr);
  EXPECT_EQ(block->last_op, nullptr);
  loom_module_free(module);
}

TEST_F(ModuleTest, CompactSymbolsDropsUnreferencedTombstonesAndRenumbersRefs) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t keep_a_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("keep_a"), &keep_a_name));
  uint16_t keep_a_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, keep_a_name, &keep_a_symbol_id));

  loom_string_id_t drop_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("drop"), &drop_name));
  uint16_t drop_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, drop_name, &drop_symbol_id));

  loom_string_id_t keep_b_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("keep_b"), &keep_b_name));
  uint16_t keep_b_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, keep_b_name, &keep_b_symbol_id));

  ASSERT_EQ(keep_a_symbol_id, 0u);
  ASSERT_EQ(drop_symbol_id, 1u);
  ASSERT_EQ(keep_b_symbol_id, 2u);

  loom_string_id_t peer_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("peer"), &peer_name));
  loom_named_attr_t keep_a_dict[] = {{
      /*.name_id=*/peer_name,
      /*.reserved=*/{},
      /*.value=*/loom_attr_symbol((loom_symbol_ref_t){0, keep_b_symbol_id}),
  }};

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* keep_a_op = NULL;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, (loom_symbol_ref_t){0, keep_a_symbol_id},
      loom_make_named_attr_slice(keep_a_dict, IREE_ARRAYSIZE(keep_a_dict)),
      LOOM_LOCATION_UNKNOWN, &keep_a_op));
  loom_op_t* keep_b_op = NULL;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, (loom_symbol_ref_t){0, keep_b_symbol_id},
      loom_make_named_attr_slice(NULL, 0), LOOM_LOCATION_UNKNOWN, &keep_b_op));

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  iree_host_size_t removed_count = 0;
  IREE_ASSERT_OK(
      loom_module_compact_symbols(module, &scratch_arena, &removed_count));
  iree_arena_deinitialize(&scratch_arena);

  EXPECT_EQ(removed_count, 1u);
  EXPECT_EQ(module->symbols.count, 2u);
  EXPECT_EQ(loom_module_find_symbol(module, keep_a_name), 0u);
  EXPECT_EQ(loom_module_find_symbol(module, keep_b_name), 1u);
  EXPECT_EQ(loom_module_find_symbol(module, drop_name), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_test_record_symbol(keep_b_op).symbol_id, 1u);

  loom_named_attr_slice_t dict = loom_test_record_dict(keep_a_op);
  ASSERT_EQ(dict.count, 1u);
  EXPECT_EQ(loom_attr_as_symbol(dict.entries[0].value).symbol_id, 1u);

  loom_module_free(module);
}

TEST_F(ModuleTest, CompactSymbolsCanPreserveExternalRefOrdinal) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t keep_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("keep"), &keep_name));
  uint16_t keep_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, keep_name, &keep_symbol_id));

  loom_string_id_t drop_before_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("drop_before"),
                                           &drop_before_name));
  uint16_t drop_before_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, drop_before_name, &drop_before_symbol_id));

  loom_string_id_t external_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("external"), &external_name));
  uint16_t external_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, external_name, &external_symbol_id));

  loom_string_id_t drop_after_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("drop_after"),
                                           &drop_after_name));
  uint16_t drop_after_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, drop_after_name, &drop_after_symbol_id));

  ASSERT_EQ(keep_symbol_id, 0u);
  ASSERT_EQ(drop_before_symbol_id, 1u);
  ASSERT_EQ(external_symbol_id, 2u);
  ASSERT_EQ(drop_after_symbol_id, 3u);

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* keep_op = NULL;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, (loom_symbol_ref_t){0, keep_symbol_id},
      loom_make_named_attr_slice(NULL, 0), LOOM_LOCATION_UNKNOWN, &keep_op));
  loom_op_t* external_op = NULL;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, (loom_symbol_ref_t){0, external_symbol_id},
      loom_make_named_attr_slice(NULL, 0), LOOM_LOCATION_UNKNOWN,
      &external_op));

  const loom_symbol_ref_t preserved_symbol_refs[] = {
      (loom_symbol_ref_t){0, external_symbol_id},
  };
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  iree_host_size_t removed_count = 0;
  IREE_ASSERT_OK(loom_module_compact_symbols_preserving_symbol_refs(
      module, preserved_symbol_refs, IREE_ARRAYSIZE(preserved_symbol_refs),
      &scratch_arena, &removed_count));
  iree_arena_deinitialize(&scratch_arena);

  EXPECT_EQ(removed_count, 1u);
  ASSERT_EQ(module->symbols.count, 3u);
  EXPECT_EQ(loom_module_find_symbol(module, keep_name), 0u);
  EXPECT_EQ(loom_module_find_symbol(module, drop_before_name),
            LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(module->symbols.entries[1].name_id, LOOM_STRING_ID_INVALID);
  EXPECT_EQ(loom_module_find_symbol(module, external_name), 2u);
  EXPECT_EQ(loom_test_record_symbol(external_op).symbol_id, 2u);
  EXPECT_EQ(loom_module_find_symbol(module, drop_after_name),
            LOOM_SYMBOL_ID_INVALID);

  loom_module_free(module);
}

TEST_F(ModuleTest, RegionAppendBlockGrowthKeepsBlockReferencesStable) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_region_t* body = module->body;
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 1u);
  ASSERT_EQ(body->block_capacity, 1u);

  loom_block_t* old_entry = loom_region_entry_block(body);
  EXPECT_EQ(loom_block_region_index(old_entry), 0u);

  loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &arg_id));
  IREE_ASSERT_OK(loom_block_add_arg(module, old_entry, arg_id));

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op));
  memset(op, 0, sizeof(loom_op_t));
  op->kind = LOOM_OP_TEST_CONSTANT;
  op->parent_block = old_entry;
  IREE_ASSERT_OK(loom_block_append_op(module, old_entry, op));

  loom_block_t* appended = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &appended));

  ASSERT_EQ(body->block_count, 2u);
  EXPECT_GE(body->block_capacity, body->block_count);
  EXPECT_EQ(appended, loom_region_block(body, 1));
  EXPECT_EQ(loom_block_region_index(appended), 1u);
  EXPECT_EQ(loom_region_entry_block(body), old_entry);
  EXPECT_EQ(loom_block_region_index(old_entry), 0u);
  EXPECT_EQ(appended->label_id, LOOM_STRING_ID_INVALID);
  EXPECT_EQ(appended->arg_count, 0u);
  EXPECT_EQ(appended->op_count, 0u);
  EXPECT_EQ(appended->first_op, nullptr);
  EXPECT_EQ(appended->last_op, nullptr);

  loom_block_t* entry = loom_region_entry_block(body);
  EXPECT_EQ(loom_value_def_block(&module->values.entries[arg_id]), entry);
  EXPECT_EQ(loom_value_def_index(&module->values.entries[arg_id]), 0u);
  EXPECT_EQ(op->parent_block, entry);

  loom_module_free(module);
}

TEST_F(ModuleTest, RegionInsertBlockShiftsBlockTableOnly) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_region_t* body = module->body;
  loom_block_t* entry = loom_region_entry_block(body);
  loom_block_t* tail0 = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &tail0));
  loom_block_t* tail1 = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &tail1));

  loom_block_t* inserted = NULL;
  IREE_ASSERT_OK(loom_region_insert_block(module, body, 1, &inserted));

  ASSERT_EQ(body->block_count, 4u);
  EXPECT_EQ(loom_region_block(body, 0), entry);
  EXPECT_EQ(loom_region_block(body, 1), inserted);
  EXPECT_EQ(loom_region_block(body, 2), tail0);
  EXPECT_EQ(loom_region_block(body, 3), tail1);
  EXPECT_EQ(loom_block_region_index(entry), 0u);
  EXPECT_EQ(loom_block_region_index(inserted), 1u);
  EXPECT_EQ(loom_block_region_index(tail0), 2u);
  EXPECT_EQ(loom_block_region_index(tail1), 3u);
  EXPECT_EQ(loom_region_entry_block(body), entry);

  loom_module_free(module);
}

TEST_F(ModuleTest, BlockAddArgRejectsStorageOverflow) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &arg_id));

  loom_block_t* entry = loom_region_entry_block(module->body);
  entry->arg_count = UINT16_MAX;
  entry->arg_capacity = UINT16_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_block_add_arg(module, entry, arg_id));

  loom_module_free(module);
}

TEST_F(ModuleTest, RegionRemoveBlocksCompactsAndDropsClosedUses) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_region_t* body = module->body;
  loom_block_t* dead_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &dead_block));
  loom_block_t* kept_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &kept_block));

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, i32_type, &arg));
  IREE_ASSERT_OK(loom_block_add_arg(module, dead_block, arg));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, dead_block, &builder);
  loom_op_t* constant = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant));
  loom_op_t* add = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder, arg,
                                      loom_test_constant_result(constant),
                                      i32_type, LOOM_LOCATION_UNKNOWN, &add));

  bool remove_blocks[] = {false, true, false};
  uint16_t removed_count = 0;
  IREE_ASSERT_OK(loom_region_remove_blocks(module, body, remove_blocks,
                                           IREE_ARRAYSIZE(remove_blocks),
                                           &removed_count));

  EXPECT_EQ(removed_count, 1u);
  EXPECT_EQ(body->block_count, 2u);
  EXPECT_EQ(loom_region_block(body, 0), &body->entry_block);
  EXPECT_EQ(loom_block_region_index(&body->entry_block), 0u);
  EXPECT_EQ(loom_region_block(body, 1), kept_block);
  EXPECT_EQ(loom_block_region_index(kept_block), 1u);
  EXPECT_EQ(dead_block->parent_region, nullptr);
  EXPECT_EQ(dead_block->region_index, LOOM_BLOCK_REGION_INDEX_INVALID);
  EXPECT_EQ(dead_block->arg_count, 0u);
  EXPECT_EQ(dead_block->op_count, 0u);
  EXPECT_NE(constant->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_NE(add->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_FALSE(loom_value_is_block_arg(loom_module_value(module, arg)));
  EXPECT_EQ(loom_module_value(module, arg)->use_count, 0u);
  EXPECT_EQ(
      loom_module_value(module, loom_test_constant_result(constant))->use_count,
      0u);

  loom_module_free(module);
}

TEST_F(ModuleTest, RegionRemoveBlocksRejectsKeptSuccessor) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_region_t* body = module->body;
  loom_block_t* dead_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &dead_block));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_region_entry_block(body),
                          &builder);
  loom_op_t* branch = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op_with_successors(
      &builder, LOOM_OP_TEST_YIELD, 0, 0, 1, 0, 0, 0, LOOM_LOCATION_UNKNOWN,
      &branch));
  loom_op_successors(branch)[0] = dead_block;
  IREE_ASSERT_OK(loom_builder_finalize_op(&builder, branch));

  bool remove_blocks[] = {false, true};
  uint16_t removed_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_region_remove_blocks(module, body, remove_blocks,
                                IREE_ARRAYSIZE(remove_blocks), &removed_count));
  EXPECT_EQ(removed_count, 0u);
  EXPECT_EQ(body->block_count, 2u);
  EXPECT_EQ(dead_block->parent_region, body);

  loom_module_free(module);
}

TEST_F(ModuleTest, RegionRemoveBlocksRejectsExternalUse) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_region_t* body = module->body;
  loom_block_t* dead_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &dead_block));
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, dead_block, &builder);
  loom_op_t* constant = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant));
  loom_builder_set_block(&builder, loom_region_entry_block(body));
  loom_op_t* external_use = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder, loom_test_constant_result(constant),
                          i32_type, LOOM_LOCATION_UNKNOWN, &external_use));

  bool remove_blocks[] = {false, true};
  uint16_t removed_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_region_remove_blocks(module, body, remove_blocks,
                                IREE_ARRAYSIZE(remove_blocks), &removed_count));
  EXPECT_EQ(removed_count, 0u);
  EXPECT_EQ(body->block_count, 2u);
  EXPECT_EQ(constant->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_EQ(external_use->flags & LOOM_OP_FLAG_DEAD, 0u);

  loom_module_free(module);
}

TEST_F(ModuleTest, BlockRemoveArgCompactsDefinitions) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_block_t* block = loom_module_block(module);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t dim = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, index_type, &dim));

  loom_value_id_t first = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, i32_type, &first));
  IREE_ASSERT_OK(loom_block_add_arg(module, block, first));

  loom_type_t removed_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(dim), 0);
  loom_value_id_t removed = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, removed_type, &removed));
  IREE_ASSERT_OK(loom_block_add_arg(module, block, removed));
  EXPECT_TRUE(loom_module_value_has_type_uses(module, dim));

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, i32_type, &shifted));
  IREE_ASSERT_OK(loom_block_add_arg(module, block, shifted));

  IREE_ASSERT_OK(loom_block_remove_arg(module, block, 1));

  EXPECT_EQ(block->arg_count, 2u);
  EXPECT_EQ(loom_block_arg_id(block, 0), first);
  EXPECT_EQ(loom_block_arg_id(block, 1), shifted);
  EXPECT_FALSE(loom_value_is_block_arg(loom_module_value(module, removed)));
  EXPECT_EQ(loom_value_def_block(loom_module_value(module, first)), block);
  EXPECT_EQ(loom_value_def_index(loom_module_value(module, first)), 0u);
  EXPECT_EQ(loom_value_def_block(loom_module_value(module, shifted)), block);
  EXPECT_EQ(loom_value_def_index(loom_module_value(module, shifted)), 1u);
  EXPECT_FALSE(loom_module_value_has_type_uses(module, dim));

  loom_module_free(module);
}

TEST_F(ModuleTest, BlockRemoveArgRejectsLiveUses) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_block_t* block = loom_module_block(module);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, i32_type, &arg));
  IREE_ASSERT_OK(loom_block_add_arg(module, block, arg));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, block, &builder);
  loom_op_t* add = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder, arg, arg, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &add));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_block_remove_arg(module, block, 0));

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Value definition
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, DefineValue) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
  EXPECT_EQ(id, 0u);
  EXPECT_EQ(module->values.count, 1u);

  loom_value_t* value = &module->values.entries[id];
  EXPECT_EQ(loom_type_kind(value->type), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(value->type), LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(module->types.count, 1u);
  EXPECT_TRUE(loom_type_equal(module->types.entries[0], f32));
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueInternsShapedTypeClosure) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, vector_type, &id));

  ASSERT_EQ(module->types.count, 2u);
  EXPECT_TRUE(loom_type_equal(module->types.entries[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(loom_type_equal(module->types.entries[1], vector_type));
  EXPECT_TRUE(loom_type_equal(module->values.entries[id].type,
                              module->types.entries[1]));

  loom_module_free(module);
}

TEST_F(ModuleTest, DefineMultipleValues) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t id0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t id1 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t id2 = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id0));
  IREE_ASSERT_OK(loom_module_define_value(module, i32, &id1));
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id2));

  EXPECT_EQ(id0, 0u);
  EXPECT_EQ(id1, 1u);
  EXPECT_EQ(id2, 2u);
  EXPECT_EQ(module->values.count, 3u);
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueAlignment) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
  EXPECT_EQ((uintptr_t)&module->values.entries[id] % 64, 0u)
      << "Value entries must be 64-byte aligned";
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueGrowth) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  // Fill past the default capacity to trigger growth.
  iree_host_size_t initial_capacity = module->values.capacity;
  for (iree_host_size_t i = 0; i < initial_capacity + 100; ++i) {
    loom_value_id_t id = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
    EXPECT_EQ(id, (loom_value_id_t)i);
  }
  EXPECT_GT(module->values.capacity, initial_capacity);
  EXPECT_EQ(module->values.count, initial_capacity + 100);

  // Verify all values are still accessible after growth.
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    EXPECT_EQ(loom_type_kind(module->values.entries[i].type), LOOM_TYPE_SCALAR);
  }
  loom_module_free(module);
}

TEST_F(ModuleTest, ValueOrdinalScratchTracksActiveFrameOnly) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t first_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t second_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &first_id));
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &second_id));

  loom_module_value_ordinal_scratch_acquire(module);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, first_id),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, second_id),
            LOOM_VALUE_ORDINAL_INVALID);

  loom_module_value_ordinal_scratch_set(module, first_id, 7);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, first_id), 7u);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, second_id),
            LOOM_VALUE_ORDINAL_INVALID);

  loom_module_value_ordinal_scratch_clear(module, first_id);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, first_id),
            LOOM_VALUE_ORDINAL_INVALID);
  loom_module_value_ordinal_scratch_release(module);

  loom_module_free(module);
}

TEST_F(ModuleTest, ValueOrdinalScratchGrowsWithValueTable) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const iree_host_size_t initial_capacity = module->values.capacity;
  ASSERT_EQ(module->scratch.values.capacity, initial_capacity);

  loom_value_id_t first_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &first_id));
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, first_id, 3);

  loom_value_id_t last_id = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = module->values.count; i <= initial_capacity; ++i) {
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &last_id));
  }

  EXPECT_GT(module->values.capacity, initial_capacity);
  EXPECT_EQ(module->scratch.values.capacity, module->values.capacity);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, first_id), 3u);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, last_id),
            LOOM_VALUE_ORDINAL_INVALID);

  loom_module_value_ordinal_scratch_clear(module, first_id);
  loom_module_value_ordinal_scratch_release(module);

  loom_module_free(module);
}

TEST_F(ModuleTest, ValueU32ScratchAliasesOrdinalAndZeroedModes) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t first_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t second_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &first_id));
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &second_id));

  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, first_id, 7);
  loom_module_value_ordinal_scratch_clear(module, first_id);
  loom_module_value_ordinal_scratch_release(module);

  loom_value_u32_scratch_acquire_zeroed(&module->scratch.values,
                                        module->values.count);
  EXPECT_EQ(module->scratch.values.values_by_value_id[first_id], 0u);
  EXPECT_EQ(module->scratch.values.values_by_value_id[second_id], 0u);
  module->scratch.values.values_by_value_id[first_id] = 0x3u;
  loom_value_u32_scratch_release_zeroed(&module->scratch.values);

  loom_module_value_ordinal_scratch_acquire(module);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, first_id),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, second_id),
            LOOM_VALUE_ORDINAL_INVALID);
  loom_module_value_ordinal_scratch_release(module);

  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueRejectsInvalidSentinelId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  module->values.count = LOOM_VALUE_ID_INVALID;

  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_define_value(module, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                               &id));

  loom_module_free(module);
}

TEST_F(ModuleTest, TypeUseTableTracksDynamicDims) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &dim_id));
  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim_id), 0);
  loom_value_id_t vector_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, vector_type, &vector_id));

  EXPECT_TRUE(loom_module_value_has_type_uses(module, dim_id));
  IREE_ASSERT_OK(loom_module_set_value_type(
      module, vector_id,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), 0)));
  EXPECT_FALSE(loom_module_value_has_type_uses(module, dim_id));

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceValueTypeUsesUpdatesDynamicDims) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t old_dim_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &old_dim_id));
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &new_dim_id));
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim_id), 0);
  loom_value_id_t vector_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, vector_type, &vector_id));

  IREE_ASSERT_OK(
      loom_module_replace_value_type_uses(module, old_dim_id, new_dim_id));

  loom_type_t replaced_type = loom_module_value_type(module, vector_id);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_type, 0), new_dim_id);
  EXPECT_FALSE(loom_module_value_has_type_uses(module, old_dim_id));
  EXPECT_TRUE(loom_module_value_has_type_uses(module, new_dim_id));

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceTypeValueReferencesDoesNotMutateCarriers) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t old_dim_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &old_dim_id));
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &new_dim_id));
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim_id), 0);
  loom_value_id_t vector_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, vector_type, &vector_id));

  loom_type_t replaced_type = vector_type;
  bool changed = false;
  IREE_ASSERT_OK(loom_module_replace_type_value_references(
      module, vector_type, old_dim_id, new_dim_id, &replaced_type, &changed));

  EXPECT_TRUE(changed);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_type, 0), new_dim_id);
  EXPECT_EQ(
      loom_type_dim_value_id_at(loom_module_value_type(module, vector_id), 0),
      old_dim_id);
  EXPECT_TRUE(loom_module_value_has_type_uses(module, old_dim_id));
  EXPECT_FALSE(loom_module_value_has_type_uses(module, new_dim_id));

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceTypeValueReferencesDoesNotMutateSsaEncodingCarrier) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t layout_type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_value_id_t old_layout_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_layout_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, layout_type, &old_layout_id));
  IREE_ASSERT_OK(loom_module_define_value(module, layout_type, &new_layout_id));
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  view_type.encoding_id = (uint16_t)old_layout_id;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_value_id_t view_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, view_type, &view_id));

  loom_type_t replaced_type = view_type;
  bool changed = false;
  IREE_ASSERT_OK(loom_module_replace_type_value_references(
      module, view_type, old_layout_id, new_layout_id, &replaced_type,
      &changed));

  EXPECT_TRUE(changed);
  ASSERT_TRUE(loom_type_has_ssa_encoding(replaced_type));
  EXPECT_EQ(loom_type_encoding_value_id(replaced_type), new_layout_id);
  ASSERT_TRUE(
      loom_type_has_ssa_encoding(loom_module_value_type(module, view_id)));
  EXPECT_EQ(
      loom_type_encoding_value_id(loom_module_value_type(module, view_id)),
      old_layout_id);
  EXPECT_TRUE(loom_module_value_has_type_uses(module, old_layout_id));
  EXPECT_FALSE(loom_module_value_has_type_uses(module, new_layout_id));

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceValueTypeUsesUpdatesSsaEncoding) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t layout_type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_value_id_t old_layout_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_layout_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, layout_type, &old_layout_id));
  IREE_ASSERT_OK(loom_module_define_value(module, layout_type, &new_layout_id));
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  view_type.encoding_id = (uint16_t)old_layout_id;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_value_id_t view_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, view_type, &view_id));

  IREE_ASSERT_OK(loom_module_replace_value_type_uses(module, old_layout_id,
                                                     new_layout_id));

  loom_type_t replaced_type = loom_module_value_type(module, view_id);
  ASSERT_TRUE(loom_type_has_ssa_encoding(replaced_type));
  EXPECT_EQ(loom_type_encoding_value_id(replaced_type), new_layout_id);
  EXPECT_FALSE(loom_module_value_has_type_uses(module, old_layout_id));
  EXPECT_TRUE(loom_module_value_has_type_uses(module, new_layout_id));

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// String interning
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, InternString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id));
  EXPECT_NE(id, LOOM_STRING_ID_INVALID);

  iree_string_view_t stored = module->strings.entries[id];
  EXPECT_TRUE(iree_string_view_equal(stored, IREE_SV("hello")));
  loom_module_free(module);
}

TEST_F(ModuleTest, InternStringDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id1 = LOOM_STRING_ID_INVALID;
  loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id1));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id2));
  EXPECT_EQ(id1, id2);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternDifferentStrings) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id1 = LOOM_STRING_ID_INVALID;
  loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id1));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("world"), &id2));
  EXPECT_NE(id1, id2);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternEmptyString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, iree_string_view_empty(), &id));
  EXPECT_NE(id, LOOM_STRING_ID_INVALID);
  iree_string_view_t stored = module->strings.entries[id];
  EXPECT_EQ(stored.size, 0u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternStringStress) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  // Intern 1000 unique strings and verify dedup.
  char buffer[32];
  for (int i = 0; i < 1000; ++i) {
    int length = iree_snprintf(buffer, sizeof(buffer), "string_%d", i);
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_string_view(buffer, length), &id));
    // Intern again, expect same ID.
    loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_string_view(buffer, length), &id2));
    EXPECT_EQ(id, id2);
  }
  loom_module_free(module);
}

TEST_F(ModuleTest, LookupString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t hello_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &hello_id));

  EXPECT_EQ(loom_module_lookup_string(module, IREE_SV("hello")), hello_id);
  EXPECT_EQ(loom_module_lookup_string(module, IREE_SV("missing")),
            LOOM_STRING_ID_INVALID);
  EXPECT_EQ(module->strings.count, 2u);  // "test" module name + "hello".

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Canonical dictionary attributes
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, MakeCanonicalAttrDictSortsByKeySpellingAndCopiesEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));
  ASSERT_LT(zeta_id, alpha_id)
      << "setup should prove string_id order is not the"
         " same as spelling order";

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      &attr));

  EXPECT_EQ(attr.kind, LOOM_ATTR_DICT);
  EXPECT_EQ(attr.count, 2);
  ASSERT_NE(attr.dict_entries, nullptr);
  EXPECT_NE(attr.dict_entries, entries);
  EXPECT_EQ(attr.dict_entries[0].name_id, alpha_id);
  EXPECT_EQ(attr.dict_entries[0].reserved, 0u);
  EXPECT_EQ(attr.dict_entries[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(attr.dict_entries[0].value.i64, 1);
  EXPECT_EQ(attr.dict_entries[1].name_id, zeta_id);
  EXPECT_EQ(attr.dict_entries[1].reserved, 0u);
  EXPECT_EQ(attr.dict_entries[1].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(attr.dict_entries[1].value.i64, 2);

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRecursivelyCanonicalizesNestedDicts) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t outer_zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t outer_axis_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t inner_zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t inner_alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("zeta"), &outer_zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("axis"), &outer_axis_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("inner_z"), &inner_zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("inner_a"), &inner_alpha_id));

  int64_t original_values[3] = {7, 8, 9};
  loom_predicate_t original_predicates[1] = {{
      /*.kind=*/LOOM_PREDICATE_RANGE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST, 0},
      /*.reserved=*/{},
      /*.args=*/{0, 16, 0},
  }};
  loom_named_attr_t inner_entries[2] = {
      {
          /*.name_id=*/inner_zeta_id,
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64_array(original_values, 3),
      },
      {
          /*.name_id=*/inner_alpha_id,
          /*.reserved=*/{},
          /*.value=*/loom_attr_predicate_list(original_predicates, 1),
      },
  };
  loom_named_attr_t outer_entries[2] = {
      {
          /*.name_id=*/outer_zeta_id,
          /*.reserved=*/{},
          /*.value=*/
          loom_make_canonical_attr_dict(inner_entries,
                                        IREE_ARRAYSIZE(inner_entries)),
      },
      {/*.name_id=*/outer_axis_id, /*.reserved=*/{},
       /*.value=*/loom_attr_i64(4)},
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(outer_entries, IREE_ARRAYSIZE(outer_entries)),
      &attr));

  // Mutate the caller-owned buffers after construction. The canonical dict
  // must keep its own arena-owned snapshots.
  inner_entries[0].name_id = outer_axis_id;
  inner_entries[0].value = loom_attr_i64(99);
  outer_entries[0].name_id = outer_axis_id;
  outer_entries[0].value = loom_attr_i64(42);
  original_values[0] = 1234;
  original_predicates[0].kind = LOOM_PREDICATE_EQ;

  EXPECT_EQ(attr.count, 2);
  ASSERT_NE(attr.dict_entries, nullptr);
  EXPECT_EQ(attr.dict_entries[0].name_id, outer_axis_id);
  EXPECT_EQ(attr.dict_entries[0].value.i64, 4);
  EXPECT_EQ(attr.dict_entries[1].name_id, outer_zeta_id);
  ASSERT_EQ(attr.dict_entries[1].value.kind, LOOM_ATTR_DICT);

  loom_attribute_t nested = attr.dict_entries[1].value;
  EXPECT_EQ(nested.count, 2);
  ASSERT_NE(nested.dict_entries, nullptr);
  EXPECT_EQ(nested.dict_entries[0].name_id, inner_alpha_id);
  EXPECT_EQ(nested.dict_entries[0].reserved, 0u);
  EXPECT_EQ(nested.dict_entries[0].value.kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_NE(nested.dict_entries[0].value.predicate_list, original_predicates);
  EXPECT_EQ(nested.dict_entries[0].value.count, 1);
  EXPECT_EQ(nested.dict_entries[0].value.predicate_list[0].kind,
            LOOM_PREDICATE_RANGE);
  EXPECT_EQ(nested.dict_entries[1].name_id, inner_zeta_id);
  EXPECT_EQ(nested.dict_entries[1].reserved, 0u);
  EXPECT_EQ(nested.dict_entries[1].value.kind, LOOM_ATTR_I64_ARRAY);
  ASSERT_NE(nested.dict_entries[1].value.i64_array, original_values);
  ASSERT_EQ(nested.dict_entries[1].value.count, 3);
  EXPECT_EQ(nested.dict_entries[1].value.i64_array[0], 7);
  EXPECT_EQ(nested.dict_entries[1].value.i64_array[1], 8);
  EXPECT_EQ(nested.dict_entries[1].value.i64_array[2], 9);

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRejectsDuplicateKeys) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("axis"), &key_id));

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/key_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/key_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };

  loom_attribute_t attr = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_make_canonical_attr_dict(
          module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
          &attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRejectsNonEmptyNullEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t attr = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_make_canonical_attr_dict(
          module, loom_make_named_attr_slice(/*entries=*/NULL, /*count=*/1),
          &attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictNormalizesEqualityAndHash) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t zeta_first_entries[2] = {
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_named_attr_t alpha_first_entries[2] = {
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
  };

  loom_attribute_t zeta_first_attr = {0};
  loom_attribute_t alpha_first_attr = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(zeta_first_entries,
                                 IREE_ARRAYSIZE(zeta_first_entries)),
      &zeta_first_attr));
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(alpha_first_entries,
                                 IREE_ARRAYSIZE(alpha_first_entries)),
      &alpha_first_attr));

  EXPECT_TRUE(loom_attribute_equal(&zeta_first_attr, &alpha_first_attr));
  EXPECT_EQ(loom_attribute_hash(&zeta_first_attr),
            loom_attribute_hash(&alpha_first_attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRejectsUnknownKeyStringId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_named_attr_t entries[1] = {{
      /*.name_id=*/99,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64(1),
  }};

  loom_attribute_t attr = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_make_canonical_attr_dict(
          module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
          &attr));

  loom_module_free(module);
}

TEST_F(ModuleTest,
       ReplaceCanonicalAttrDictAppliesAddReplaceRemoveAndKeepsSortedOrder) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t beta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("beta"), &beta_id));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));

  loom_named_attr_t base_entries[2] = {
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(3)},
  };
  loom_named_attr_update_t updates[3] = {
      loom_named_attr_replace(zeta_id, loom_attr_i64(30)),
      loom_named_attr_remove(alpha_id),
      loom_named_attr_replace(beta_id, loom_attr_i64(20)),
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_replace_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(base_entries, IREE_ARRAYSIZE(base_entries)),
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates)),
      &attr));

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  ASSERT_EQ(attr.kind, LOOM_ATTR_DICT);
  ASSERT_EQ(attr.count, 2u);
  ASSERT_NE(attr.dict_entries, nullptr);
  EXPECT_EQ(attr.dict_entries[0].name_id, beta_id);
  EXPECT_EQ(loom_attr_as_i64(attr.dict_entries[0].value), 20);
  EXPECT_EQ(attr.dict_entries[1].name_id, zeta_id);
  EXPECT_EQ(loom_attr_as_i64(attr.dict_entries[1].value), 30);

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceCanonicalAttrDictRejectsDuplicateUpdateKeysByNameId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t base_entries[1] = {
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_named_attr_update_t updates[2] = {
      loom_named_attr_replace(alpha_id, loom_attr_i64(2)),
      loom_named_attr_remove(alpha_id),
  };

  loom_attribute_t attr = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_replace_canonical_attr_dict(
          module,
          loom_make_named_attr_slice(base_entries,
                                     IREE_ARRAYSIZE(base_entries)),
          loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates)),
          &attr));

  loom_module_free(module);
}

TEST_F(ModuleTest,
       ReplaceCanonicalAttrDictRecursivelyCanonicalizesNestedUpdateValues) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t outer_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("outer"), &outer_id));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));

  loom_named_attr_t nested_entries[2] = {
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_named_attr_t base_entries[1] = {
      {/*.name_id=*/outer_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
  };
  loom_named_attr_update_t updates[1] = {
      loom_named_attr_replace(
          outer_id, loom_make_canonical_attr_dict(
                        nested_entries, IREE_ARRAYSIZE(nested_entries))),
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_replace_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(base_entries, IREE_ARRAYSIZE(base_entries)),
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates)),
      &attr));

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  ASSERT_EQ(attr.count, 1u);
  ASSERT_NE(attr.dict_entries, nullptr);
  ASSERT_EQ(attr.dict_entries[0].name_id, outer_id);
  ASSERT_EQ(attr.dict_entries[0].value.kind, LOOM_ATTR_DICT);
  loom_attribute_t nested = attr.dict_entries[0].value;
  ASSERT_EQ(nested.count, 2u);
  ASSERT_NE(nested.dict_entries, nullptr);
  EXPECT_NE(nested.dict_entries, nested_entries);
  EXPECT_EQ(nested.dict_entries[0].name_id, alpha_id);
  EXPECT_EQ(loom_attr_as_i64(nested.dict_entries[0].value), 1);
  EXPECT_EQ(nested.dict_entries[1].name_id, zeta_id);
  EXPECT_EQ(loom_attr_as_i64(nested.dict_entries[1].value), 2);

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsUnsortedInput) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_attribute_t attr =
      loom_make_canonical_attr_dict(entries, IREE_ARRAYSIZE(entries));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_verify_canonical_attr_dict(module, attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsDuplicateKeys) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("axis"), &key_id));

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/key_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/key_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_attribute_t attr =
      loom_make_canonical_attr_dict(entries, IREE_ARRAYSIZE(entries));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_verify_canonical_attr_dict(module, attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsNonEmptyNullEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t attr =
      loom_make_canonical_attr_dict(/*entries=*/NULL, /*count=*/1);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_verify_canonical_attr_dict(module, attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsEmptyDictWithNonNullEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("axis"), &key_id));
  loom_named_attr_t entries[1] = {{
      /*.name_id=*/key_id,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64(0),
  }};
  loom_attribute_t attr = {};
  attr.kind = LOOM_ATTR_DICT;
  attr.count = 0;
  attr.dict_entries = entries;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_verify_canonical_attr_dict(module, attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, InternStringRejectsInvalidSentinelIdButKeepsDedupWorking) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t existing_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &existing_id));

  module->strings.count = LOOM_STRING_ID_INVALID;

  loom_string_id_t duplicate_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &duplicate_id));
  EXPECT_EQ(duplicate_id, existing_id);

  loom_string_id_t new_id = LOOM_STRING_ID_INVALID;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_intern_string(module, IREE_SV("world"), &new_id));

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Type interning
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, InternScalarType) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t interned = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned));
  EXPECT_EQ(loom_type_kind(interned), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(interned), LOOM_SCALAR_TYPE_F32);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternTypeDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t interned1 = {0};
  loom_type_t interned2 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned1));
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned2));
  // Same type interned twice should produce identical results.
  EXPECT_EQ(interned1.header, interned2.header);
  EXPECT_EQ(module->types.count, 1u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternTypeIdReturnsCanonicalId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_id_t f32_id = LOOM_TYPE_ID_INVALID;
  loom_type_id_t duplicate_f32_id = LOOM_TYPE_ID_INVALID;
  loom_type_id_t i32_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(module, f32, &f32_id));
  IREE_ASSERT_OK(loom_module_intern_type_id(module, f32, &duplicate_f32_id));
  IREE_ASSERT_OK(loom_module_intern_type_id(module, i32, &i32_id));

  EXPECT_EQ(f32_id, 0u);
  EXPECT_EQ(duplicate_f32_id, f32_id);
  EXPECT_EQ(i32_id, 1u);
  EXPECT_EQ(module->types.count, 2u);
  EXPECT_TRUE(loom_type_equal(module->types.entries[f32_id], f32));
  EXPECT_TRUE(loom_type_equal(module->types.entries[i32_id], i32));
  loom_module_free(module);
}

TEST_F(ModuleTest, InternDifferentTypes) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t interned_f32 = {0};
  loom_type_t interned_i32 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned_f32));
  IREE_ASSERT_OK(loom_module_intern_type(module, i32, &interned_i32));
  EXPECT_NE(interned_f32.header, interned_i32.header);
  EXPECT_EQ(module->types.count, 2u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternShapedType) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t tile_4x4_f32 =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(4), 0);
  loom_type_t interned1 = {0};
  loom_type_t interned2 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, tile_4x4_f32, &interned1));
  IREE_ASSERT_OK(loom_module_intern_type(module, tile_4x4_f32, &interned2));
  EXPECT_EQ(interned1.header, interned2.header);
  EXPECT_EQ(interned1.dims[0], interned2.dims[0]);
  EXPECT_EQ(interned1.dims[1], interned2.dims[1]);
  ASSERT_EQ(module->types.count, 2u);
  EXPECT_TRUE(loom_type_equal(module->types.entries[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(loom_type_equal(module->types.entries[1], interned1));
  loom_module_free(module);
}

TEST_F(ModuleTest, InternFunctionTypeDedupsStructurallyAndOwnsPayload) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t arg_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t source_a = {0};
  IREE_ASSERT_OK(loom_type_function_build(&arg_type, 1, &result_type, 1,
                                          iree_allocator_system(), &source_a));
  const loom_func_type_data_t* source_a_data = loom_type_func_data(source_a);

  loom_type_t source_b = {0};
  IREE_ASSERT_OK(loom_type_function_build(&arg_type, 1, &result_type, 1,
                                          iree_allocator_system(), &source_b));

  loom_type_t interned_a = {0};
  loom_type_t interned_b = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, source_a, &interned_a));
  IREE_ASSERT_OK(loom_module_intern_type(module, source_b, &interned_b));

  EXPECT_EQ(module->types.count, 3u);
  EXPECT_TRUE(loom_type_equal(interned_a, interned_b));
  EXPECT_EQ(loom_type_hash(interned_a), loom_type_hash(interned_b));
  EXPECT_NE(loom_type_func_data(interned_a), source_a_data);

  iree_allocator_free(iree_allocator_system(), (void*)source_a_data);
  iree_allocator_free(iree_allocator_system(),
                      (void*)loom_type_func_data(source_b));

  ASSERT_EQ(loom_type_func_arg_count(interned_a), 1u);
  ASSERT_EQ(loom_type_func_result_count(interned_a), 1u);
  EXPECT_TRUE(
      loom_type_equal(loom_type_func_arg_types(interned_a)[0], arg_type));
  EXPECT_TRUE(
      loom_type_equal(loom_type_func_result_types(interned_a)[0], result_type));

  loom_module_free(module);
}

TEST_F(ModuleTest, InternFunctionTypeDirectAndPackedFormsDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t arg_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
  };
  loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };

  loom_type_t direct_interned = {0};
  IREE_ASSERT_OK(loom_module_intern_function_type(
      module, arg_types, IREE_ARRAYSIZE(arg_types), result_types,
      IREE_ARRAYSIZE(result_types), &direct_interned));
  iree_host_size_t allocation_size = module->arena.total_allocation_size;

  loom_type_t packed_source = {0};
  IREE_ASSERT_OK(loom_type_function_build(
      arg_types, IREE_ARRAYSIZE(arg_types), result_types,
      IREE_ARRAYSIZE(result_types), iree_allocator_system(), &packed_source));

  loom_type_t packed_interned = {0};
  IREE_ASSERT_OK(
      loom_module_intern_type(module, packed_source, &packed_interned));

  EXPECT_EQ(module->types.count, 4u);
  EXPECT_TRUE(loom_type_equal(direct_interned, packed_interned));
  EXPECT_EQ(loom_type_func_data(direct_interned),
            loom_type_func_data(packed_interned));
  EXPECT_EQ(module->arena.total_allocation_size, allocation_size);

  iree_allocator_free(iree_allocator_system(),
                      (void*)loom_type_func_data(packed_source));
  loom_module_free(module);
}

TEST_F(ModuleTest, InternDialectTypeCopiesTemporaryParamsAndPreservesNameId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("dialect.type"), &name_id));

  loom_type_t params[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t source =
      loom_type_dialect(name_id, IREE_ARRAYSIZE(params), params);

  loom_type_t interned = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, source, &interned));

  params[0] = loom_type_scalar(LOOM_SCALAR_TYPE_I64);

  EXPECT_EQ(loom_type_dialect_name_id(interned), name_id);
  ASSERT_EQ(loom_type_dialect_param_count(interned), 2u);
  EXPECT_TRUE(loom_type_equal(loom_type_dialect_params(interned)[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(
      loom_type_equal(loom_type_dialect_params(interned)[1],
                      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                                          loom_dim_pack_static(4), 0)));

  loom_type_t duplicate_params[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t duplicate_source = loom_type_dialect(
      name_id, IREE_ARRAYSIZE(duplicate_params), duplicate_params);
  loom_type_t duplicate_interned = {0};
  IREE_ASSERT_OK(
      loom_module_intern_type(module, duplicate_source, &duplicate_interned));

  EXPECT_EQ(module->types.count, 4u);
  EXPECT_TRUE(loom_type_equal(interned, duplicate_interned));

  loom_module_free(module);
}

TEST_F(ModuleTest, InternTypeRejectsInvalidSentinelIdButKeepsDedupWorking) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t interned_f32 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned_f32));

  module->types.count = LOOM_TYPE_ID_INVALID;

  loom_type_t duplicate_f32 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &duplicate_f32));
  EXPECT_EQ(duplicate_f32.header, interned_f32.header);

  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t new_i32 = {0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_module_intern_type(module, i32, &new_i32));

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Block operations
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, BlockAppendOp) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  // Allocate a dummy op.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op));
  memset(op, 0, sizeof(loom_op_t));
  op->kind = 0x0100;

  IREE_ASSERT_OK(loom_block_append_op(module, block, op));
  EXPECT_EQ(block->op_count, 1);
  EXPECT_EQ(block->first_op, op);
  EXPECT_EQ(block->last_op, op);
  EXPECT_EQ(op->parent_block, block);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockAppendOrdering) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* ops[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_arena_allocate(&module->arena, sizeof(loom_op_t),
                                       (void**)&ops[i]));
    memset(ops[i], 0, sizeof(loom_op_t));
    ops[i]->kind = (loom_op_kind_t)(0x0100 + i);
    IREE_ASSERT_OK(loom_block_append_op(module, block, ops[i]));
  }

  EXPECT_EQ(block->op_count, 3);
  EXPECT_EQ(loom_block_op(block, 0), ops[0]);
  EXPECT_EQ(loom_block_op(block, 1), ops[1]);
  EXPECT_EQ(loom_block_op(block, 2), ops[2]);
  EXPECT_EQ(block->first_op, ops[0]);
  EXPECT_EQ(block->last_op, ops[2]);
  EXPECT_EQ(ops[0]->next_op, ops[1]);
  EXPECT_EQ(ops[1]->prev_op, ops[0]);
  EXPECT_EQ(ops[1]->next_op, ops[2]);
  EXPECT_EQ(ops[2]->prev_op, ops[1]);
  EXPECT_LT(ops[0]->block_ordinal, ops[1]->block_ordinal);
  EXPECT_LT(ops[1]->block_ordinal, ops[2]->block_ordinal);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockInsertAtBeginning) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* op_a = NULL;
  loom_op_t* op_b = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_a));
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_b));
  memset(op_a, 0, sizeof(loom_op_t));
  memset(op_b, 0, sizeof(loom_op_t));

  IREE_ASSERT_OK(loom_block_append_op(module, block, op_a));
  IREE_ASSERT_OK(loom_block_insert_op(module, block, 0, op_b));

  EXPECT_EQ(block->op_count, 2);
  EXPECT_EQ(loom_block_op(block, 0), op_b);
  EXPECT_EQ(loom_block_op(block, 1), op_a);
  EXPECT_EQ(block->first_op, op_b);
  EXPECT_EQ(block->last_op, op_a);
  EXPECT_EQ(op_b->next_op, op_a);
  EXPECT_EQ(op_a->prev_op, op_b);
  EXPECT_LT(op_b->block_ordinal, op_a->block_ordinal);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockFindOp) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* op_a = NULL;
  loom_op_t* op_b = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_a));
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_b));
  memset(op_a, 0, sizeof(loom_op_t));
  memset(op_b, 0, sizeof(loom_op_t));

  IREE_ASSERT_OK(loom_block_append_op(module, block, op_a));
  IREE_ASSERT_OK(loom_block_append_op(module, block, op_b));

  EXPECT_EQ(loom_block_find_op(block, op_a), 0);
  EXPECT_EQ(loom_block_find_op(block, op_b), 1);

  // Not-found case.
  loom_op_t dummy = {0};
  EXPECT_EQ(loom_block_find_op(block, &dummy), IREE_HOST_SIZE_MAX);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockAppendSupportsMoreThanUint16Ops) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  constexpr uint32_t kOpCount = 70000;
  for (uint32_t i = 0; i < kOpCount; ++i) {
    loom_op_t* op = NULL;
    IREE_ASSERT_OK(
        iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op));
    memset(op, 0, sizeof(loom_op_t));
    op->kind = (loom_op_kind_t)i;
    IREE_ASSERT_OK(loom_block_append_op(module, block, op));
  }

  EXPECT_EQ(block->op_count, kOpCount);

  uint32_t expected_kind = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    EXPECT_EQ(op->kind, (loom_op_kind_t)expected_kind++);
    if (op->next_op) {
      EXPECT_LT(op->block_ordinal, op->next_op->block_ordinal);
    }
  }
  EXPECT_EQ(expected_kind, kOpCount);
  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Size hints
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, SizeHints) {
  loom_module_size_hints_t hints = {
      /*.value_count=*/100,
      /*.string_count=*/50,
      /*.type_count=*/20,
      /*.symbol_count=*/10,
  };
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      &hints, iree_allocator_system(),
                                      &module));
  // Capacities should be at least hint * growth_factor.
  EXPECT_GE(module->values.capacity, 100u);
  EXPECT_GE(module->strings.capacity, 50u);
  EXPECT_GE(module->types.capacity, 20u);
  EXPECT_GE(module->symbols.capacity, 10u);
  loom_module_free(module);
}

TEST_F(ModuleTest, AddLocationRejectsWrappedIdZero) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  module->locations.count = (iree_host_size_t)UINT32_MAX + 1;

  loom_location_id_t id = LOOM_LOCATION_UNKNOWN;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_add_location(
          module,
          loom_location_file_range(/*source_id=*/0, /*start_line=*/1,
                                   /*start_col=*/1, /*end_line=*/1,
                                   /*end_col=*/2),
          &id));

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Encoding table
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, AddEncodingBasic) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  // Intern the encoding name.
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));

  // Intern a param name and build a named attribute.
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &block_id));
  loom_named_attr_t param = {
      /*.name_id=*/block_id,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64(32),
  };

  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/&param,
  };

  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
  EXPECT_EQ(encoding_id, 1);
  EXPECT_EQ(module->encodings.count, 1u);

  // Look it up.
  const loom_encoding_t* found = loom_module_encoding(module, encoding_id);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name_id, name_id);
  EXPECT_EQ(found->attribute_count, 1);
  EXPECT_EQ(found->attributes[0].name_id, block_id);
  EXPECT_EQ(found->attributes[0].value.i64, 32);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("dense"), &name_id));

  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/0,
  };

  uint16_t id1 = 0, id2 = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &id1));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &id2));
  EXPECT_EQ(id1, id2);
  EXPECT_EQ(module->encodings.count, 1u);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingDedupStructuralParamsAndBackfillsAlias) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_string_id_t shape_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("shape"), &shape_id));
  loom_string_id_t alias_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("enc"), &alias_id));

  int64_t shape_a[] = {16, 32};
  int64_t shape_b[] = {16, 32};
  loom_named_attr_t attrs_a[] = {{
      /*.name_id=*/shape_id,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64_array(shape_a, IREE_ARRAYSIZE(shape_a)),
  }};
  loom_named_attr_t attrs_b[] = {{
      /*.name_id=*/shape_id,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64_array(shape_b, IREE_ARRAYSIZE(shape_b)),
  }};

  loom_encoding_t plain = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/attrs_a,
  };
  loom_encoding_t aliased = {
      /*.name_id=*/name_id,
      /*.alias_id=*/alias_id,
      /*.attribute_count=*/1,
      /*.reserved=*/{},       /*.attributes=*/attrs_b,
  };

  uint16_t plain_id = 0;
  uint16_t aliased_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &plain, &plain_id));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &aliased, &aliased_id));

  EXPECT_EQ(plain_id, aliased_id);
  ASSERT_EQ(module->encodings.count, 1u);
  const loom_encoding_t* encoding = loom_module_encoding(module, plain_id);
  ASSERT_NE(encoding, nullptr);
  EXPECT_EQ(encoding->alias_id, alias_id);
  ASSERT_EQ(encoding->attribute_count, 1u);
  EXPECT_EQ(encoding->attributes[0].value.count, 2u);
  ASSERT_NE(encoding->attributes[0].value.i64_array, nullptr);
  EXPECT_EQ(encoding->attributes[0].value.i64_array[0], 16);
  EXPECT_EQ(encoding->attributes[0].value.i64_array[1], 32);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingRejectsDuplicateAliasForDifferentEncodings) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t q8_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("q8_0"), &q8_name_id));
  loom_string_id_t dense_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("dense"), &dense_name_id));
  loom_string_id_t alias_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("enc"), &alias_id));

  loom_encoding_t q8_encoding = {
      /*.name_id=*/q8_name_id,
      /*.alias_id=*/alias_id,
  };
  uint16_t q8_encoding_id = 0;
  IREE_ASSERT_OK(
      loom_module_add_encoding(module, &q8_encoding, &q8_encoding_id));

  loom_encoding_t dense_encoding = {
      /*.name_id=*/dense_name_id,
      /*.alias_id=*/alias_id,
  };
  uint16_t dense_encoding_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_add_encoding(module, &dense_encoding, &dense_encoding_id));
  EXPECT_EQ(module->encodings.count, 1u);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingDifferentParams) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &block_id));

  // Same name, different block size — two distinct entries.
  loom_named_attr_t param32 = {/*.name_id=*/block_id, /*.reserved=*/{},
                               /*.value=*/loom_attr_i64(32)};
  loom_named_attr_t param64 = {/*.name_id=*/block_id, /*.reserved=*/{},
                               /*.value=*/loom_attr_i64(64)};

  loom_encoding_t enc32 = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/&param32,
  };
  loom_encoding_t enc64 = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/&param64,
  };

  uint16_t id32 = 0, id64 = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &enc32, &id32));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &enc64, &id64));
  EXPECT_NE(id32, id64);
  EXPECT_EQ(module->encodings.count, 2u);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingRejectsUnknownFamilyWhenRegistryIsPopulated) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t unknown_name_id = 0;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("mystery_q"),
                                           &unknown_name_id));
  loom_encoding_t encoding = {
      /*.name_id=*/unknown_name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_add_encoding(module, &encoding, &encoding_id));
  EXPECT_EQ(module->encodings.count, 0u);

  loom_module_free(module);
}

TEST_F(ModuleTest, EncodingVtableLookupReturnsRegisteredFamily) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));

  EXPECT_EQ(loom_module_encoding_vtable(module, encoding_id),
            &kQ8_0EncodingVtable);

  loom_module_free(module);
}

TEST_F(ModuleTest, EncodingLookupOutOfRange) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  EXPECT_EQ(loom_module_encoding(module, 0), nullptr);
  EXPECT_EQ(loom_module_encoding(module, 1), nullptr);
  EXPECT_EQ(loom_module_encoding(module, UINT16_MAX), nullptr);
  EXPECT_EQ(loom_module_encoding_vtable(module, 0), nullptr);
  EXPECT_EQ(loom_module_encoding_vtable(module, 1), nullptr);
  EXPECT_EQ(loom_module_encoding_vtable(module, UINT16_MAX), nullptr);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
