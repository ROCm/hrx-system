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
#include "loom/ops/test/types.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kQ8_0EncodingParameters[] = {{
    /*.name=*/LOOM_BSTRING_REF(5, "block"),
    /*.attr_kind=*/LOOM_ATTR_I64,
    /*.flags=*/LOOM_ATTR_OPTIONAL,
}};
static const loom_encoding_family_descriptor_t kQ8_0EncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(4, "q8_0"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/IREE_ARRAYSIZE(kQ8_0EncodingParameters),
    /*.parameter_descriptors=*/kQ8_0EncodingParameters,
};
static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.descriptor=*/&kQ8_0EncodingDescriptor,
};

static const loom_encoding_family_descriptor_t kQ6KEncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(4, "q6_k"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};
static const loom_encoding_vtable_t kQ6KEncodingVtable = {
    /*.descriptor=*/&kQ6KEncodingDescriptor,
};

static const loom_encoding_family_descriptor_t kDenseEncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(5, "dense"),
    /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
    /*.family_flags=*/LOOM_ENCODING_FAMILY_IMPLICIT_SHAPED_ATTACHMENT,
};
static const loom_encoding_vtable_t kDenseEncodingVtable = {
    /*.descriptor=*/&kDenseEncodingDescriptor,
};

class ModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ6KEncodingVtable));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kDenseEncodingVtable));
    iree_host_size_t parameterized_attr_count = 0;
    const loom_parameterized_attr_descriptor_t* parameterized_attrs =
        loom_test_dialect_parameterized_attrs(&parameterized_attr_count);
    IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
        &context_, LOOM_DIALECT_TEST, parameterized_attrs,
        parameterized_attr_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

typedef struct AttributeValueRefCapture {
  loom_value_id_t values[4];
  iree_host_size_t count;
} AttributeValueRefCapture;

static iree_status_t CaptureAttributeValueRef(loom_value_id_t value_id,
                                              void* user_data) {
  AttributeValueRefCapture* capture = (AttributeValueRefCapture*)user_data;
  if (capture->count >= IREE_ARRAYSIZE(capture->values)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "attribute value reference capture is full");
  }
  capture->values[capture->count++] = value_id;
  return iree_ok_status();
}

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

TEST_F(ModuleTest, AppendSourcePreservesInsertionOrder) {
  loom_module_size_hints_t hints = {};
  hints.source_count = 2;
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      &hints, iree_allocator_system(),
                                      &module));

  loom_source_id_t first_id = LOOM_SOURCE_ID_INVALID;
  loom_source_id_t second_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_append_source(module, IREE_SV("a.loom"), &first_id));
  IREE_ASSERT_OK(
      loom_module_append_source(module, IREE_SV("b.loom"), &second_id));

  EXPECT_EQ(first_id, 0u);
  EXPECT_EQ(second_id, 1u);
  EXPECT_GE(module->sources.capacity, 2u);
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
      &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_DICT, 0,
      (loom_symbol_ref_t){0, keep_a_symbol_id},
      loom_make_named_attr_slice(keep_a_dict, IREE_ARRAYSIZE(keep_a_dict)),
      LOOM_LOCATION_UNKNOWN, &keep_a_op));
  loom_op_t* keep_b_op = NULL;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, (loom_symbol_ref_t){0, keep_b_symbol_id},
      loom_make_named_attr_slice(NULL, 0), LOOM_LOCATION_UNKNOWN, &keep_b_op));
  loom_symbol_ref_t dependency_refs[] = {
      {0, keep_b_symbol_id},
      {0, keep_a_symbol_id},
      {0, keep_b_symbol_id},
  };
  loom_symbol_ref_t availability_refs[] = {{0, keep_a_symbol_id}};
  loom_op_t* refs_op = NULL;
  IREE_ASSERT_OK(loom_test_symbol_array_attrs_build(
      &builder, LOOM_TEST_SYMBOL_ARRAY_ATTRS_BUILD_FLAG_HAS_AVAILABLE,
      loom_make_symbol_ref_array(dependency_refs,
                                 IREE_ARRAYSIZE(dependency_refs)),
      loom_make_symbol_ref_array(availability_refs,
                                 IREE_ARRAYSIZE(availability_refs)),
      LOOM_LOCATION_UNKNOWN, &refs_op));

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
  loom_symbol_ref_array_t dependencies =
      loom_test_symbol_array_attrs_dependencies(refs_op);
  ASSERT_EQ(dependencies.count, 3u);
  EXPECT_EQ(dependencies.values[0].symbol_id, 1u);
  EXPECT_EQ(dependencies.values[1].symbol_id, 0u);
  EXPECT_EQ(dependencies.values[2].symbol_id, 1u);
  loom_symbol_ref_array_t available =
      loom_test_symbol_array_attrs_available(refs_op);
  ASSERT_EQ(available.count, 1u);
  EXPECT_EQ(available.values[0].symbol_id, 0u);

  loom_module_free(module);
}

TEST_F(ModuleTest, CompactSymbolsRebuildsEncodingInternTable) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t drop_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("drop"), &drop_name));
  uint16_t drop_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, drop_name, &drop_symbol_id));
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("target"), &target_name));
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, target_name, &target_symbol_id));
  ASSERT_EQ(drop_symbol_id, 0u);
  ASSERT_EQ(target_symbol_id, 1u);

  loom_string_id_t encoding_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("q8_0"), &encoding_name));
  loom_string_id_t parameter_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &parameter_name));
  loom_named_attr_t parameter = {
      /*.name_id=*/parameter_name,
      /*.reserved=*/{},
      /*.value=*/loom_attr_symbol((loom_symbol_ref_t){0, target_symbol_id}),
  };
  loom_encoding_t encoding = {
      /*.name_id=*/encoding_name,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.family=*/{},
      /*.attributes=*/&parameter,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  iree_host_size_t removed_count = 0;
  IREE_ASSERT_OK(
      loom_module_compact_symbols(module, &scratch_arena, &removed_count));
  iree_arena_deinitialize(&scratch_arena);

  EXPECT_EQ(removed_count, 1u);
  const loom_encoding_t* compacted = loom_module_encoding(module, encoding_id);
  ASSERT_NE(compacted, nullptr);
  ASSERT_EQ(compacted->attribute_count, 1u);
  EXPECT_EQ(loom_attr_as_symbol(compacted->attributes[0].value).symbol_id, 0u);

  parameter.value = loom_attr_symbol((loom_symbol_ref_t){0, 0});
  uint16_t duplicate_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &duplicate_id));
  EXPECT_EQ(duplicate_id, encoding_id);
  EXPECT_EQ(module->encodings.count, 1u);

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

TEST_F(ModuleTest, CompactSymbolsPreservesParameterizedTypeSymbolOrdinals) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t drop_before_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("drop_before"),
                                           &drop_before_name));
  uint16_t drop_before_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, drop_before_name, &drop_before_symbol_id));

  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("target"), &target_name));
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, target_name, &target_symbol_id));

  loom_string_id_t drop_after_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("drop_after"),
                                           &drop_after_name));
  uint16_t drop_after_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, drop_after_name, &drop_after_symbol_id));

  ASSERT_EQ(drop_before_symbol_id, 0u);
  ASSERT_EQ(target_symbol_id, 1u);
  ASSERT_EQ(drop_after_symbol_id, 2u);

  loom_type_id_t bf16_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &bf16_type_id));
  loom_type_t matrix_type = {};
  IREE_ASSERT_OK(loom_test_matrix_type_make(
      module, LOOM_TEST_MATRIX_TYPE_BUILD_FLAG_HAS_TARGET, bf16_type_id,
      LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP, 16,
      (loom_symbol_ref_t){0, target_symbol_id}, &matrix_type));

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  iree_host_size_t removed_count = 0;
  IREE_ASSERT_OK(
      loom_module_compact_symbols(module, &scratch_arena, &removed_count));
  iree_arena_deinitialize(&scratch_arena);

  EXPECT_EQ(removed_count, 1u);
  ASSERT_EQ(module->symbols.count, 2u);
  EXPECT_EQ(module->symbols.entries[0].name_id, LOOM_STRING_ID_INVALID);
  EXPECT_EQ(loom_module_find_symbol(module, target_name), target_symbol_id);
  EXPECT_EQ(loom_module_find_symbol(module, drop_after_name),
            LOOM_SYMBOL_ID_INVALID);
  ASSERT_TRUE(loom_test_matrix_type_has_target(matrix_type));
  EXPECT_EQ(loom_test_matrix_type_target(matrix_type).symbol_id,
            target_symbol_id);

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
  EXPECT_EQ(loom_value_def_block(loom_module_value(module, arg_id)), entry);
  EXPECT_EQ(loom_value_def_index(loom_module_value(module, arg_id)), 0u);
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

TEST_F(ModuleTest, BlockRemoveArgRejectsPredicateAttributeUses) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("predicate"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  const loom_symbol_ref_t callee = {
      /*.module_id=*/0,
      /*.symbol_id=*/symbol_id,
  };
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* function_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0, callee,
      &i32_type, /*arg_types_count=*/1, /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
      &function_op));

  loom_block_t* entry_block =
      loom_region_entry_block(loom_test_func_body(function_op));
  const loom_value_id_t argument = loom_block_arg_id(entry_block, 0);
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_EQ,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, 0},
      /*.reserved=*/{},
      /*.args=*/{argument, 3, 0},
  };
  loom_op_attrs(function_op)[3] = loom_attr_predicate_list(&predicate, 1);
  IREE_ASSERT_OK(loom_module_compute_uses(module));

  EXPECT_TRUE(loom_module_value_has_predicate_attribute_uses(module, argument));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_block_remove_arg(module, entry_block, 0));
  EXPECT_EQ(entry_block->arg_count, 1u);
  EXPECT_EQ(loom_block_arg_id(entry_block, 0), argument);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Source comments
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, AttachedCommentsUseOneModuleOwnedStorageSpan) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_block_t* block = loom_module_block(module);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, block, &builder);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &op));

  char op_head[] = " op";
  char op_tail[] = " operation";
  iree_string_view_t op_comments[] = {
      iree_make_string_view(op_head, sizeof(op_head) - 1),
      iree_string_view_empty(),
      iree_make_string_view(op_tail, sizeof(op_tail) - 1),
  };
  IREE_ASSERT_OK(loom_module_attach_op_comments(module, op, op_comments,
                                                IREE_ARRAYSIZE(op_comments)));

  char block_head[] = " block";
  char block_tail[] = " tail";
  iree_string_view_t block_comments[] = {
      iree_make_string_view(block_head, sizeof(block_head) - 1),
      iree_string_view_empty(),
      iree_make_string_view(block_tail, sizeof(block_tail) - 1),
  };
  const iree_host_size_t used_size_before = module->arena.used_allocation_size;
  IREE_ASSERT_OK(loom_module_attach_block_comments(
      module, block, block_comments, IREE_ARRAYSIZE(block_comments)));
  const iree_host_size_t block_storage_size =
      sizeof(block_comments) + sizeof(block_head) - 1 + sizeof(block_tail) - 1;
  EXPECT_EQ(module->arena.used_allocation_size - used_size_before,
            iree_host_align(block_storage_size, iree_max_align_t));

  memset(op_head, '#', sizeof(op_head) - 1);
  memset(op_tail, '#', sizeof(op_tail) - 1);
  memset(op_comments, 0, sizeof(op_comments));
  memset(block_head, '#', sizeof(block_head) - 1);
  memset(block_tail, '#', sizeof(block_tail) - 1);
  memset(block_comments, 0, sizeof(block_comments));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* stored_comments =
      loom_module_op_comments(module, op, &comment_count);
  ASSERT_EQ(comment_count, 3u);
  ASSERT_NE(stored_comments, nullptr);
  EXPECT_EQ(stored_comments[0].data,
            reinterpret_cast<const char*>(stored_comments + comment_count));
  EXPECT_TRUE(iree_string_view_equal(stored_comments[0], IREE_SV(" op")));
  EXPECT_EQ(stored_comments[1].data, nullptr);
  EXPECT_EQ(stored_comments[1].size, 0u);
  EXPECT_EQ(stored_comments[2].data,
            stored_comments[0].data + stored_comments[0].size);
  EXPECT_TRUE(
      iree_string_view_equal(stored_comments[2], IREE_SV(" operation")));

  stored_comments = loom_module_block_comments(module, block, &comment_count);
  ASSERT_EQ(comment_count, 3u);
  ASSERT_NE(stored_comments, nullptr);
  EXPECT_EQ(stored_comments[0].data,
            reinterpret_cast<const char*>(stored_comments + comment_count));
  EXPECT_TRUE(iree_string_view_equal(stored_comments[0], IREE_SV(" block")));
  EXPECT_EQ(stored_comments[1].data, nullptr);
  EXPECT_EQ(stored_comments[1].size, 0u);
  EXPECT_EQ(stored_comments[2].data,
            stored_comments[0].data + stored_comments[0].size);
  EXPECT_TRUE(iree_string_view_equal(stored_comments[2], IREE_SV(" tail")));

  loom_module_free(module);
}

TEST_F(ModuleTest, AttachedCommentsRejectStorageSizeOverflow) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  const iree_string_view_t comments[] = {
      iree_make_string_view("x", IREE_HOST_SIZE_MAX),
  };
  const iree_host_size_t used_size_before = module->arena.used_allocation_size;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_attach_block_comments(module, loom_module_block(module),
                                        comments, IREE_ARRAYSIZE(comments)));
  EXPECT_EQ(module->comments.count, 0u);
  EXPECT_EQ(module->arena.used_allocation_size, used_size_before);

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

  loom_value_t* value = loom_module_value(module, id);
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
  EXPECT_TRUE(loom_type_equal(loom_module_value_type(module, id),
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
  EXPECT_EQ((uintptr_t)loom_module_value(module, id) % 64, 0u)
      << "Value entries must be 64-byte aligned";
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueSegmentGrowthKeepsPointersStable) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  EXPECT_EQ(loom_value_table_capacity(&module->values), 0u);
  loom_value_id_t first_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &first_id));
  loom_value_t* first_value = loom_module_value(module, first_id);
  const iree_host_size_t initial_capacity =
      loom_value_table_capacity(&module->values);
  EXPECT_EQ(initial_capacity, LOOM_VALUE_SEGMENT_CAPACITY);

  const iree_host_size_t value_count =
      LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT *
          LOOM_VALUE_SEGMENT_CAPACITY +
      1;
  for (iree_host_size_t i = 1; i < value_count; ++i) {
    loom_value_id_t id = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
    EXPECT_EQ(id, (loom_value_id_t)i);
  }
  EXPECT_GT(loom_value_table_capacity(&module->values), initial_capacity);
  EXPECT_EQ(module->values.count, value_count);
  EXPECT_EQ(loom_module_value(module, first_id), first_value);

  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    EXPECT_EQ(
        loom_type_kind(loom_module_value_type(module, (loom_value_id_t)i)),
        LOOM_TYPE_SCALAR);
  }
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineUntypedValuesPreservesRangeAndScratchState) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t typed_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &typed_id));
  loom_value_u32_scratch_acquire_zeroed(&module->scratch.values,
                                        module->values.count);
  loom_value_u32_scratch_store(&module->scratch.values, typed_id, 42);

  const iree_host_size_t range_count = LOOM_VALUE_SEGMENT_CAPACITY + 7;
  loom_value_id_t base_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_define_untyped_values(module, range_count, &base_value_id));
  EXPECT_EQ(base_value_id, typed_id + 1);
  EXPECT_EQ(module->values.count, range_count + 1);
  EXPECT_EQ(loom_value_u32_scratch_load(&module->scratch.values, typed_id),
            42u);
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    const loom_value_id_t value_id = base_value_id + (loom_value_id_t)i;
    const loom_value_t* value = loom_module_value(module, value_id);
    EXPECT_EQ(loom_type_kind(value->type), LOOM_TYPE_NONE);
    EXPECT_EQ(value->name_id, LOOM_STRING_ID_INVALID);
    EXPECT_EQ(value->def, loom_value_def_make_none());
    EXPECT_EQ(loom_value_u32_scratch_load(&module->scratch.values, value_id),
              0u);
  }
  loom_value_u32_scratch_release_zeroed(&module->scratch.values);

  loom_value_id_t empty_base_value_id = 0;
  IREE_ASSERT_OK(
      loom_module_define_untyped_values(module, 0, &empty_base_value_id));
  EXPECT_EQ(empty_base_value_id, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(module->values.count, range_count + 1);

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
  loom_value_id_t first_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &first_id));
  const iree_host_size_t initial_capacity =
      loom_value_table_capacity(&module->values);
  ASSERT_EQ(initial_capacity, LOOM_VALUE_SEGMENT_CAPACITY);
  ASSERT_EQ(module->scratch.values.value_table, &module->values);
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, first_id, 3);

  loom_value_id_t last_id = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = module->values.count; i <= initial_capacity; ++i) {
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &last_id));
  }

  EXPECT_GT(loom_value_table_capacity(&module->values), initial_capacity);
  EXPECT_EQ(module->scratch.values.value_table, &module->values);
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
  EXPECT_EQ(loom_value_u32_scratch_load(&module->scratch.values, first_id), 0u);
  EXPECT_EQ(loom_value_u32_scratch_load(&module->scratch.values, second_id),
            0u);
  loom_value_u32_scratch_store(&module->scratch.values, first_id, 0x3u);
  const iree_host_size_t initial_capacity =
      loom_value_table_capacity(&module->values);
  loom_value_id_t last_id = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = module->values.count; i <= initial_capacity; ++i) {
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &last_id));
  }
  EXPECT_EQ(loom_value_u32_scratch_load(&module->scratch.values, last_id), 0u);
  loom_value_u32_scratch_release_zeroed(&module->scratch.values);

  loom_module_value_ordinal_scratch_acquire(module);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, first_id),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, second_id),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_EQ(loom_module_value_ordinal_scratch_lookup(module, last_id),
            LOOM_VALUE_ORDINAL_INVALID);
  loom_module_value_ordinal_scratch_release(module);

  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValuesRejectInvalidSentinelId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  module->values.count = LOOM_VALUE_ID_INVALID;

  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_define_value(module, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                               &id));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_module_define_untyped_values(module, 1, &id));

  loom_module_free(module);
}

TEST_F(ModuleTest, TypeUseTableTracksDynamicDims) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &dim_id));
  EXPECT_FALSE(loom_module_has_active_type_uses(module));
  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim_id), 0);
  loom_value_id_t vector_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, vector_type, &vector_id));

  EXPECT_TRUE(loom_module_value_has_type_uses(module, dim_id));
  EXPECT_TRUE(loom_module_has_active_type_uses(module));
  IREE_ASSERT_OK(loom_module_set_value_type(
      module, vector_id,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), 0)));
  EXPECT_FALSE(loom_module_value_has_type_uses(module, dim_id));
  EXPECT_FALSE(loom_module_has_active_type_uses(module));

  loom_module_free(module);
}

TEST_F(ModuleTest, TypeUseTableCrossesValueSegments) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, index, &dim_id));
  for (uint32_t i = 1; i < LOOM_VALUE_SEGMENT_CAPACITY; ++i) {
    loom_value_id_t padding_id = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &padding_id));
  }

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim_id), 0);
  loom_value_id_t vector_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, vector_type, &vector_id));
  ASSERT_EQ(vector_id, LOOM_VALUE_SEGMENT_CAPACITY);
  ASSERT_TRUE(loom_module_value_has_type_uses(module, dim_id));
  loom_type_use_id_t use_id =
      loom_module_value_first_incoming_type_use(module, dim_id);
  ASSERT_NE(use_id, LOOM_TYPE_USE_ID_INVALID);
  EXPECT_EQ(module->type_uses.records[use_id].user_value_id, vector_id);

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

TEST_F(ModuleTest, ReplaceValueTypeUsesUpdatesRegisterValueType) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_value_id_t old_dim_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &old_dim_id));
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &new_dim_id));
  loom_type_t value_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim_id), 0);
  loom_type_t register_type = {};
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module, /*carrier_payload0=*/42, /*carrier_payload1=*/4, value_type,
      &register_type));
  loom_value_id_t register_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, register_type, &register_id));

  EXPECT_TRUE(loom_module_value_has_type_uses(module, old_dim_id));
  IREE_ASSERT_OK(
      loom_module_replace_value_type_uses(module, old_dim_id, new_dim_id));

  loom_type_t replaced_type = loom_module_value_type(module, register_id);
  const loom_type_t* replaced_value_type =
      loom_type_register_value_type(replaced_type);
  ASSERT_NE(replaced_value_type, nullptr);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(*replaced_value_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(*replaced_value_type, 0), new_dim_id);
  EXPECT_EQ(loom_type_register_payload0(replaced_type), 42u);
  EXPECT_EQ(loom_type_register_payload1(replaced_type), 4u);
  EXPECT_FALSE(loom_module_value_has_type_uses(module, old_dim_id));
  EXPECT_TRUE(loom_module_value_has_type_uses(module, new_dim_id));

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceValueTypeUsesUpdatesParameterizedTypeSlots) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t old_dim_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, index_type, &old_dim_id));
  IREE_ASSERT_OK(loom_module_define_value(module, index_type, &new_dim_id));

  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim_id), 0);
  loom_type_id_t vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module, vector_type, &vector_type_id));
  loom_type_t array_type = {};
  IREE_ASSERT_OK(loom_test_array_type_make(
      module, /*build_flags=*/0, vector_type_id, /*alignment=*/0,
      loom_named_attr_slice_empty(), &array_type));
  loom_value_id_t array_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, array_type, &array_id));

  EXPECT_TRUE(loom_module_value_has_type_uses(module, old_dim_id));
  IREE_ASSERT_OK(
      loom_module_replace_value_type_uses(module, old_dim_id, new_dim_id));

  loom_type_t replaced_array_type = loom_module_value_type(module, array_id);
  ASSERT_TRUE(loom_test_array_type_isa(replaced_array_type));
  loom_type_id_t replaced_vector_type_id =
      loom_test_array_type_element_type(replaced_array_type);
  ASSERT_LT(replaced_vector_type_id, module->types.count);
  loom_type_t replaced_vector_type =
      module->types.entries[replaced_vector_type_id];
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_vector_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_vector_type, 0), new_dim_id);
  EXPECT_FALSE(loom_module_value_has_type_uses(module, old_dim_id));
  EXPECT_TRUE(loom_module_value_has_type_uses(module, new_dim_id));

  loom_module_free(module);
}

TEST_F(ModuleTest,
       ReplaceValueTypeUsesUpdatesParameterizedAttributeArraySlots) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t old_dim_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t new_dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, index_type, &old_dim_id));
  IREE_ASSERT_OK(loom_module_define_value(module, index_type, &new_dim_id));

  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim_id), 0);
  loom_type_id_t vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module, vector_type, &vector_type_id));
  loom_attribute_t options = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE,
      LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(), vector_type_id,
      loom_attr_absent(), loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &options));
  loom_attribute_t variants[] = {options};
  loom_type_t variant_set_type = {};
  IREE_ASSERT_OK(loom_test_variant_set_type_make(
      module, /*build_flags=*/0,
      loom_make_parameterized_attr_array(variants, IREE_ARRAYSIZE(variants)),
      loom_parameterized_attr_array_empty(), &variant_set_type));
  loom_value_id_t variant_set_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_define_value(module, variant_set_type, &variant_set_id));

  EXPECT_TRUE(loom_module_value_has_type_uses(module, old_dim_id));
  IREE_ASSERT_OK(
      loom_module_replace_value_type_uses(module, old_dim_id, new_dim_id));

  loom_type_t replaced_type = loom_module_value_type(module, variant_set_id);
  ASSERT_TRUE(loom_test_variant_set_type_isa(replaced_type));
  loom_parameterized_attr_array_t replaced_variants =
      loom_test_variant_set_type_values(replaced_type);
  ASSERT_EQ(replaced_variants.count, 1u);
  ASSERT_TRUE(loom_test_options_attr_isa(replaced_variants.values[0]));
  loom_type_id_t replaced_vector_type_id =
      loom_test_options_attr_element_type(replaced_variants.values[0]);
  ASSERT_LT(replaced_vector_type_id, module->types.count);
  loom_type_t replaced_vector_type =
      module->types.entries[replaced_vector_type_id];
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_vector_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_vector_type, 0), new_dim_id);
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
// Symbol-set attributes
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, MakeSymbolSetSortsByNameAndCopiesReferences) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("zeta"), &zeta_name));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_name));
  uint16_t zeta_symbol = LOOM_SYMBOL_ID_INVALID;
  uint16_t alpha_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, zeta_name, &zeta_symbol));
  IREE_ASSERT_OK(loom_module_add_symbol(module, alpha_name, &alpha_symbol));
  ASSERT_LT(zeta_symbol, alpha_symbol)
      << "setup must distinguish symbol-id order from name order";

  loom_symbol_ref_t refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/zeta_symbol},
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol},
  };
  loom_symbol_ref_t duplicate_ref = loom_symbol_ref_null();
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_try_make_symbol_set(
      module, loom_make_symbol_ref_array(refs, IREE_ARRAYSIZE(refs)),
      &duplicate_ref, &attr));

  EXPECT_FALSE(loom_symbol_ref_is_valid(duplicate_ref));
  ASSERT_EQ(attr.kind, LOOM_ATTR_SYMBOL_SET);
  loom_symbol_ref_array_t set = loom_attr_as_symbol_set(attr);
  ASSERT_EQ(set.count, 2);
  ASSERT_NE(set.values, refs);
  EXPECT_EQ(set.values[0].symbol_id, alpha_symbol);
  EXPECT_EQ(set.values[1].symbol_id, zeta_symbol);

  loom_symbol_ref_t reversed_refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol},
      {/*.module_id=*/0, /*.symbol_id=*/zeta_symbol},
  };
  loom_attribute_t equal_attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_try_make_symbol_set(
      module,
      loom_make_symbol_ref_array(reversed_refs, IREE_ARRAYSIZE(reversed_refs)),
      &duplicate_ref, &equal_attr));
  EXPECT_TRUE(loom_attribute_equal(&attr, &equal_attr));
  EXPECT_EQ(loom_attribute_hash(&attr), loom_attribute_hash(&equal_attr));

  refs[0].symbol_id = alpha_symbol;
  EXPECT_EQ(set.values[1].symbol_id, zeta_symbol);
  loom_module_free(module);
}

TEST_F(ModuleTest, MakeSymbolSetReportsDuplicateName) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("shared"), &name));
  uint16_t first_symbol = LOOM_SYMBOL_ID_INVALID;
  uint16_t second_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name, &first_symbol));
  IREE_ASSERT_OK(loom_module_add_symbol(module, name, &second_symbol));

  loom_symbol_ref_t refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/first_symbol},
      {/*.module_id=*/0, /*.symbol_id=*/second_symbol},
  };
  loom_symbol_ref_t duplicate_ref = loom_symbol_ref_null();
  loom_attribute_t attr = loom_attr_i64(42);
  IREE_ASSERT_OK(loom_module_try_make_symbol_set(
      module, loom_make_symbol_ref_array(refs, IREE_ARRAYSIZE(refs)),
      &duplicate_ref, &attr));

  EXPECT_TRUE(loom_symbol_ref_is_valid(duplicate_ref));
  EXPECT_EQ(module->symbols.entries[duplicate_ref.symbol_id].name_id, name);
  EXPECT_TRUE(loom_attr_is_absent(attr));
  loom_module_free(module);
}

TEST_F(ModuleTest, MakeSymbolSetValidatesReferencesAndEmptySet) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_symbol_ref_t duplicate_ref = loom_symbol_ref_null();
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_try_make_symbol_set(
      module, loom_symbol_ref_array_empty(), &duplicate_ref, &attr));
  EXPECT_FALSE(loom_symbol_ref_is_valid(duplicate_ref));
  EXPECT_EQ(attr.kind, LOOM_ATTR_SYMBOL_SET);
  EXPECT_EQ(attr.count, 0);
  EXPECT_EQ(loom_attr_as_symbol_set(attr).values, nullptr);

  loom_symbol_ref_t remote_ref = {/*.module_id=*/1, /*.symbol_id=*/0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_try_make_symbol_set(
                            module, loom_make_symbol_ref_array(&remote_ref, 1),
                            &duplicate_ref, &attr));
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

TEST_F(ModuleTest, MakeCanonicalAttributeCopiesTemporaryNestedPayloads) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("key"), &key_id));
  int64_t temporary_values[] = {3, 5, 8};
  loom_named_attr_t temporary_entries[] = {
      {
          /*.name_id=*/key_id,
          /*.reserved=*/{},
          /*.value=*/
          loom_attr_i64_array(temporary_values,
                              IREE_ARRAYSIZE(temporary_values)),
      },
  };
  loom_attribute_t canonical = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_canonical_attribute(
      module, /*descriptor=*/NULL,
      loom_make_canonical_attr_dict(temporary_entries,
                                    IREE_ARRAYSIZE(temporary_entries)),
      &canonical));

  temporary_values[0] = 99;
  temporary_entries[0].value = loom_attr_i64(42);
  ASSERT_EQ(canonical.kind, LOOM_ATTR_DICT);
  ASSERT_EQ(canonical.count, 1u);
  ASSERT_EQ(canonical.dict_entries[0].value.kind, LOOM_ATTR_I64_ARRAY);
  ASSERT_EQ(canonical.dict_entries[0].value.count, 3u);
  EXPECT_EQ(canonical.dict_entries[0].value.i64_array[0], 3);
  EXPECT_EQ(canonical.dict_entries[0].value.i64_array[1], 5);
  EXPECT_EQ(canonical.dict_entries[0].value.i64_array[2], 8);

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

TEST_F(ModuleTest, ParameterizedAttrBuilderFreezesNestedPayloads) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t tile = {0};
  IREE_ASSERT_OK(loom_test_tile_attr_make(module, 16, &tile));
  uint8_t scope_values[] = {
      LOOM_TEST_OPTIONS_SCOPES_WORKGROUP,
      254,
  };
  loom_attribute_t options = {0};
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module,
      LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_SCOPES |
          LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TILE,
      LOOM_TEST_OPTIONS_MODE_FAST,
      loom_make_enum_array(scope_values, IREE_ARRAYSIZE(scope_values)),
      LOOM_TYPE_ID_INVALID, tile, loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &options));

  scope_values[0] = 99;
  EXPECT_TRUE(loom_test_options_attr_isa(options));
  EXPECT_EQ(loom_test_options_attr_mode(options), LOOM_TEST_OPTIONS_MODE_FAST);
  ASSERT_TRUE(loom_test_options_attr_has_scopes(options));
  loom_enum_array_t scopes = loom_test_options_attr_scopes(options);
  ASSERT_EQ(scopes.count, 2u);
  EXPECT_EQ(scopes.values[0], LOOM_TEST_OPTIONS_SCOPES_WORKGROUP);
  EXPECT_EQ(scopes.values[1], 254u);
  EXPECT_FALSE(loom_test_options_attr_has_element_type(options));
  ASSERT_TRUE(loom_test_options_attr_has_tile(options));
  loom_attribute_t nested_tile = loom_test_options_attr_tile(options);
  EXPECT_TRUE(loom_test_tile_attr_isa(nested_tile));
  EXPECT_EQ(loom_test_tile_attr_width(nested_tile), 16);
  EXPECT_NE(loom_attr_as_parameterized_slots(nested_tile),
            loom_attr_as_parameterized_slots(tile));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("options"), &key_id));
  loom_named_attr_t entries[] = {{
      /*.name_id=*/key_id,
      /*.reserved=*/0,
      /*.value=*/options,
  }};
  loom_attribute_t dict = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      &dict));
  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, dict));

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedAttrPreservesOptionalPresence) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t absent_scopes = {0};
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, /*build_flags=*/0, LOOM_TEST_OPTIONS_MODE_PRECISE,
      loom_enum_array_empty(), LOOM_TYPE_ID_INVALID, loom_attr_absent(),
      loom_symbol_ref_null(), loom_parameterized_attr_array_empty(),
      &absent_scopes));
  loom_attribute_t empty_scopes = {0};
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_SCOPES,
      LOOM_TEST_OPTIONS_MODE_PRECISE, loom_enum_array_empty(),
      LOOM_TYPE_ID_INVALID, loom_attr_absent(), loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &empty_scopes));

  EXPECT_FALSE(loom_test_options_attr_has_scopes(absent_scopes));
  EXPECT_TRUE(loom_test_options_attr_has_scopes(empty_scopes));
  EXPECT_EQ(loom_test_options_attr_scopes(empty_scopes).count, 0u);
  EXPECT_FALSE(loom_attribute_equal(&absent_scopes, &empty_scopes));

  loom_attribute_t second_empty_scopes = {0};
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_SCOPES,
      LOOM_TEST_OPTIONS_MODE_PRECISE, loom_enum_array_empty(),
      LOOM_TYPE_ID_INVALID, loom_attr_absent(), loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &second_empty_scopes));
  EXPECT_TRUE(loom_attribute_equal(&empty_scopes, &second_empty_scopes));
  EXPECT_EQ(loom_attribute_hash(&empty_scopes),
            loom_attribute_hash(&second_empty_scopes));

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedAttrArrayFreezesOrderedMixedFamilies) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t tile = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_tile_attr_make(module, 8, &tile));
  loom_attribute_t options = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, /*build_flags=*/0, LOOM_TEST_OPTIONS_MODE_FAST,
      loom_enum_array_empty(), LOOM_TYPE_ID_INVALID, loom_attr_absent(),
      loom_symbol_ref_null(), loom_parameterized_attr_array_empty(), &options));
  loom_attribute_t values[] = {tile, options, tile};

  loom_attribute_t array = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      module,
      loom_make_parameterized_attr_array(values, IREE_ARRAYSIZE(values)),
      &array));
  values[0] = options;

  loom_parameterized_attr_array_t frozen =
      loom_attr_as_parameterized_array(array);
  ASSERT_EQ(frozen.count, 3u);
  EXPECT_TRUE(loom_test_tile_attr_isa(frozen.values[0]));
  EXPECT_TRUE(loom_test_options_attr_isa(frozen.values[1]));
  EXPECT_TRUE(loom_test_tile_attr_isa(frozen.values[2]));
  EXPECT_TRUE(loom_attribute_equal(&frozen.values[0], &frozen.values[2]));
  EXPECT_NE(frozen.values, values);
  EXPECT_NE(loom_attr_as_parameterized_slots(frozen.values[0]),
            loom_attr_as_parameterized_slots(tile));

  loom_attribute_t repeated_values[] = {tile, options, tile};
  loom_attribute_t equal_array = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      module,
      loom_make_parameterized_attr_array(repeated_values,
                                         IREE_ARRAYSIZE(repeated_values)),
      &equal_array));
  EXPECT_TRUE(loom_attribute_equal(&array, &equal_array));
  EXPECT_EQ(loom_attribute_hash(&array), loom_attribute_hash(&equal_array));

  loom_attribute_t empty_array = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      module, loom_parameterized_attr_array_empty(), &empty_array));
  EXPECT_EQ(loom_attr_as_parameterized_array(empty_array).count, 0u);
  loom_attribute_t absent_array = loom_attr_absent();
  EXPECT_FALSE(loom_attribute_equal(&empty_array, &absent_array));

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedAttrArrayPreservesExactFamilyPresence) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t tile = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_tile_attr_make(module, 8, &tile));
  loom_attribute_t tile_values[] = {tile, tile};
  const loom_parameterized_attr_array_t tiles =
      loom_make_parameterized_attr_array(tile_values,
                                         IREE_ARRAYSIZE(tile_values));

  loom_attribute_t present = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TILES,
      LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(),
      LOOM_TYPE_ID_INVALID, loom_attr_absent(), loom_symbol_ref_null(), tiles,
      &present));
  ASSERT_TRUE(loom_test_options_attr_has_tiles(present));
  loom_parameterized_attr_array_t frozen =
      loom_test_options_attr_tiles(present);
  ASSERT_EQ(frozen.count, 2u);
  EXPECT_NE(frozen.values, tile_values);
  EXPECT_TRUE(loom_test_tile_attr_isa(frozen.values[0]));
  EXPECT_TRUE(loom_test_tile_attr_isa(frozen.values[1]));

  loom_attribute_t absent = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, /*build_flags=*/0, LOOM_TEST_OPTIONS_MODE_FAST,
      loom_enum_array_empty(), LOOM_TYPE_ID_INVALID, loom_attr_absent(),
      loom_symbol_ref_null(), loom_parameterized_attr_array_empty(), &absent));
  EXPECT_FALSE(loom_test_options_attr_has_tiles(absent));

  loom_attribute_t present_empty = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TILES,
      LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(),
      LOOM_TYPE_ID_INVALID, loom_attr_absent(), loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &present_empty));
  EXPECT_TRUE(loom_test_options_attr_has_tiles(present_empty));
  EXPECT_EQ(loom_test_options_attr_tiles(present_empty).count, 0u);
  EXPECT_FALSE(loom_attribute_equal(&absent, &present_empty));

  loom_attribute_t wrong_family_values[] = {absent};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_test_options_attr_make(
          module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TILES,
          LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(),
          LOOM_TYPE_ID_INVALID, loom_attr_absent(), loom_symbol_ref_null(),
          loom_make_parameterized_attr_array(wrong_family_values, 1),
          &present));

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedAttrArrayWalksAndReplacesNestedValueRefs) {
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
  loom_type_id_t vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module, vector_type, &vector_type_id));

  loom_attribute_t options = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE,
      LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(), vector_type_id,
      loom_attr_absent(), loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &options));
  loom_attribute_t values[] = {options};
  loom_attribute_t array = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      module,
      loom_make_parameterized_attr_array(values, IREE_ARRAYSIZE(values)),
      &array));

  AttributeValueRefCapture original_capture = {};
  IREE_ASSERT_OK(loom_module_walk_attribute_value_refs(
      module, array, CaptureAttributeValueRef, &original_capture));
  ASSERT_EQ(original_capture.count, 1u);
  EXPECT_EQ(original_capture.values[0], old_dim_id);

  loom_attribute_t replaced_array = loom_attr_absent();
  bool changed = false;
  IREE_ASSERT_OK(loom_module_replace_attribute_value_references(
      module, array, old_dim_id, new_dim_id, &replaced_array, &changed));
  EXPECT_TRUE(changed);
  EXPECT_NE(loom_attr_as_parameterized_array(replaced_array).values,
            loom_attr_as_parameterized_array(array).values);

  loom_attribute_t original_options =
      loom_attr_as_parameterized_array(array).values[0];
  loom_type_t original_vector =
      module->types
          .entries[loom_test_options_attr_element_type(original_options)];
  EXPECT_EQ(loom_type_dim_value_id_at(original_vector, 0), old_dim_id);

  loom_attribute_t replaced_options =
      loom_attr_as_parameterized_array(replaced_array).values[0];
  loom_type_t replaced_vector =
      module->types
          .entries[loom_test_options_attr_element_type(replaced_options)];
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_vector, 0), new_dim_id);

  AttributeValueRefCapture replaced_capture = {};
  IREE_ASSERT_OK(loom_module_walk_attribute_value_refs(
      module, replaced_array, CaptureAttributeValueRef, &replaced_capture));
  ASSERT_EQ(replaced_capture.count, 1u);
  EXPECT_EQ(replaced_capture.values[0], new_dim_id);

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedAttrArrayRejectsMalformedAndDeepValues) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t malformed_values[] = {loom_attr_i64(1)};
  loom_attribute_t array = loom_attr_absent();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_make_parameterized_attr_array(
          module, loom_make_parameterized_attr_array(malformed_values, 1),
          &array));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_make_parameterized_attr_array(
          module, loom_make_parameterized_attr_array(NULL, 1), &array));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_module_make_parameterized_attr_array(
                            module,
                            loom_make_parameterized_attr_array(
                                NULL, (iree_host_size_t)UINT16_MAX + 1),
                            &array));

  loom_attribute_t tile = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_tile_attr_make(module, 8, &tile));
  loom_attribute_t valid_values[] = {tile};
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      module,
      loom_make_parameterized_attr_array(valid_values,
                                         IREE_ARRAYSIZE(valid_values)),
      &array));
  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("values"), &key_id));
  loom_named_attr_t dict_entries[] = {{
      /*.name_id=*/key_id,
      /*.reserved=*/0,
      /*.value=*/array,
  }};
  loom_attribute_t dict = loom_attr_absent();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_make_canonical_attr_dict(
                            module,
                            loom_make_named_attr_slice(
                                dict_entries, IREE_ARRAYSIZE(dict_entries)),
                            &dict));

  loom_attribute_t node = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_node_attr_make(module, /*build_flags=*/0, 0,
                                          loom_parameterized_attr_array_empty(),
                                          &node));
  for (int64_t value = 1; value < 4; ++value) {
    loom_attribute_t child[] = {node};
    IREE_ASSERT_OK(loom_test_node_attr_make(
        module, LOOM_TEST_NODE_ATTR_BUILD_FLAG_HAS_CHILDREN, value,
        loom_make_parameterized_attr_array(child, IREE_ARRAYSIZE(child)),
        &node));
  }
  loom_attribute_t child[] = {node};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_test_node_attr_make(
          module, LOOM_TEST_NODE_ATTR_BUILD_FLAG_HAS_CHILDREN, 4,
          loom_make_parameterized_attr_array(child, IREE_ARRAYSIZE(child)),
          &node));

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedAttrRejectsMalformedSlots) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t slots[6] = {
      loom_attr_enum(LOOM_TEST_OPTIONS_MODE_FAST),
      loom_attr_absent(),
      loom_attr_absent(),
      loom_attr_absent(),
      loom_attr_absent(),
      loom_attr_absent(),
  };
  loom_attribute_t options = {0};
  IREE_ASSERT_OK(loom_module_make_parameterized_attr(
      module, LOOM_PARAMETERIZED_ATTR_TEST_OPTIONS, slots,
      IREE_ARRAYSIZE(slots), &options));

  slots[0] = loom_attr_absent();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_make_parameterized_attr(
                            module, LOOM_PARAMETERIZED_ATTR_TEST_OPTIONS, slots,
                            IREE_ARRAYSIZE(slots), &options));
  slots[0] = loom_attr_enum(99);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_make_parameterized_attr(
                            module, LOOM_PARAMETERIZED_ATTR_TEST_OPTIONS, slots,
                            IREE_ARRAYSIZE(slots), &options));
  slots[0] = loom_attr_enum(LOOM_TEST_OPTIONS_MODE_FAST);
  slots[3] = options;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_make_parameterized_attr(
                            module, LOOM_PARAMETERIZED_ATTR_TEST_OPTIONS, slots,
                            IREE_ARRAYSIZE(slots), &options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_module_make_parameterized_attr(
                            module, LOOM_PARAMETERIZED_ATTR_TEST_OPTIONS, slots,
                            IREE_ARRAYSIZE(slots) - 1, &options));

  loom_module_free(module);
}

TEST_F(ModuleTest, SignedEnumSetParameterizedAttrCanonicalizesAndOwnsWords) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  uint64_t source_words[] = {
      UINT64_C(1) << 1, 0, 0, 0, UINT64_C(1) << 7, 0, 0, 0,
  };
  loom_attribute_t feature_set = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_feature_set_attr_make(
      module,
      loom_make_signed_enum_set(source_words, IREE_ARRAYSIZE(source_words) / 2),
      &feature_set));

  loom_signed_enum_set_t canonical =
      loom_test_feature_set_attr_features(feature_set);
  EXPECT_EQ(canonical.word_count, 1u);
  EXPECT_NE(canonical.words, source_words);
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(canonical, 1));
  EXPECT_TRUE(loom_signed_enum_set_contains_negative(canonical, 7));
  source_words[0] = 0;
  source_words[4] = 0;
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(canonical, 1));
  EXPECT_TRUE(loom_signed_enum_set_contains_negative(canonical, 7));

  loom_module_free(module);
}

TEST_F(ModuleTest, SignedEnumSetParameterizedAttrRejectsInvalidValues) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  uint64_t undeclared_words[] = {
      UINT64_C(1) << 2,
      0,
  };
  loom_attribute_t feature_set = loom_attr_absent();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_test_feature_set_attr_make(
          module, loom_make_signed_enum_set(undeclared_words, 1),
          &feature_set));

  uint64_t contradictory_words[] = {
      UINT64_C(1) << 1,
      UINT64_C(1) << 1,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_test_feature_set_attr_make(
          module, loom_make_signed_enum_set(contradictory_words, 1),
          &feature_set));

  loom_module_free(module);
}

TEST_F(ModuleTest, ParameterizedTypeBuilderInternsDescriptorIndexedSlots) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_id_t bf16_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &bf16_type_id));
  loom_type_t matrix_type = {0};
  IREE_ASSERT_OK(
      loom_test_matrix_type_make(module, /*build_flags=*/0, bf16_type_id,
                                 LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP, 16,
                                 loom_symbol_ref_null(), &matrix_type));
  loom_type_t duplicate_type = {0};
  IREE_ASSERT_OK(
      loom_test_matrix_type_make(module, /*build_flags=*/0, bf16_type_id,
                                 LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP, 16,
                                 loom_symbol_ref_null(), &duplicate_type));

  EXPECT_TRUE(loom_test_matrix_type_isa(matrix_type));
  EXPECT_EQ(loom_test_matrix_type_element_type(matrix_type), bf16_type_id);
  EXPECT_EQ(loom_test_matrix_type_scope(matrix_type),
            LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP);
  EXPECT_EQ(loom_test_matrix_type_rows(matrix_type), 16);
  EXPECT_TRUE(loom_type_equal(matrix_type, duplicate_type));
  EXPECT_EQ(loom_type_hash(matrix_type), loom_type_hash(duplicate_type));
  EXPECT_EQ(loom_type_parameterized_parameters(matrix_type),
            loom_type_parameterized_parameters(duplicate_type));

  iree_host_size_t allocation_size = module->arena.used_allocation_size;
  for (int i = 0; i < 32; ++i) {
    loom_type_t repeated_type = {0};
    IREE_ASSERT_OK(
        loom_test_matrix_type_make(module, /*build_flags=*/0, bf16_type_id,
                                   LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP, 16,
                                   loom_symbol_ref_null(), &repeated_type));
    EXPECT_EQ(loom_type_parameterized_parameters(repeated_type),
              loom_type_parameterized_parameters(matrix_type));
  }
  EXPECT_EQ(module->arena.used_allocation_size, allocation_size);

  loom_type_t packed_array = {0};
  IREE_ASSERT_OK(loom_test_array_type_make(
      module, /*build_flags=*/0, bf16_type_id, /*alignment=*/0,
      loom_named_attr_slice_empty(), &packed_array));
  EXPECT_FALSE(loom_test_array_type_has_alignment(packed_array));
  loom_type_t aligned_array = {0};
  IREE_ASSERT_OK(loom_test_array_type_make(
      module, LOOM_TEST_ARRAY_TYPE_BUILD_FLAG_HAS_ALIGNMENT, bf16_type_id,
      /*alignment=*/32, loom_named_attr_slice_empty(), &aligned_array));
  EXPECT_TRUE(loom_test_array_type_has_alignment(aligned_array));
  EXPECT_EQ(loom_test_array_type_alignment(aligned_array), 32);

  loom_module_free(module);
}

TEST_F(ModuleTest, CompactParameterizedTypeBuilderPacksWithoutAllocation) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  const iree_host_size_t allocation_size = module->arena.used_allocation_size;
  loom_attribute_t role = loom_attr_enum(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_type_t encoding_type = {0};
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module, &loom_encoding_type_parameterized_descriptor, &role, 1,
      &encoding_type));
  EXPECT_TRUE(loom_type_equal(
      encoding_type,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  EXPECT_EQ(module->arena.used_allocation_size, allocation_size);

  role = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module, &loom_encoding_type_parameterized_descriptor, &role, 1,
      &encoding_type));
  EXPECT_TRUE(loom_type_equal(encoding_type, loom_type_encoding()));
  EXPECT_EQ(module->arena.used_allocation_size, allocation_size);

  loom_attribute_t space = loom_attr_enum(LOOM_STORAGE_SPACE_WORKGROUP);
  loom_type_t storage_type = {0};
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module, &loom_low_storage_type_parameterized_descriptor, &space, 1,
      &storage_type));
  EXPECT_TRUE(loom_type_equal(storage_type,
                              loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP)));
  space = loom_attr_enum(LOOM_STORAGE_SPACE_STACK);
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module, &loom_low_storage_type_parameterized_descriptor, &space, 1,
      &storage_type));
  EXPECT_TRUE(loom_type_equal(storage_type,
                              loom_type_storage(LOOM_STORAGE_SPACE_STACK)));
  EXPECT_EQ(module->arena.used_allocation_size, allocation_size);

  role = loom_attr_enum(99);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_make_parameterized_type(
          module, &loom_encoding_type_parameterized_descriptor, &role, 1,
          &encoding_type));
  EXPECT_EQ(module->arena.used_allocation_size, allocation_size);

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

TEST_F(ModuleTest, InternTopologicalTypeHandlesDeepCanonicalChain) {
  constexpr iree_host_size_t kDepth = 4096;
  loom_module_t* module = NULL;
  const loom_module_size_hints_t hints = {
      /*.type_count=*/kDepth + 1,
  };
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      &hints, iree_allocator_system(),
                                      &module));

  loom_type_id_t dependency_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_topological_type_id(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      /*structural_dependency_ids=*/NULL,
      /*structural_dependency_count=*/0, &dependency_id));
  ASSERT_EQ(dependency_id, 0u);

  struct FunctionTypeStorage {
    // Number of argument types.
    uint16_t arg_count;
    // Number of result types.
    uint16_t result_count;
    // Alignment padding matching loom_func_type_data_t.
    uint32_t reserved;
    // Single immediate dependency payload.
    loom_type_t types[1];
  } source = {};
  static_assert(sizeof(FunctionTypeStorage) ==
                sizeof(loom_func_type_data_t) + sizeof(loom_type_t));
  source.arg_count = 1;
  for (iree_host_size_t i = 0; i < kDepth; ++i) {
    source.types[0] = module->types.entries[dependency_id];
    const loom_type_t type =
        loom_type_function(reinterpret_cast<loom_func_type_data_t*>(&source));
    loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_topological_type_id(
        module, type, &dependency_id, /*structural_dependency_count=*/1,
        &type_id));
    ASSERT_EQ(type_id, i + 1);
    dependency_id = type_id;
  }

  source.types[0] = module->types.entries[dependency_id - 1];
  const loom_type_t duplicate =
      loom_type_function(reinterpret_cast<loom_func_type_data_t*>(&source));
  const loom_type_id_t duplicate_dependency_id = dependency_id - 1;
  loom_type_id_t duplicate_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_topological_type_id(
      module, duplicate, &duplicate_dependency_id,
      /*structural_dependency_count=*/1, &duplicate_id));
  EXPECT_EQ(duplicate_id, dependency_id);
  EXPECT_EQ(module->types.count, kDepth + 1);

  const loom_func_type_data_t* deepest_data =
      loom_type_func_data(module->types.entries[dependency_id]);
  ASSERT_NE(deepest_data, nullptr);
  const loom_type_t expected_dependency =
      module->types.entries[dependency_id - 1];
  EXPECT_EQ(deepest_data->types[0].header, expected_dependency.header);
  EXPECT_EQ(deepest_data->types[0].encoding_id,
            expected_dependency.encoding_id);
  EXPECT_EQ(deepest_data->types[0].encoding_flags,
            expected_dependency.encoding_flags);
  EXPECT_EQ(deepest_data->types[0].dims[0], expected_dependency.dims[0]);
  EXPECT_EQ(deepest_data->types[0].dims[1], expected_dependency.dims[1]);
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

TEST_F(ModuleTest, InternImplicitShapedAttachmentUsesAbsentIdentity) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t dense_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("dense"), &dense_name_id));
  const loom_encoding_t dense_encoding = {
      /*.name_id=*/dense_name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
  };
  uint16_t dense_encoding_id = 0;
  IREE_ASSERT_OK(
      loom_module_add_encoding(module, &dense_encoding, &dense_encoding_id));

  const uint64_t rows = loom_dim_pack_static(4);
  const uint64_t columns = loom_dim_pack_static(8);
  const loom_type_t implicit_type = loom_type_shaped_2d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, rows, columns, 0);
  const loom_type_t explicit_type = loom_type_shaped_2d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, rows, columns, dense_encoding_id);
  loom_type_id_t implicit_type_id = LOOM_TYPE_ID_INVALID;
  loom_type_id_t explicit_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module, implicit_type, &implicit_type_id));
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module, explicit_type, &explicit_type_id));

  EXPECT_EQ(explicit_type_id, implicit_type_id);
  EXPECT_FALSE(loom_type_has_encoding(module->types.entries[implicit_type_id]));
  loom_module_free(module);
}

TEST_F(ModuleTest, InternRegisterTypeOwnsAndDeduplicatesValueType) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t dialect_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("dialect.payload"),
                                           &dialect_name_id));
  loom_type_t value_type_param = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t value_type =
      loom_type_dialect(dialect_name_id, 1, &value_type_param);

  loom_type_t first = {};
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module, /*carrier_payload0=*/42, /*carrier_payload1=*/4, value_type,
      &first));
  iree_host_size_t allocation_size = module->arena.total_allocation_size;
  loom_type_t canonical_again = {};
  IREE_ASSERT_OK(loom_module_intern_type(module, first, &canonical_again));
  EXPECT_EQ(loom_type_register_data(first),
            loom_type_register_data(canonical_again));
  EXPECT_EQ(module->arena.total_allocation_size, allocation_size);
  value_type_param = loom_type_scalar(LOOM_SCALAR_TYPE_I64);

  loom_type_t duplicate_param = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t duplicate_value_type =
      loom_type_dialect(dialect_name_id, 1, &duplicate_param);

  loom_type_t duplicate = {};
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module, /*carrier_payload0=*/42, /*carrier_payload1=*/4,
      duplicate_value_type, &duplicate));

  ASSERT_EQ(module->types.count, 4u);
  EXPECT_TRUE(loom_type_equal(module->types.entries[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(loom_type_equal(module->types.entries[1], duplicate_param));
  EXPECT_TRUE(loom_type_equal(module->types.entries[2], duplicate_value_type));
  EXPECT_TRUE(loom_type_equal(module->types.entries[3], first));
  EXPECT_TRUE(loom_type_equal(first, duplicate));
  EXPECT_EQ(loom_type_register_data(first), loom_type_register_data(duplicate));
  EXPECT_EQ(module->arena.total_allocation_size, allocation_size);
  const loom_type_t* owned_value_type = loom_type_register_value_type(first);
  ASSERT_NE(owned_value_type, nullptr);
  ASSERT_TRUE(loom_type_is_dialect(*owned_value_type));
  ASSERT_EQ(loom_type_dialect_param_count(*owned_value_type), 1u);
  EXPECT_TRUE(loom_type_equal(loom_type_dialect_params(*owned_value_type)[0],
                              duplicate_param));
  EXPECT_NE(loom_type_dialect_params(*owned_value_type), &value_type_param);

  loom_module_free(module);
}

TEST_F(ModuleTest, InternRegisterTypeRejectsNullTypedPayload) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t malformed_type = loom_type_register_payload_with_value_type(NULL);
  loom_type_t interned_type = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_intern_type(module, malformed_type, &interned_type));

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

  const loom_type_id_t dependency_ids[] = {0, 1, 2};
  loom_type_id_t topological_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_topological_type_id(
      module, packed_source, dependency_ids, IREE_ARRAYSIZE(dependency_ids),
      &topological_type_id));
  EXPECT_EQ(topological_type_id, 3u);
  EXPECT_EQ(module->types.hashes[topological_type_id],
            loom_type_hash(module->types.entries[topological_type_id]));
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

TEST_F(ModuleTest, BlockAppendSuccessorOpMarksRegionCfg) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(
      &module->arena, sizeof(loom_op_t) + sizeof(loom_block_t*), (void**)&op));
  memset(op, 0, sizeof(loom_op_t) + sizeof(loom_block_t*));
  op->kind = 0x0100;
  op->successor_count = 1;

  IREE_ASSERT_OK(loom_block_append_op(module, block, op));
  EXPECT_TRUE(
      iree_any_bit_set(module->body->flags, LOOM_REGION_INSTANCE_FLAG_CFG));
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
      /*.encoding_count=*/12,
      /*.source_count=*/6,
      /*.symbol_count=*/10,
  };
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      &hints, iree_allocator_system(),
                                      &module));
  // Stable value segments allocate lazily instead of reserving speculative
  // headroom from the hint. Contiguous tables retain growth-factor sizing.
  EXPECT_EQ(loom_value_table_capacity(&module->values), 0u);
  EXPECT_GE(module->strings.capacity, 50u);
  EXPECT_GE(module->types.capacity, 20u);
  EXPECT_GE(module->encodings.capacity, 12u);
  EXPECT_GE(module->encoding_intern.capacity, 12u);
  EXPECT_GE(module->sources.capacity, 6u);
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
      /*.family=*/{},
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
  EXPECT_NE(found->family.id, LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_EQ(loom_module_encoding_vtable(module, found), &kQ8_0EncodingVtable);
  EXPECT_EQ(found->attributes[0].name_id, block_id);
  EXPECT_EQ(found->attributes[0].value.i64, 32);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingRetainsMalformedParametersForVerification) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &block_id));
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("thirty_two"), &value_id));
  loom_named_attr_t parameter = {
      /*.name_id=*/block_id,
      /*.reserved=*/{},
      /*.value=*/loom_attr_string(value_id),
  };
  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.family=*/{},
      /*.attributes=*/&parameter,
  };

  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
  const loom_encoding_t* stored = loom_module_encoding(module, encoding_id);
  ASSERT_NE(stored, nullptr);
  EXPECT_FALSE(loom_encoding_static_parameters_are_valid(stored));
  EXPECT_FALSE(loom_encoding_static_is_valid(stored));
  EXPECT_EQ(loom_encoding_parameter_descriptor_index(&stored->attributes[0]),
            0u);

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

TEST_F(ModuleTest, AddEncodingDedupAfterInternTableGrowth) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &block_id));

  constexpr uint16_t kEncodingCount = 128;
  for (uint16_t i = 0; i < kEncodingCount; ++i) {
    loom_named_attr_t parameter = {
        /*.name_id=*/block_id,
        /*.reserved=*/{},
        /*.value=*/loom_attr_i64(i),
    };
    loom_encoding_t encoding = {
        /*.name_id=*/name_id,
        /*.alias_id=*/LOOM_STRING_ID_INVALID,
        /*.attribute_count=*/1,
        /*.family=*/{},
        /*.attributes=*/&parameter,
    };
    uint16_t encoding_id = 0;
    IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
    EXPECT_EQ(encoding_id, i + 1);
  }

  const uint16_t duplicate_ordinals[] = {0, 63, 127};
  for (uint16_t ordinal : duplicate_ordinals) {
    loom_named_attr_t parameter = {
        /*.name_id=*/block_id,
        /*.reserved=*/{},
        /*.value=*/loom_attr_i64(ordinal),
    };
    loom_encoding_t encoding = {
        /*.name_id=*/name_id,
        /*.alias_id=*/LOOM_STRING_ID_INVALID,
        /*.attribute_count=*/1,
        /*.family=*/{},
        /*.attributes=*/&parameter,
    };
    uint16_t encoding_id = 0;
    IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
    EXPECT_EQ(encoding_id, ordinal + 1);
  }
  EXPECT_EQ(module->encodings.count, kEncodingCount);

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
      /*.family=*/{},
      /*.attributes=*/attrs_a,
  };
  loom_encoding_t aliased = {
      /*.name_id=*/name_id,
      /*.alias_id=*/alias_id,
      /*.attribute_count=*/1,
      /*.family=*/{},         /*.attributes=*/attrs_b,
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
      /*.family=*/{},
      /*.attributes=*/&param32,
  };
  loom_encoding_t enc64 = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.family=*/{},
      /*.attributes=*/&param64,
  };

  uint16_t id32 = 0, id64 = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &enc32, &id32));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &enc64, &id64));
  EXPECT_NE(id32, id64);
  EXPECT_EQ(module->encodings.count, 2u);
  EXPECT_EQ(loom_encoding_parameter_descriptor_index(
                &loom_module_encoding(module, id32)->attributes[0]),
            0u);
  EXPECT_EQ(loom_encoding_parameter_descriptor_index(
                &loom_module_encoding(module, id64)->attributes[0]),
            0u);

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

  EXPECT_EQ(loom_module_encoding_vtable(
                module, loom_module_encoding(module, encoding_id)),
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
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
