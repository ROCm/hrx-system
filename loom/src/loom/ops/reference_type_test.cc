// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/reference_type.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/type/ops.h"
#include "loom/ops/type_builders.h"
#include "loom/testing/diagnostic_matchers.h"

namespace {

class ReferenceTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_type_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TYPE,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("types"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void Declare(iree_string_view_t local_name, iree_string_view_t provider,
               loom_symbol_ref_t* out_symbol) {
    loom_string_id_t local_id, provider_id, payload_id;
    IREE_ASSERT_OK(loom_module_intern_string(module_, local_name, &local_id));
    IREE_ASSERT_OK(loom_module_intern_string(module_, provider, &provider_id));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("decoder"), &payload_id));
    IREE_ASSERT_OK(
        loom_module_add_symbol(module_, local_id, &out_symbol->symbol_id));
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_type_decl_build(&builder_, provider_id, payload_id,
                                        *out_symbol, LOOM_LOCATION_UNKNOWN,
                                        &op));
  }

  // Shared storage outlives the module and its interned keys.
  iree_arena_block_pool_t block_pool_;
  // Owns the registered type-declaration operation metadata.
  loom_context_t context_;
  // Owns the authored symbols and canonical handle types.
  loom_module_t* module_ = nullptr;
  // Appends declarations to the module body.
  loom_builder_t builder_ = {};
};

TEST_F(ReferenceTypeTest, AliasesInternByExternalKeyAndOwnership) {
  loom_symbol_ref_t decoder = {}, alias = {}, other = {};
  Declare(IREE_SV("Decoder"), IREE_SV("input"), &decoder);
  Declare(IREE_SV("Alias"), IREE_SV("input"), &alias);
  Declare(IREE_SV("Other"), IREE_SV("other"), &other);
  loom_type_t owned, aliased, borrowed, distinct;
  IREE_ASSERT_OK(loom_ref_type_make(module_, decoder, &owned));
  IREE_ASSERT_OK(loom_ref_type_make(module_, alias, &aliased));
  IREE_ASSERT_OK(loom_ptr_type_make(module_, decoder, &borrowed));
  IREE_ASSERT_OK(loom_ref_type_make(module_, other, &distinct));
  EXPECT_TRUE(loom_ref_type_isa(owned));
  EXPECT_TRUE(loom_ptr_type_isa(borrowed));
  EXPECT_FALSE(loom_ref_type_isa(borrowed));
  EXPECT_TRUE(loom_type_equal(owned, aliased));
  EXPECT_EQ(loom_type_reference_declaration_id(aliased), decoder.symbol_id);
  EXPECT_FALSE(loom_type_equal(owned, borrowed));
  EXPECT_FALSE(loom_type_equal(owned, distinct));
  EXPECT_EQ(module_->types.count, 3u);
}

TEST_F(ReferenceTypeTest, UndefinedDeclarationCannotConstructAHandle) {
  loom_string_id_t local_id;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("Missing"), &local_id));
  loom_symbol_ref_t missing = {};
  IREE_ASSERT_OK(loom_module_add_symbol(module_, local_id, &missing.symbol_id));
  loom_type_t type;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_ref_type_make(module_, missing, &type));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_ptr_type_make(module_, missing, &type));
}

TEST_F(ReferenceTypeTest, DeclarationKeysValidateBothStringFields) {
  loom_symbol_ref_t decoder = {};
  Declare(IREE_SV("Decoder"), IREE_SV("input"), &decoder);
  loom_op_t* op = module_->symbols.entries[decoder.symbol_id].defining_op;
  const loom_string_id_t good_namespace = loom_type_decl_namespace(op);
  const loom_string_id_t good_name = loom_type_decl_type_name(op);
  const iree_string_view_t invalid[] = {
      IREE_SVL(""),
      IREE_SVL("a\0b"),
      IREE_SVL("\xC0\xAF"),
  };
  for (iree_string_view_t value : invalid) {
    loom_string_id_t value_id;
    IREE_ASSERT_OK(loom_module_intern_string(module_, value, &value_id));
    IREE_ASSERT_OK(
        loom_type_decl_set_namespace(module_, op, loom_attr_string(value_id)));
    IREE_ASSERT_OK(
        loom_type_decl_set_type_name(module_, op, loom_attr_string(value_id)));
    loom::testing::DiagnosticEmissionCapture capture;
    IREE_ASSERT_OK(loom_type_decl_verify(module_, op, capture.emitter()));
    ASSERT_EQ(capture.emissions.size(), 2u);
    EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_TYPE_021);
    EXPECT_EQ(capture.emissions[0].string_params[0], "namespace");
    EXPECT_EQ(capture.emissions[1].error, LOOM_ERR_TYPE_021);
    EXPECT_EQ(capture.emissions[1].string_params[0], "type_name");
  }
  IREE_ASSERT_OK(loom_type_decl_set_namespace(
      module_, op, loom_attr_string(good_namespace)));
  IREE_ASSERT_OK(
      loom_type_decl_set_type_name(module_, op, loom_attr_string(good_name)));
  loom::testing::DiagnosticEmissionCapture capture;
  IREE_ASSERT_OK(loom_type_decl_verify(module_, op, capture.emitter()));
  EXPECT_TRUE(capture.emissions.empty());
}

}  // namespace
