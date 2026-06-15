// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
  }

  void TearDown() override { loom_context_deinitialize(&context_); }

  loom_context_t context_;
};

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.name=*/IREE_SV("q8_0"),
};

TEST_F(ContextTest, FinalizeBuildsOpNameLookupTable) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      /*.traits=*/{},
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/{},
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kTestOpName,
  };
  static const loom_op_vtable_t* const kTestDialectVtables[] = {
      &kTestOpVtable,
  };

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
  const loom_op_vtable_t* vtable =
      loom_context_lookup_op_by_name(&context_, IREE_SV("test.nop"), &kind);
  ASSERT_EQ(vtable, &kTestOpVtable);
  EXPECT_EQ(kind, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0));
  EXPECT_EQ(loom_context_resolve_op(&context_, kind), &kTestOpVtable);
}

TEST_F(ContextTest, RegisterDialectSemanticsResolvesByOpKind) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      /*.traits=*/{},
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/{},
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kTestOpName,
  };
  static const loom_op_vtable_t* const kTestDialectVtables[] = {
      &kTestOpVtable,
  };
  static const loom_op_semantics_t kTestDialectSemantics[] = {
      {
          /*.phase=*/LOOM_OP_PHASE_EXECUTABLE,
          /*.contract_families=*/LOOM_CONTRACT_VECTOR_CONTRACTION,
      },
  };

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));
  IREE_ASSERT_OK(loom_context_register_dialect_semantics(
      &context_, LOOM_DIALECT_TEST, kTestDialectSemantics,
      IREE_ARRAYSIZE(kTestDialectSemantics)));

  loom_op_semantics_t semantics = loom_context_resolve_op_semantics(
      &context_, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0));
  EXPECT_EQ(semantics.phase, LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      semantics.contract_families, LOOM_CONTRACT_VECTOR_CONTRACTION));

  loom_op_semantics_t missing = loom_context_resolve_op_semantics(
      &context_, LOOM_OP_KIND(LOOM_DIALECT_TEST, 1));
  EXPECT_EQ(missing.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(missing.contract_families, 0u);
}

TEST_F(ContextTest, RegisterDialectSemanticsRequiresMatchingVtables) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      /*.traits=*/{},
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/{},
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kTestOpName,
  };
  static const loom_op_vtable_t* const kTestDialectVtables[] = {
      &kTestOpVtable,
  };
  static const loom_op_semantics_t kTestDialectSemantics[] = {
      {
          /*.phase=*/LOOM_OP_PHASE_EXECUTABLE,
      },
  };

  iree_status_t status = loom_context_register_dialect_semantics(
      &context_, LOOM_DIALECT_TEST, kTestDialectSemantics,
      IREE_ARRAYSIZE(kTestDialectSemantics));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));

  status = loom_context_register_dialect_semantics(&context_, LOOM_DIALECT_TEST,
                                                   kTestDialectSemantics, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);
}

TEST_F(ContextTest, RegisterEncodingVtableAndLookupByName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));

  EXPECT_EQ(loom_context_lookup_encoding_vtable(&context_, IREE_SV("q8_0")),
            &kQ8_0EncodingVtable);
  EXPECT_EQ(loom_context_lookup_encoding_vtable(&context_, IREE_SV("q6_k")),
            nullptr);
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsDuplicateName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));

  static const loom_encoding_vtable_t kDuplicate = {
      /*.name=*/IREE_SV("q8_0"),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_context_register_encoding_vtable(&context_, &kDuplicate));
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsMissingName) {
  static const loom_encoding_vtable_t kMissingName = {};
  iree_status_t status =
      loom_context_register_encoding_vtable(&context_, &kMissingName);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  status = loom_context_register_encoding_vtable(&context_, NULL);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

}  // namespace
}  // namespace loom
