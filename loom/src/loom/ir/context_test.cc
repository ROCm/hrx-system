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

static const loom_encoding_family_descriptor_t kQ8_0EncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(4, "q8_0"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.descriptor=*/&kQ8_0EncodingDescriptor,
};

static bool TestEncodingIsStaticValid(const loom_module_t* module,
                                      const loom_encoding_t* encoding) {
  (void)module;
  (void)encoding;
  return true;
}

static iree_status_t TestEncodingDiagnoseStatic(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  (void)module;
  (void)encoding;
  (void)op;
  (void)emitter;
  return iree_ok_status();
}

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

TEST_F(ContextTest, ParameterizedAttributeFamiliesResolveByKindAndName) {
  static const uint8_t kFamilyName[] =
      "\x09"
      "test.tile";
  static const loom_parameterized_attr_descriptor_t kFamilies[] = {
      {
          /*.name=*/kFamilyName,
          /*.kind=*/LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 0),
      },
  };

  IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
      &context_, LOOM_DIALECT_TEST, kFamilies, IREE_ARRAYSIZE(kFamilies)));
  EXPECT_EQ(
      loom_context_resolve_parameterized_attr(&context_, kFamilies[0].kind),
      &kFamilies[0]);
  EXPECT_EQ(loom_context_lookup_parameterized_attr_by_name(
                &context_, IREE_SV("test.tile")),
            nullptr);

  IREE_ASSERT_OK(loom_context_finalize(&context_));
  EXPECT_EQ(loom_context_lookup_parameterized_attr_by_name(
                &context_, IREE_SV("test.tile")),
            &kFamilies[0]);
  EXPECT_EQ(loom_context_lookup_parameterized_attr_by_name(
                &context_, IREE_SV("test.missing")),
            nullptr);
}

TEST_F(ContextTest, ParameterizedAttributeRegistrationRejectsWrongKind) {
  static const uint8_t kFamilyName[] =
      "\x09"
      "test.tile";
  static const loom_parameterized_attr_descriptor_t kFamilies[] = {
      {
          /*.name=*/kFamilyName,
          /*.kind=*/LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 1),
      },
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_parameterized_attrs(
          &context_, LOOM_DIALECT_TEST, kFamilies, IREE_ARRAYSIZE(kFamilies)));
}

TEST_F(ContextTest, RegisterEncodingVtableAndLookupByName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  loom_encoding_family_id_t family_id =
      loom_context_lookup_encoding_family_by_name(&context_, IREE_SV("q8_0"));
  EXPECT_NE(family_id, LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_EQ(loom_context_resolve_encoding_vtable(&context_, family_id),
            &kQ8_0EncodingVtable);
  EXPECT_EQ(
      loom_context_lookup_encoding_family_by_name(&context_, IREE_SV("q6_k")),
      LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_EQ(loom_context_resolve_encoding_vtable(
                &context_, LOOM_ENCODING_FAMILY_ID_INVALID),
            nullptr);
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsDuplicateName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));

  static const loom_encoding_family_descriptor_t kDuplicateDescriptor = {
      /*.name=*/LOOM_BSTRING_REF(4, "q8_0"),
      /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
  };
  static const loom_encoding_vtable_t kDuplicate = {
      /*.descriptor=*/&kDuplicateDescriptor,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_context_register_encoding_vtable(&context_, &kDuplicate));
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsMissingName) {
  static const loom_encoding_family_descriptor_t kMissingNameDescriptor = {};
  static const loom_encoding_vtable_t kMissingName = {
      /*.descriptor=*/&kMissingNameDescriptor,
  };
  iree_status_t status =
      loom_context_register_encoding_vtable(&context_, &kMissingName);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  status = loom_context_register_encoding_vtable(&context_, NULL);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsMissingParameterDescriptors) {
  static const loom_encoding_family_descriptor_t kMalformedDescriptor = {
      /*.name=*/LOOM_BSTRING_REF(9, "malformed"),
      /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
      /*.parameter_count=*/1,
      /*.parameter_descriptors=*/nullptr,
  };
  static const loom_encoding_vtable_t kMalformedVtable = {
      /*.descriptor=*/&kMalformedDescriptor,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_encoding_vtable(&context_, &kMalformedVtable));
}

TEST_F(ContextTest, RegisterEncodingVtableRequiresPairedStaticCallbacks) {
  static const loom_encoding_vtable_t kPredicateOnlyVtable = {
      /*.descriptor=*/&kQ8_0EncodingDescriptor,
      /*.is_static_valid=*/TestEncodingIsStaticValid,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_encoding_vtable(&context_, &kPredicateOnlyVtable));

  static const loom_encoding_vtable_t kDiagnosticOnlyVtable = {
      /*.descriptor=*/&kQ8_0EncodingDescriptor,
      /*.is_static_valid=*/{},
      /*.diagnose_static=*/TestEncodingDiagnoseStatic,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_encoding_vtable(&context_, &kDiagnosticOnlyVtable));
}

}  // namespace
}  // namespace loom
