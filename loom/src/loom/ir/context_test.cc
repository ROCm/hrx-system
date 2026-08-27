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

static const loom_bstring_t kQ8_0EncodingFormatNames[] = {
    LOOM_BSTRING_REF(2, "i8"),
};
static const loom_attr_descriptor_t kQ8_0EncodingParameters[] = {{
    /*.name=*/LOOM_BSTRING_REF(6, "format"),
    /*.attr_kind=*/LOOM_ATTR_ENUM,
    /*.flags=*/0,
    /*.enum_max_value=*/0,
    /*.enum_case_names=*/kQ8_0EncodingFormatNames,
}};
static const loom_encoding_alias_parameter_t kQ8_0EncodingAliasParameters[] = {
    {
        /*.parameter_index=*/0,
        /*.flags=*/LOOM_ENCODING_ALIAS_PARAMETER_FIXED,
        /*.value=*/loom_attr_enum(0),
    },
};
static const loom_encoding_alias_descriptor_t kQ8_0EncodingAliases[] = {
    {
        /*.name=*/LOOM_BSTRING_REF(16, "encoding.test_i8"),
        /*.parameter_count=*/IREE_ARRAYSIZE(kQ8_0EncodingAliasParameters),
        /*.parameters=*/kQ8_0EncodingAliasParameters,
    },
};
static const uint8_t kQ8_0EncodingAliasOrdinals[] = {1};

static const loom_encoding_family_descriptor_t kQ8_0EncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(4, "q8_0"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/IREE_ARRAYSIZE(kQ8_0EncodingParameters),
    /*.parameter_descriptors=*/kQ8_0EncodingParameters,
    /*.dynamic_parameter_count=*/0,
    /*.dynamic_parameter_descriptors=*/nullptr,
    /*.fixed_metadata=*/nullptr,
    /*.alias_count=*/IREE_ARRAYSIZE(kQ8_0EncodingAliases),
    /*.alias_discriminator_parameter_index=*/0,
    /*.aliases=*/kQ8_0EncodingAliases,
    /*.alias_ordinals_by_discriminator=*/kQ8_0EncodingAliasOrdinals,
};

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.descriptor=*/&kQ8_0EncodingDescriptor,
};

static const loom_type_descriptor_t kTestTypeDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(9, "test.type"),
    /*.ir_kind=*/LOOM_TYPE_DIALECT,
};

static const loom_type_registry_entry_t kTestTypeEntries[] = {
    {IREE_SV("test.type"), &kTestTypeDescriptor},
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

static iree_status_t TestMaterializeConditionRefinement(
    loom_rewriter_t* rewriter, const loom_op_t* condition_op,
    loom_value_id_t source, bool assumed_truth,
    loom_value_id_t* out_refined_value) {
  (void)rewriter;
  (void)condition_op;
  (void)assumed_truth;
  *out_refined_value = source;
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
      /*.operand_role_mask=*/{},
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
      /*.operand_role_mask=*/{},
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
          /*.condition_refinement_index=*/0,
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
          /*.condition_refinement_index=*/0,
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

TEST_F(ContextTest, ConditionRefinementsResolveThroughOpSemantics) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      /*.traits=*/{},
      /*.fixed_operand_count=*/1,
      /*.fixed_result_count=*/1,
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.operand_role_mask=*/{},
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
          /*.condition_refinement_index=*/1,
      },
  };
  static const loom_condition_refinement_descriptor_t kRefinements[] = {
      {
          /*.materialize=*/TestMaterializeConditionRefinement,
          /*.source_operand_index=*/0,
          /*.truth_flags=*/LOOM_CONDITION_REFINEMENT_TRUTH_TRUE,
      },
  };

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));
  IREE_ASSERT_OK(loom_context_register_dialect_semantics(
      &context_, LOOM_DIALECT_TEST, kTestDialectSemantics,
      IREE_ARRAYSIZE(kTestDialectSemantics)));
  IREE_ASSERT_OK(loom_context_register_condition_refinements(
      &context_, LOOM_DIALECT_TEST, kRefinements,
      IREE_ARRAYSIZE(kRefinements)));
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  const loom_condition_refinement_descriptor_t* refinement =
      loom_context_resolve_condition_refinement(
          &context_, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0));
  ASSERT_EQ(refinement, &kRefinements[0]);
  EXPECT_EQ(refinement->source_operand_index, 0);
  EXPECT_EQ(refinement->truth_flags, LOOM_CONDITION_REFINEMENT_TRUTH_TRUE);
}

TEST_F(ContextTest, FinalizeRejectsInvalidConditionRefinementIndex) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      /*.traits=*/{},
      /*.fixed_operand_count=*/1,
      /*.fixed_result_count=*/1,
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.operand_role_mask=*/{},
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
          /*.condition_refinement_index=*/1,
      },
  };

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));
  IREE_ASSERT_OK(loom_context_register_dialect_semantics(
      &context_, LOOM_DIALECT_TEST, kTestDialectSemantics,
      IREE_ARRAYSIZE(kTestDialectSemantics)));
  iree_status_t status = loom_context_finalize(&context_);
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
          /*.parameter_count=*/0,
          /*.primary_parameter_index=*/
          LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER,
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
          /*.parameter_count=*/0,
          /*.primary_parameter_index=*/
          LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER,
      },
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_parameterized_attrs(
          &context_, LOOM_DIALECT_TEST, kFamilies, IREE_ARRAYSIZE(kFamilies)));
}

TEST_F(ContextTest, ParameterizedAttributeRegistrationRejectsInvalidPrimary) {
  static const uint8_t kFamilyName[] =
      "\x0C"
      "test.compact";
  static const loom_parameterized_attr_descriptor_t kFamilies[] = {
      {
          /*.name=*/kFamilyName,
          /*.kind=*/LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 0),
          /*.parameter_count=*/0,
          /*.primary_parameter_index=*/0,
      },
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_parameterized_attrs(
          &context_, LOOM_DIALECT_TEST, kFamilies, IREE_ARRAYSIZE(kFamilies)));
}

TEST_F(ContextTest, ParameterizedAttributeRegistrationRejectsOptionalPrimary) {
  static const uint8_t kFamilyName[] =
      "\x0C"
      "test.compact";
  static const uint8_t kParameterName[] =
      "\x05"
      "value";
  static const loom_attr_descriptor_t kParameters[] = {
      {
          /*.name=*/kParameterName,
          /*.attr_kind=*/LOOM_ATTR_I64,
          /*.flags=*/LOOM_ATTR_OPTIONAL,
      },
  };
  static const loom_parameterized_attr_descriptor_t kFamilies[] = {
      {
          /*.name=*/kFamilyName,
          /*.kind=*/LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 0),
          /*.parameter_count=*/IREE_ARRAYSIZE(kParameters),
          /*.primary_parameter_index=*/0,
          /*.parameter_descriptors=*/kParameters,
      },
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_parameterized_attrs(
          &context_, LOOM_DIALECT_TEST, kFamilies, IREE_ARRAYSIZE(kFamilies)));
}

TEST_F(ContextTest, DialectTypeDescriptorsResolveAfterFinalization) {
  IREE_ASSERT_OK(loom_context_register_type_descriptors(
      &context_, kTestTypeEntries, IREE_ARRAYSIZE(kTestTypeEntries)));
  EXPECT_EQ(loom_context_lookup_type_by_name(&context_, IREE_SV("test.type")),
            nullptr);

  IREE_ASSERT_OK(loom_context_finalize(&context_));
  EXPECT_EQ(loom_context_lookup_type_by_name(&context_, IREE_SV("test.type")),
            &kTestTypeDescriptor);
  EXPECT_EQ(
      loom_context_lookup_type_by_name(&context_, IREE_SV("test.missing")),
      nullptr);
}

TEST_F(ContextTest, DialectTypeRegistrationRejectsDuplicateNames) {
  IREE_ASSERT_OK(loom_context_register_type_descriptors(
      &context_, kTestTypeEntries, IREE_ARRAYSIZE(kTestTypeEntries)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_context_register_type_descriptors(&context_, kTestTypeEntries,
                                             IREE_ARRAYSIZE(kTestTypeEntries)));
}

TEST_F(ContextTest, DialectTypeRegistrationRejectsInconsistentNames) {
  static const loom_type_registry_entry_t kInconsistentEntries[] = {
      {IREE_SV("test.other"), &kTestTypeDescriptor},
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_context_register_type_descriptors(
                            &context_, kInconsistentEntries,
                            IREE_ARRAYSIZE(kInconsistentEntries)));
}

TEST_F(ContextTest, RegisterEncodingVtableAndLookupByName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  const loom_encoding_name_resolution_t family_resolution =
      loom_context_resolve_encoding_name(&context_, IREE_SV("q8_0"));
  loom_encoding_family_id_t family_id = family_resolution.family_id;
  EXPECT_NE(family_id, LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_EQ(family_resolution.alias, nullptr);
  EXPECT_EQ(loom_context_resolve_encoding_vtable(&context_, family_id),
            &kQ8_0EncodingVtable);
  const loom_encoding_name_resolution_t alias_resolution =
      loom_context_resolve_encoding_name(&context_,
                                         IREE_SV("encoding.test_i8"));
  EXPECT_EQ(alias_resolution.family_id, family_id);
  EXPECT_EQ(alias_resolution.alias, &kQ8_0EncodingAliases[0]);
  EXPECT_EQ(
      loom_context_resolve_encoding_name(&context_, IREE_SV("q6_k")).family_id,
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
      /*.family_flags=*/{},
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

TEST_F(ContextTest, RegisterEncodingVtableRejectsMalformedFixedMetadata) {
  static const loom_encoding_family_fixed_metadata_t kFixedMetadata = {
      /*.operand_summary=*/{},
      /*.required_auxiliary_keys=*/{},
      /*.record=*/
      {
          /*.logical_element_count=*/32,
          /*.storage_byte_count=*/18,
          /*.required_alignment=*/3,
      },
  };
  static const loom_encoding_family_descriptor_t kMalformedDescriptor = {
      /*.name=*/LOOM_BSTRING_REF(9, "malformed"),
      /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
      /*.family_flags=*/{},
      /*.parameter_count=*/{},
      /*.parameter_descriptors=*/{},
      /*.dynamic_parameter_count=*/{},
      /*.dynamic_parameter_descriptors=*/{},
      /*.fixed_metadata=*/&kFixedMetadata,
  };
  static const loom_encoding_vtable_t kMalformedVtable = {
      /*.descriptor=*/&kMalformedDescriptor,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_context_register_encoding_vtable(&context_, &kMalformedVtable));
}

TEST_F(ContextTest, RegisterEncodingVtableRestrictsFixedMetadataToSchemas) {
  static const loom_encoding_family_fixed_metadata_t kFixedMetadata = {};
  static const loom_encoding_family_descriptor_t kMalformedDescriptor = {
      /*.name=*/LOOM_BSTRING_REF(9, "malformed"),
      /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      /*.family_flags=*/{},
      /*.parameter_count=*/{},
      /*.parameter_descriptors=*/{},
      /*.dynamic_parameter_count=*/{},
      /*.dynamic_parameter_descriptors=*/{},
      /*.fixed_metadata=*/&kFixedMetadata,
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
