// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/requirements.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

// clang-format off
static const uint8_t kRequirementStrings[] =
    LOOM_BSTRING_LITERAL(0, "")
    LOOM_BSTRING_LITERAL(9, "test.core")
    LOOM_BSTRING_LITERAL(11, "test.target")
    LOOM_BSTRING_LITERAL(13, "test.features")
    LOOM_BSTRING_LITERAL(8, "test.gpr")
    LOOM_BSTRING_LITERAL(3, "dst")
    LOOM_BSTRING_LITERAL(3, "lhs")
    LOOM_BSTRING_LITERAL(3, "rhs")
    LOOM_BSTRING_LITERAL(8, "test.alu")
    LOOM_BSTRING_LITERAL(12, "test.alu.i32")
    LOOM_BSTRING_LITERAL(12, "test.add.i32")
    LOOM_BSTRING_LITERAL(7, "add.i32")
    LOOM_BSTRING_LITERAL(15, "integer.add.i32");
// clang-format on

enum {
  REQUIREMENT_STRING_empty = 0,
  REQUIREMENT_STRING_set_key = REQUIREMENT_STRING_empty + sizeof(""),
  REQUIREMENT_STRING_target_key =
      REQUIREMENT_STRING_set_key + sizeof("test.core"),
  REQUIREMENT_STRING_feature_key =
      REQUIREMENT_STRING_target_key + sizeof("test.target"),
  REQUIREMENT_STRING_reg_gpr =
      REQUIREMENT_STRING_feature_key + sizeof("test.features"),
  REQUIREMENT_STRING_field_dst =
      REQUIREMENT_STRING_reg_gpr + sizeof("test.gpr"),
  REQUIREMENT_STRING_field_lhs = REQUIREMENT_STRING_field_dst + sizeof("dst"),
  REQUIREMENT_STRING_field_rhs = REQUIREMENT_STRING_field_lhs + sizeof("lhs"),
  REQUIREMENT_STRING_resource_alu =
      REQUIREMENT_STRING_field_rhs + sizeof("rhs"),
  REQUIREMENT_STRING_schedule_alu =
      REQUIREMENT_STRING_resource_alu + sizeof("test.alu"),
  REQUIREMENT_STRING_descriptor_add =
      REQUIREMENT_STRING_schedule_alu + sizeof("test.alu.i32"),
  REQUIREMENT_STRING_mnemonic_add =
      REQUIREMENT_STRING_descriptor_add + sizeof("test.add.i32"),
  REQUIREMENT_STRING_semantic_add =
      REQUIREMENT_STRING_mnemonic_add + sizeof("add.i32"),
  REQUIREMENT_STRING_END =
      REQUIREMENT_STRING_semantic_add + sizeof("integer.add.i32"),
};

static_assert(REQUIREMENT_STRING_END == sizeof(kRequirementStrings) - 1,
              "requirement test string offsets must cover the table payload");

#define REQUIREMENT_STRING_OFFSET(field) \
  static_cast<loom_bstring_table_offset_t>(REQUIREMENT_STRING_##field)

struct RequirementTables {
  // Descriptor rows owned by the test descriptor set.
  loom_low_descriptor_t descriptors[1];
  // Sorted descriptor key map owned by the test descriptor set.
  loom_low_descriptor_ref_t descriptor_refs[1];
  // Operand/result rows referenced by the test descriptor.
  loom_low_operand_t operands[3];
  // Register class rows accepted by the test operands.
  loom_low_reg_class_t reg_classes[1];
  // Register-class alternative rows accepted by the test operands.
  loom_low_reg_class_alt_t reg_class_alts[1];
  // Schedule class rows referenced by the test descriptor.
  loom_low_schedule_class_t schedule_classes[1];
  // Issue-use rows referenced by the test schedule class.
  loom_low_issue_use_t issue_uses[1];
  // Resource rows referenced by the test issue-use rows.
  loom_low_resource_t resources[1];
  // Descriptor set assembled from the rows above.
  loom_low_descriptor_set_t set;
};

void InitializeRequirementTables(RequirementTables* tables) {
  *tables = {};
  for (loom_low_operand_t& operand : tables->operands) {
    operand.register_part_id = LOOM_LOW_REGISTER_PART_NONE;
  }
  for (loom_low_reg_class_t& reg_class : tables->reg_classes) {
    reg_class.full_register_part_mask = 1;
  }

  tables->reg_classes[0].name_string_offset =
      REQUIREMENT_STRING_OFFSET(reg_gpr);
  tables->reg_classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY;
  tables->reg_classes[0].alloc_unit_bits = 32;
  tables->reg_classes[0].spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  tables->reg_classes[0].spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK;

  tables->reg_class_alts[0].reg_class_id = 0;
  tables->reg_class_alts[0].flags = LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED;

  tables->operands[0].field_name_string_offset =
      REQUIREMENT_STRING_OFFSET(field_dst);
  tables->operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  tables->operands[0].reg_class_alt_start = 0;
  tables->operands[0].reg_class_alt_count = 1;
  tables->operands[0].unit_count = 1;

  tables->operands[1].field_name_string_offset =
      REQUIREMENT_STRING_OFFSET(field_lhs);
  tables->operands[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[1].reg_class_alt_start = 0;
  tables->operands[1].reg_class_alt_count = 1;
  tables->operands[1].unit_count = 1;

  tables->operands[2].field_name_string_offset =
      REQUIREMENT_STRING_OFFSET(field_rhs);
  tables->operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[2].reg_class_alt_start = 0;
  tables->operands[2].reg_class_alt_count = 1;
  tables->operands[2].unit_count = 1;

  tables->resources[0].name_string_offset =
      REQUIREMENT_STRING_OFFSET(resource_alu);
  tables->resources[0].capacity_per_cycle = 1;
  tables->resources[0].kind = LOOM_LOW_RESOURCE_KIND_SCALAR_ALU;

  tables->issue_uses[0].resource_id = 0;
  tables->issue_uses[0].cycles = 1;
  tables->issue_uses[0].units = 1;

  tables->schedule_classes[0].name_string_offset =
      REQUIREMENT_STRING_OFFSET(schedule_alu);
  tables->schedule_classes[0].latency_cycles = 1;
  tables->schedule_classes[0].latency_kind = LOOM_LOW_LATENCY_KIND_EXACT;
  tables->schedule_classes[0].issue_use_start = 0;
  tables->schedule_classes[0].issue_use_count = 1;
  tables->schedule_classes[0].model_quality = LOOM_LOW_MODEL_QUALITY_EXACT;

  tables->descriptors[0].key_string_offset =
      REQUIREMENT_STRING_OFFSET(descriptor_add);
  tables->descriptors[0].stable_id =
      loom_low_descriptor_stable_id_from_key(IREE_SV("test.add.i32"));
  tables->descriptors[0].mnemonic_string_offset =
      REQUIREMENT_STRING_OFFSET(mnemonic_add);
  tables->descriptors[0].semantic_tag_string_offset =
      REQUIREMENT_STRING_OFFSET(semantic_add);
  tables->descriptors[0].canonical_asm_form_ordinal =
      LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  tables->descriptors[0].encoding_id = 1;
  tables->descriptors[0].operand_start = 0;
  tables->descriptors[0].operand_count = 3;
  tables->descriptors[0].result_count = 1;
  tables->descriptors[0].schedule_class_id = 0;
  tables->descriptors[0].flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE;

  tables->descriptor_refs[0].key_string_offset =
      REQUIREMENT_STRING_OFFSET(descriptor_add);
  tables->descriptor_refs[0].descriptor_ordinal = 0;

  tables->set.abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION;
  tables->set.generator_version = 7;
  tables->set.stable_id =
      loom_low_descriptor_stable_id_from_key(IREE_SV("test.core"));
  tables->set.target_stable_id =
      loom_low_descriptor_stable_id_from_key(IREE_SV("test.target"));
  tables->set.key_string_offset = REQUIREMENT_STRING_OFFSET(set_key);
  tables->set.target_key_string_offset = REQUIREMENT_STRING_OFFSET(target_key);
  tables->set.feature_key_string_offset =
      REQUIREMENT_STRING_OFFSET(feature_key);
  tables->set.string_table.data = kRequirementStrings;
  tables->set.string_table.data_length = sizeof(kRequirementStrings) - 1;
  tables->set.descriptors = tables->descriptors;
  tables->set.descriptor_count = IREE_ARRAYSIZE(tables->descriptors);
  tables->set.descriptor_refs = tables->descriptor_refs;
  tables->set.descriptor_ref_count = IREE_ARRAYSIZE(tables->descriptor_refs);
  tables->set.operands = tables->operands;
  tables->set.operand_count = IREE_ARRAYSIZE(tables->operands);
  tables->set.reg_classes = tables->reg_classes;
  tables->set.reg_class_count = IREE_ARRAYSIZE(tables->reg_classes);
  tables->set.reg_class_alts = tables->reg_class_alts;
  tables->set.reg_class_alt_count = IREE_ARRAYSIZE(tables->reg_class_alts);
  tables->set.schedule_classes = tables->schedule_classes;
  tables->set.schedule_class_count = IREE_ARRAYSIZE(tables->schedule_classes);
  tables->set.issue_uses = tables->issue_uses;
  tables->set.issue_use_count = IREE_ARRAYSIZE(tables->issue_uses);
  tables->set.resources = tables->resources;
  tables->set.resource_count = IREE_ARRAYSIZE(tables->resources);
}

TEST(LowDescriptorRequirementsTest, VerifiesTargetLowFoundation) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify_requirements(
      &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowRegisterTypeResolverTest, ResolvesCompactRegisterTypes) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  loom_low_register_type_resolver_t resolver =
      loom_low_register_type_resolver_for_descriptor_set(&tables.set);

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = loom_low_register_type_resolver_try_resolve(
      &resolver, loom_low_register_type(tables.set.stable_id, 0, 4),
      &descriptor_register_class_id, &descriptor_register_class);
  ASSERT_TRUE(found);
  ASSERT_NE(descriptor_register_class, nullptr);
  EXPECT_EQ(descriptor_register_class_id, 0);
  EXPECT_TRUE(iree_string_view_equal(
      loom_low_descriptor_set_string(
          &tables.set, descriptor_register_class->name_string_offset),
      IREE_SV("test.gpr")));
}

TEST(LowRegisterTypeResolverTest, RejectsUnknownAndNonRegisterTypes) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  loom_low_register_type_resolver_t resolver =
      loom_low_register_type_resolver_for_descriptor_set(&tables.set);

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = loom_low_register_type_resolver_try_resolve(
      &resolver,
      loom_low_register_type(
          /*descriptor_set_stable_id=*/tables.set.stable_id + 1, 0, 1),
      &descriptor_register_class_id, &descriptor_register_class);
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);

  found = loom_low_register_type_resolver_try_resolve(
      &resolver,
      loom_low_register_type(tables.set.stable_id,
                             (uint16_t)tables.set.reg_class_count, 1),
      &descriptor_register_class_id, &descriptor_register_class);
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);

  found = loom_low_register_type_resolver_try_resolve(
      &resolver, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      &descriptor_register_class_id, &descriptor_register_class);
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);
}

TEST(LowRegisterClassLookupTest, LooksUpDescriptorRegisterClassNames) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = loom_low_descriptor_set_lookup_register_class(
      &tables.set, IREE_SV("test.gpr"), &descriptor_register_class_id,
      &descriptor_register_class);
  ASSERT_TRUE(found);
  ASSERT_NE(descriptor_register_class, nullptr);
  EXPECT_EQ(descriptor_register_class_id, 0);
  EXPECT_TRUE(iree_string_view_equal(
      loom_low_descriptor_set_string(
          &tables.set, descriptor_register_class->name_string_offset),
      IREE_SV("test.gpr")));

  found = loom_low_descriptor_set_lookup_register_class(
      &tables.set, IREE_SV("test.missing"), &descriptor_register_class_id,
      &descriptor_register_class);
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);
}

TEST(LowDescriptorRequirementsTest, RejectsMissingMnemonic) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.descriptors[0].mnemonic_string_offset = LOOM_LOW_STRING_OFFSET_NONE;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest, RejectsMissingSemanticTag) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.descriptors[0].semantic_tag_string_offset =
      LOOM_LOW_STRING_OFFSET_NONE;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest, RejectsOperandWithoutRegClassAlternative) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.operands[1].reg_class_alt_count = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest,
     AcceptsZeroCostPureScheduleWithoutIssueUse) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.schedule_classes[0].latency_cycles = 0;
  tables.schedule_classes[0].issue_use_count = 0;
  tables.set.issue_use_count = 0;

  IREE_ASSERT_OK(loom_low_descriptor_set_verify_requirements(
      &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest,
     RejectsNonZeroLatencyScheduleWithoutIssueUse) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.schedule_classes[0].issue_use_count = 0;
  tables.set.issue_use_count = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

}  // namespace
}  // namespace loom
