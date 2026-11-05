// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/requirements.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static const char kRequirementStrings[] =
    ""
    "test.core"
    "test.target"
    "test.features"
    "test.gpr"
    "dst"
    "lhs"
    "rhs"
    "test.alu"
    "test.alu.i32"
    "test.add.i32"
    "add.i32"
    "integer.add.i32";

enum : loom_string_ref_t {
  REQUIREMENT_STRING_empty = LOOM_STRING_REF(0, 0, 0),
  REQUIREMENT_STRING_set_key = LOOM_STRING_REF(0, 0, 9),
  REQUIREMENT_STRING_target_key = LOOM_STRING_REF(0, 9, 11),
  REQUIREMENT_STRING_feature_key = LOOM_STRING_REF(0, 20, 13),
  REQUIREMENT_STRING_reg_gpr = LOOM_STRING_REF(0, 33, 8),
  REQUIREMENT_STRING_field_dst = LOOM_STRING_REF(0, 41, 3),
  REQUIREMENT_STRING_field_lhs = LOOM_STRING_REF(0, 44, 3),
  REQUIREMENT_STRING_field_rhs = LOOM_STRING_REF(0, 47, 3),
  REQUIREMENT_STRING_resource_alu = LOOM_STRING_REF(0, 50, 8),
  REQUIREMENT_STRING_schedule_alu = LOOM_STRING_REF(0, 58, 12),
  REQUIREMENT_STRING_descriptor_add = LOOM_STRING_REF(0, 70, 12),
  REQUIREMENT_STRING_mnemonic_add = LOOM_STRING_REF(0, 82, 7),
  REQUIREMENT_STRING_semantic_add = LOOM_STRING_REF(0, 89, 15),
};

#define REQUIREMENT_STRING_REF(field) REQUIREMENT_STRING_##field

struct RequirementTables {
  // Stable descriptor identities owned by the test descriptor set.
  loom_low_descriptor_identity_t descriptor_identities[1];
  // Stable IDs indexed by descriptor ordinal.
  uint64_t descriptor_stable_ids[1];
  // Descriptor rows owned by the test descriptor set.
  loom_low_descriptor_t descriptors[1];
  // Structural layouts referenced by the test descriptors.
  loom_low_descriptor_layout_t descriptor_layouts[1];
  // View-owned descriptor rows paired with the structural rows.
  loom_low_descriptor_view_t descriptor_views[1];
  // Descriptor ordinals ordered by stable identity.
  uint16_t descriptor_refs[1];
  // Operand/result rows referenced by the test descriptor.
  loom_low_operand_t operands[3];
  // Register class rows accepted by the test operands.
  loom_low_reg_class_t reg_classes[1];
  // Register-class alternative rows accepted by the test operands.
  loom_low_reg_class_alt_t reg_class_alts[1];
  // Schedule class rows referenced by the test descriptor.
  loom_low_schedule_class_t schedule_classes[1];
  // Stable semantic names indexed by schedule class ID.
  loom_string_ref_t schedule_class_names[1];
  // Issue-use rows referenced by the test schedule class.
  loom_low_issue_use_t issue_uses[1];
  // Resource rows referenced by the test issue-use rows.
  loom_low_resource_t resources[1];
  // Schedule model assembled from the rows above.
  loom_low_schedule_model_t schedule_model;
  // Explicit catalog entry naming the schedule model above.
  loom_low_schedule_model_entry_t schedule_models[1];
  // Descriptor set assembled from the rows above.
  loom_low_descriptor_set_t set;
};

void InitializeRequirementTables(RequirementTables* tables) {
  *tables = {};
  for (loom_low_reg_class_t& reg_class : tables->reg_classes) {
    reg_class.full_register_part_mask = 1;
  }

  tables->reg_classes[0].name_string_ref = REQUIREMENT_STRING_REF(reg_gpr);
  tables->reg_classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY;
  tables->reg_classes[0].alloc_unit_bits = 32;
  tables->reg_classes[0].spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  tables->reg_classes[0].spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK;

  tables->reg_class_alts[0].reg_class_id = 0;
  tables->reg_class_alts[0].register_part_id = LOOM_LOW_REGISTER_PART_NONE;
  tables->reg_class_alts[0].flags = LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED;

  tables->operands[0].field_name_string_ref = REQUIREMENT_STRING_REF(field_dst);
  tables->operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  tables->operands[0].reg_class_alt_start = 0;
  tables->operands[0].reg_class_alt_count = 1;
  tables->operands[0].unit_count = 1;

  tables->operands[1].field_name_string_ref = REQUIREMENT_STRING_REF(field_lhs);
  tables->operands[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[1].reg_class_alt_start = 0;
  tables->operands[1].reg_class_alt_count = 1;
  tables->operands[1].unit_count = 1;

  tables->operands[2].field_name_string_ref = REQUIREMENT_STRING_REF(field_rhs);
  tables->operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[2].reg_class_alt_start = 0;
  tables->operands[2].reg_class_alt_count = 1;
  tables->operands[2].unit_count = 1;

  tables->resources[0].name_string_ref = REQUIREMENT_STRING_REF(resource_alu);
  tables->resources[0].capacity_per_cycle = 1;
  tables->resources[0].kind = LOOM_LOW_RESOURCE_KIND_SCALAR_ALU;

  tables->issue_uses[0].resource_id = 0;
  tables->issue_uses[0].cycles = 1;
  tables->issue_uses[0].units = 1;

  tables->schedule_class_names[0] = REQUIREMENT_STRING_REF(schedule_alu);
  tables->schedule_classes[0].latency_cycles = 1;
  tables->schedule_classes[0].latency_kind = LOOM_LOW_LATENCY_KIND_EXACT;
  tables->schedule_classes[0].issue_use_start = 0;
  tables->schedule_classes[0].issue_use_count = 1;
  tables->schedule_classes[0].minimum_issue_cycles = 1;
  tables->schedule_classes[0].model_quality = LOOM_LOW_MODEL_QUALITY_EXACT;

  tables->descriptor_identities[0].key_string_ref =
      REQUIREMENT_STRING_REF(descriptor_add);
  tables->descriptor_identities[0].semantic_tag_string_ref =
      REQUIREMENT_STRING_REF(semantic_add);
  tables->descriptor_stable_ids[0] =
      loom_low_descriptor_stable_id_from_key(IREE_SV("test.add.i32"));
  tables->descriptors[0].mnemonic_string_ref =
      REQUIREMENT_STRING_REF(mnemonic_add);
  tables->descriptor_views[0].canonical_asm_form_ordinal =
      LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  tables->descriptors[0].encoding_id = 1;
  tables->descriptors[0].operand_start = 0;
  tables->descriptors[0].operand_count = 3;
  tables->descriptors[0].result_count = 1;
  tables->descriptor_layouts[0].minimum_packet_operand_count = 2;
  tables->descriptor_views[0].schedule_class_id = 0;
  tables->descriptors[0].flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE;

  tables->descriptor_refs[0] = 0;

  tables->set.abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION;
  tables->set.generator_version = 7;
  tables->set.stable_id =
      loom_low_descriptor_stable_id_from_key(IREE_SV("test.core"));
  tables->set.target_stable_id =
      loom_low_descriptor_stable_id_from_key(IREE_SV("test.target"));
  tables->set.key_string_ref = REQUIREMENT_STRING_REF(set_key);
  tables->set.target_key_string_ref = REQUIREMENT_STRING_REF(target_key);
  tables->set.feature_key_string_ref = REQUIREMENT_STRING_REF(feature_key);
  tables->set.string_pool.data[0] = kRequirementStrings;
  tables->set.string_pool.data_length[0] = sizeof(kRequirementStrings) - 1;
  tables->set.descriptor_identities = tables->descriptor_identities;
  tables->set.descriptor_stable_ids = tables->descriptor_stable_ids;
  tables->set.descriptor_identity_count =
      IREE_ARRAYSIZE(tables->descriptor_identities);
  tables->set.descriptors = tables->descriptors;
  tables->set.descriptor_layouts = tables->descriptor_layouts;
  tables->set.descriptor_layout_count =
      IREE_ARRAYSIZE(tables->descriptor_layouts);
  tables->set.descriptor_views = tables->descriptor_views;
  tables->set.descriptor_count = IREE_ARRAYSIZE(tables->descriptors);
  tables->set.descriptor_ordinal_count = IREE_ARRAYSIZE(tables->descriptors);
  tables->set.descriptor_refs = tables->descriptor_refs;
  tables->set.descriptor_ref_count = IREE_ARRAYSIZE(tables->descriptor_refs);
  tables->set.operands = tables->operands;
  tables->set.operand_count = IREE_ARRAYSIZE(tables->operands);
  static const uint16_t operand_indices[] = {0, 1, 2};
  tables->set.operand_indices = operand_indices;
  tables->set.operand_index_count = IREE_ARRAYSIZE(operand_indices);
  tables->set.reg_classes = tables->reg_classes;
  tables->set.reg_class_count = IREE_ARRAYSIZE(tables->reg_classes);
  tables->set.reg_class_alts = tables->reg_class_alts;
  tables->set.reg_class_alt_count = IREE_ARRAYSIZE(tables->reg_class_alts);
  tables->schedule_model.schedule_classes = tables->schedule_classes;
  tables->schedule_model.schedule_class_row_count =
      IREE_ARRAYSIZE(tables->schedule_classes);
  tables->schedule_model.issue_uses = tables->issue_uses;
  tables->schedule_model.issue_use_count = IREE_ARRAYSIZE(tables->issue_uses);
  tables->schedule_model.resources = tables->resources;
  tables->schedule_model.resource_count = IREE_ARRAYSIZE(tables->resources);
  tables->schedule_models[0] = {
      .stable_id =
          loom_low_descriptor_stable_id_from_key(IREE_SV("test.model")),
      .model = &tables->schedule_model,
  };
  tables->set.schedule_class_name_string_refs = tables->schedule_class_names;
  tables->set.schedule_models = tables->schedule_models;
  tables->set.schedule_model_count = IREE_ARRAYSIZE(tables->schedule_models);
  tables->set.schedule_class_count = IREE_ARRAYSIZE(tables->schedule_classes);
}

TEST(LowDescriptorRequirementsTest, VerifiesTargetLowFoundation) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify_requirements(
      &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
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
          &tables.set, descriptor_register_class->name_string_ref),
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
  tables.descriptors[0].mnemonic_string_ref = LOOM_STRING_REF_NONE;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest, RejectsMissingSemanticTag) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.descriptor_identities[0].semantic_tag_string_ref =
      LOOM_STRING_REF_NONE;

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
  tables.schedule_model.issue_use_count = 0;

  IREE_ASSERT_OK(loom_low_descriptor_set_verify_requirements(
      &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest,
     RejectsNonZeroLatencyScheduleWithoutIssueUse) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  tables.schedule_classes[0].issue_use_count = 0;
  tables.schedule_model.issue_use_count = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRequirementsTest, VerifiesEveryScheduleModel) {
  RequirementTables tables;
  InitializeRequirementTables(&tables);
  loom_low_schedule_model_t incomplete_model = tables.schedule_model;
  incomplete_model.resource_count = 0;
  const loom_low_schedule_model_entry_t schedule_models[] = {
      tables.schedule_models[0],
      {
          .stable_id = UINT64_MAX,
          .model = &incomplete_model,
      },
  };
  tables.set.schedule_models = schedule_models;
  tables.set.schedule_model_count = IREE_ARRAYSIZE(schedule_models);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_descriptor_set_verify_requirements(
          &tables.set, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

}  // namespace
}  // namespace loom
