// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/product_contract.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/facts_builder.h"

namespace loom {
namespace {

static const loom_target_fact_type_t kFactType = {
    /*.name=*/IREE_SVL("product-contract-test"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};

static const loom_target_snapshot_t kNeutralSnapshot = {
    /*.name=*/IREE_SVL("x86-64-v3"),
    /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN,
    /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
    /*.default_pointer_bitwidth=*/64,
    /*.index_bitwidth=*/64,
    /*.offset_bitwidth=*/64,
};

static const loom_target_export_plan_t kNeutralExportPlan = {
    /*.name=*/IREE_SVL("x86-64-v3"),
    /*.export_symbol=*/{},
    /*.abi_kind=*/LOOM_TARGET_ABI_UNKNOWN,
    /*.linkage=*/LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kNeutralConfig = {
    /*.name=*/IREE_SVL("x86-64-v3"),
};

static const loom_target_bundle_t kNeutralBundle = {
    /*.name=*/IREE_SVL("x86-64-v3"),
    /*.snapshot=*/&kNeutralSnapshot,
    /*.export_plan=*/&kNeutralExportPlan,
    /*.config=*/&kNeutralConfig,
};

static loom_target_facts_t MakeNeutralFacts() {
  loom_target_facts_t facts = {};
  loom_target_facts_builder_initialize(&kFactType, &kNeutralBundle, &facts);
  return facts;
}

TEST(TargetProductContractTest, AppliesDistinctFormatsToOneProfile) {
  static const loom_target_product_contract_t kElfContract = {
      /*.name=*/IREE_SVL("elf-object"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.abi_kind=*/LOOM_TARGET_ABI_OBJECT_FUNCTION,
      /*.linkage=*/LOOM_TARGET_LINKAGE_DSO_LOCAL,
  };
  static const loom_target_product_contract_t kCoffContract = {
      /*.name=*/IREE_SVL("coff-object"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_COFF,
      /*.abi_kind=*/LOOM_TARGET_ABI_OBJECT_FUNCTION,
      /*.linkage=*/LOOM_TARGET_LINKAGE_DEFAULT,
  };

  loom_target_facts_t elf_facts = MakeNeutralFacts();
  IREE_ASSERT_OK(loom_target_product_contract_apply(&kElfContract, &elf_facts));
  EXPECT_EQ(elf_facts.storage.snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
  EXPECT_EQ(elf_facts.storage.snapshot.artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(elf_facts.storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(elf_facts.storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_TRUE(iree_string_view_equal(elf_facts.storage.export_plan.name,
                                     IREE_SV("elf-object")));

  loom_target_facts_t coff_facts = MakeNeutralFacts();
  IREE_ASSERT_OK(
      loom_target_product_contract_apply(&kCoffContract, &coff_facts));
  EXPECT_EQ(coff_facts.storage.snapshot.artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_COFF);
  EXPECT_EQ(coff_facts.storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DEFAULT);

  for (const loom_target_facts_t* facts : {&elf_facts, &coff_facts}) {
    EXPECT_TRUE(loom_target_facts_field_is_explicit(
        facts, LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT));
    EXPECT_TRUE(loom_target_facts_field_is_explicit(
        facts, LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT));
    EXPECT_TRUE(
        loom_target_facts_field_is_explicit(facts, LOOM_TARGET_FACT_FIELD_ABI));
    EXPECT_TRUE(loom_target_facts_field_is_explicit(
        facts, LOOM_TARGET_FACT_FIELD_LINKAGE));
    EXPECT_EQ(facts->storage.snapshot.default_pointer_bitwidth, 64u);
  }
}

TEST(TargetProductContractTest, RejectsIncompleteContracts) {
  const loom_target_product_contract_t incomplete_contract = {
      /*.name=*/IREE_SVL("incomplete"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.abi_kind=*/LOOM_TARGET_ABI_UNKNOWN,
  };
  loom_target_facts_t facts = MakeNeutralFacts();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_product_contract_apply(&incomplete_contract, &facts));
}

TEST(TargetProductContractTest, RejectsAlreadyBoundProfileFacts) {
  const loom_target_product_contract_t contract = {
      /*.name=*/IREE_SVL("elf-object"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.abi_kind=*/LOOM_TARGET_ABI_OBJECT_FUNCTION,
      /*.linkage=*/LOOM_TARGET_LINKAGE_DSO_LOCAL,
  };
  loom_target_facts_t facts = MakeNeutralFacts();
  loom_target_fact_field_set_insert(&facts.explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_target_product_contract_apply(&contract, &facts));
}

}  // namespace
}  // namespace loom
