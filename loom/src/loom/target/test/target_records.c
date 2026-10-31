// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/target_records.h"

#include <stdint.h>

#include "loom/target/facts_builder.h"

#define LOOM_TEST_LOW_CORE_SCHEDULE_MODEL_STABLE_ID UINT64_C(0x235c515ddcc8fd92)
#define LOOM_TEST_LOW_CORE_CONTRACT_SET_STABLE_ID UINT64_C(0x235c515ddcc8fd92)

static const loom_target_snapshot_t kTestLowSnapshot = {
    .name = IREE_SVL("test-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .maximum_index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = 0,
            .host = UINT32_MAX,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_snapshot_t kTestQuirkySnapshot = {
    .name = IREE_SVL("test-quirky"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .maximum_index_bitwidth = 32,
    .offset_bitwidth = 64,
    .subgroup_size = 7,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = 0,
            .host = UINT32_MAX,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kTestLowExportPlan = {
    .name = IREE_SVL("test-low-function"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kTestLowConfig = {
    .name = IREE_SVL("test.low.core"),
    .schedule_model_stable_id = LOOM_TEST_LOW_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_TEST_LOW_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("test.low.core"),
        },
};

static const loom_target_bundle_t kTestLowTargetBundleCore = {
    .name = IREE_SVL("test-low"),
    .snapshot = &kTestLowSnapshot,
    .export_plan = &kTestLowExportPlan,
    .config = &kTestLowConfig,
};

static const loom_target_bundle_t kTestLowTargetBundleQuirky = {
    .name = IREE_SVL("test-quirky"),
    .snapshot = &kTestQuirkySnapshot,
    .export_plan = &kTestLowExportPlan,
    .config = &kTestLowConfig,
};

static const loom_target_export_plan_t kTestDeviceExportPlan = {
    .name = IREE_SVL("test-device-program"),
    .abi_kind = LOOM_TARGET_ABI_ARRAY_PROGRAM,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kTestDeviceConfig = {
    .name = IREE_SVL("test.device"),
    .schedule_model_stable_id = LOOM_TEST_LOW_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = UINT64_C(0x21009ae69c3322bf),
            .spelling = IREE_SVL("test.device"),
        },
};

static const loom_target_bundle_t kTestDeviceTargetBundle = {
    .name = IREE_SVL("test-device"),
    .snapshot = &kTestLowSnapshot,
    .export_plan = &kTestDeviceExportPlan,
    .config = &kTestDeviceConfig,
};

static iree_status_t loom_test_target_facts_project_worker(
    const loom_target_facts_t* facts, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = facts;
  if (facts->selector != LOOM_TEST_TARGET_KIND_DEVICE) {
    return iree_ok_status();
  }
  loom_target_facts_t* worker = NULL;
  IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(facts, arena, &worker));
  loom_target_facts_builder_set_worker_contract(
      LOOM_TEST_TARGET_KIND_LOW_CORE, &kTestLowTargetBundleCore, worker);
  *out_facts = worker;
  return iree_ok_status();
}

const loom_target_fact_type_t loom_test_target_fact_type = {
    .name = IREE_SVL("test"),
    .storage_size = sizeof(loom_target_facts_t),
    .satisfies_identity_requirement =
        loom_target_facts_selector_satisfies_identity_requirement,
    .satisfies_specialization_requirement =
        loom_target_facts_structural_satisfy_specialization_requirement,
    .project_worker = loom_test_target_facts_project_worker,
};

static const loom_target_bundle_t* const kTestTargetBundleValues[] = {
    NULL,
    &kTestLowTargetBundleCore,
    &kTestLowTargetBundleQuirky,
    &kTestDeviceTargetBundle,
};

const loom_target_bundle_table_t loom_test_target_bundles = {
    .values = kTestTargetBundleValues,
    .count = IREE_ARRAYSIZE(kTestTargetBundleValues),
};
