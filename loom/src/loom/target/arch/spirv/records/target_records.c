// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/records/target_records.h"

#define LOOM_SPIRV_LOGICAL_CORE_SCHEDULE_MODEL_STABLE_ID \
  UINT64_C(0x5b86e123b2b92ba9)
#define LOOM_SPIRV_LOGICAL_CORE_CONTRACT_SET_STABLE_ID \
  UINT64_C(0x5b86e123b2b92ba9)

#include <stdint.h>

#include "loom/target/arch/spirv/features.h"

static const loom_target_snapshot_t kSpirvVulkan13Snapshot = {
    .name = IREE_SVL("spirv-vulkan1.3"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
    .subgroup_partition = LOOM_TARGET_SUBGROUP_PARTITION_WORKGROUP_X_ALIGNED,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .maximum_index_bitwidth = 32,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = LOOM_SPIRV_STORAGE_CLASS_GENERIC,
            .global = LOOM_SPIRV_STORAGE_CLASS_CROSS_WORKGROUP,
            .workgroup = LOOM_SPIRV_STORAGE_CLASS_WORKGROUP,
            .constant = LOOM_SPIRV_STORAGE_CLASS_UNIFORM,
            .private_memory = LOOM_SPIRV_STORAGE_CLASS_FUNCTION,
            .host = UINT32_MAX,
            .descriptor = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
        },
};

static const loom_target_snapshot_t kSpirvVulkan13Subgroup32Snapshot = {
    .name = IREE_SVL("spirv-vulkan1.3-subgroup32"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
    .subgroup_partition = LOOM_TARGET_SUBGROUP_PARTITION_WORKGROUP_X_ALIGNED,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .maximum_index_bitwidth = 32,
    .offset_bitwidth = 64,
    .subgroup_size = 32,
    .memory_spaces =
        {
            .generic = LOOM_SPIRV_STORAGE_CLASS_GENERIC,
            .global = LOOM_SPIRV_STORAGE_CLASS_CROSS_WORKGROUP,
            .workgroup = LOOM_SPIRV_STORAGE_CLASS_WORKGROUP,
            .constant = LOOM_SPIRV_STORAGE_CLASS_UNIFORM,
            .private_memory = LOOM_SPIRV_STORAGE_CLASS_FUNCTION,
            .host = UINT32_MAX,
            .descriptor = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
        },
};

static const loom_target_export_plan_t kSpirvVulkan13ExportPlan = {
    .name = IREE_SVL("spirv-shader-entry-point"),
    .abi_kind = LOOM_TARGET_ABI_SHADER_ENTRY_POINT,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kSpirvVulkan13Config = {
    .name = IREE_SVL("spirv.logical.core"),
    .schedule_model_stable_id =
        LOOM_SPIRV_LOGICAL_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_SPIRV_LOGICAL_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("spirv.logical.core"),
        },
    .contract_feature_bits = LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
};

const loom_target_bundle_t loom_spirv_low_target_bundle_vulkan1_3 = {
    .name = IREE_SVL("spirv-vulkan1.3"),
    .snapshot = &kSpirvVulkan13Snapshot,
    .export_plan = &kSpirvVulkan13ExportPlan,
    .config = &kSpirvVulkan13Config,
};

static const loom_target_config_t kSpirvExtendedTypesConfig = {
    .name = IREE_SVL("spirv.logical.core"),
    .schedule_model_stable_id =
        LOOM_SPIRV_LOGICAL_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_SPIRV_LOGICAL_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("spirv.logical.core"),
        },
    .contract_feature_bits =
        LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA | LOOM_SPIRV_FEATURE_FLOAT16 |
        LOOM_SPIRV_FEATURE_FLOAT64 | LOOM_SPIRV_FEATURE_INT8 |
        LOOM_SPIRV_FEATURE_INT16 |
        LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS |
        LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS |
        LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |
        LOOM_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE,
};

const loom_target_bundle_t loom_spirv_low_target_bundle_extended_types = {
    .name = IREE_SVL("spirv-vulkan1.3-extended-types"),
    .snapshot = &kSpirvVulkan13Snapshot,
    .export_plan = &kSpirvVulkan13ExportPlan,
    .config = &kSpirvExtendedTypesConfig,
};

static const loom_target_export_plan_t kSpirvHalKernelExportPlan = {
    .name = IREE_SVL("spirv-hal-kernel"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

const loom_target_bundle_t loom_spirv_low_target_bundle_hal_kernel = {
    .name = IREE_SVL("spirv-vulkan1.3-hal-kernel"),
    .snapshot = &kSpirvVulkan13Snapshot,
    .export_plan = &kSpirvHalKernelExportPlan,
    .config = &kSpirvVulkan13Config,
};

static const loom_target_config_t kSpirvSubgroup32BallotConfig = {
    .name = IREE_SVL("spirv.logical.core.subgroup32.ballot"),
    .schedule_model_stable_id =
        LOOM_SPIRV_LOGICAL_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_SPIRV_LOGICAL_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("spirv.logical.core"),
        },
    .contract_feature_bits = LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA |
                             LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT,
};

const loom_target_bundle_t loom_spirv_low_target_bundle_subgroup32_ballot_hal =
    {
        .name = IREE_SVL("spirv-vulkan1.3-subgroup32-ballot-hal"),
        .snapshot = &kSpirvVulkan13Subgroup32Snapshot,
        .export_plan = &kSpirvHalKernelExportPlan,
        .config = &kSpirvSubgroup32BallotConfig,
};

static const loom_target_config_t kSpirvSubgroup32PredicateConfig = {
    .name = IREE_SVL("spirv.logical.core.subgroup32.predicates"),
    .schedule_model_stable_id =
        LOOM_SPIRV_LOGICAL_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_SPIRV_LOGICAL_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("spirv.logical.core"),
        },
    .contract_feature_bits = LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA |
                             LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT |
                             LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE,
};

const loom_target_bundle_t
    loom_spirv_low_target_bundle_subgroup32_predicate_hal = {
        .name = IREE_SVL("spirv-vulkan1.3-subgroup32-predicate-hal"),
        .snapshot = &kSpirvVulkan13Subgroup32Snapshot,
        .export_plan = &kSpirvHalKernelExportPlan,
        .config = &kSpirvSubgroup32PredicateConfig,
};

static const loom_target_config_t kSpirvSubgroup32ReductionConfig = {
    .name = IREE_SVL("spirv.logical.core.subgroup32.reductions"),
    .schedule_model_stable_id =
        LOOM_SPIRV_LOGICAL_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_SPIRV_LOGICAL_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("spirv.logical.core"),
        },
    .contract_feature_bits =
        LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA |
        LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC |
        LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED |
        LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT |
        LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE |
        LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE |
        LOOM_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES |
        LOOM_SPIRV_FEATURE_FLOAT16 | LOOM_SPIRV_FEATURE_FLOAT64 |
        LOOM_SPIRV_FEATURE_INT8 | LOOM_SPIRV_FEATURE_INT16 |
        LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS |
        LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
};

const loom_target_bundle_t
    loom_spirv_low_target_bundle_subgroup32_reduction_hal = {
        .name = IREE_SVL("spirv-vulkan1.3-subgroup32-reduction-hal"),
        .snapshot = &kSpirvVulkan13Subgroup32Snapshot,
        .export_plan = &kSpirvHalKernelExportPlan,
        .config = &kSpirvSubgroup32ReductionConfig,
};

static const loom_target_bundle_t* const kSpirvTargetBundleValues[] = {
    NULL,
    &loom_spirv_low_target_bundle_vulkan1_3,
};

const loom_target_bundle_table_t loom_spirv_target_bundles = {
    .values = kSpirvTargetBundleValues,
    .count = IREE_ARRAYSIZE(kSpirvTargetBundleValues),
};
