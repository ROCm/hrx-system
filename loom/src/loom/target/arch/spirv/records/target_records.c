// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/records/target_records.h"

#include <stdint.h>

#include "loom/target/arch/spirv/features.h"

static const loom_target_snapshot_t kSpirvVulkan13Snapshot = {
    .name = IREE_SVL("spirv-vulkan1.3"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
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

static const loom_target_export_plan_t kSpirvVulkan13ExportPlan = {
    .name = IREE_SVL("spirv-shader-entry-point"),
    .abi_kind = LOOM_TARGET_ABI_SHADER_ENTRY_POINT,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kSpirvVulkan13Config = {
    .name = IREE_SVL("spirv.logical.core"),
    .contract_set_key = IREE_SVL("spirv.logical.core"),
    .contract_feature_bits = LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
};

const loom_target_bundle_t loom_spirv_low_target_bundle_vulkan1_3 = {
    .name = IREE_SVL("spirv-vulkan1.3"),
    .snapshot = &kSpirvVulkan13Snapshot,
    .export_plan = &kSpirvVulkan13ExportPlan,
    .config = &kSpirvVulkan13Config,
};

static const loom_target_bundle_t* const kSpirvTargetBundleValues[] = {
    NULL,
    &loom_spirv_low_target_bundle_vulkan1_3,
};

const loom_target_bundle_table_t loom_spirv_target_bundles = {
    .values = kSpirvTargetBundleValues,
    .count = IREE_ARRAYSIZE(kSpirvTargetBundleValues),
};
