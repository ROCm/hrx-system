// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/target_records.h"

#include <stdint.h>

static const loom_target_snapshot_t kTestLowSnapshot = {
    .name = IREE_SVL("test-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
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
    .contract_set_key = IREE_SVL("test.low.core"),
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

static const loom_target_bundle_t* const kTestTargetBundleValues[] = {
    NULL,
    &kTestLowTargetBundleCore,
    &kTestLowTargetBundleQuirky,
};

const loom_target_bundle_table_t loom_test_target_bundles = {
    .values = kTestTargetBundleValues,
    .count = IREE_ARRAYSIZE(kTestTargetBundleValues),
};
