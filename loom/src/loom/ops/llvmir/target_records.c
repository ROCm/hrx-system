// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/llvmir/target_records.h"

#include <stdint.h>

static const loom_target_snapshot_t kLlvmirGenericObjectSnapshot = {
    .name = IREE_SVL("llvmir-generic-object"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
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
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kLlvmirGenericObjectExportPlan = {
    .name = IREE_SVL("llvmir-object-function"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kLlvmirGenericObjectConfig = {
    .name = IREE_SVL("llvmir.generic.core"),
    .contract_set_key = IREE_SVL("llvmir.generic.core"),
};

static const loom_target_bundle_t kLlvmirGenericObjectBundle = {
    .name = IREE_SVL("llvmir-generic-object"),
    .snapshot = &kLlvmirGenericObjectSnapshot,
    .export_plan = &kLlvmirGenericObjectExportPlan,
    .config = &kLlvmirGenericObjectConfig,
};

static const loom_target_bundle_t* const kLlvmirTargetBundleValues[] = {
    NULL,
    &kLlvmirGenericObjectBundle,
};

const loom_target_bundle_table_t loom_llvmir_target_bundles = {
    .values = kLlvmirTargetBundleValues,
    .count = IREE_ARRAYSIZE(kLlvmirTargetBundleValues),
};
