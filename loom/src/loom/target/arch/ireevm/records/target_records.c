// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/records/target_records.h"

#include <stdint.h>

#include "loom/target/arch/ireevm/records/feature_bits.h"

static const loom_target_snapshot_t kIreeVmSnapshot = {
    .name = IREE_SVL("iree-vm"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_VM,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE,
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

static const loom_target_export_plan_t kIreeVmExportPlan = {
    .name = IREE_SVL("iree-vm-function"),
    .abi_kind = LOOM_TARGET_ABI_VM_MODULE_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kIreeVmConfig = {
    .name = IREE_SVL("ireevm.core"),
    .contract_set_key = IREE_SVL("ireevm.core"),
    .contract_feature_bits =
        LOOM_IREEVM_FEATURE_EXT_F32 | LOOM_IREEVM_FEATURE_EXT_F64,
};

const loom_target_bundle_t loom_ireevm_low_target_bundle_core = {
    .name = IREE_SVL("iree-vm"),
    .snapshot = &kIreeVmSnapshot,
    .export_plan = &kIreeVmExportPlan,
    .config = &kIreeVmConfig,
};

static const loom_target_bundle_t* const kIreeVmTargetBundleValues[] = {
    NULL,
    &loom_ireevm_low_target_bundle_core,
};

const loom_target_bundle_table_t loom_ireevm_target_bundles = {
    .values = kIreeVmTargetBundleValues,
    .count = IREE_ARRAYSIZE(kIreeVmTargetBundleValues),
};
