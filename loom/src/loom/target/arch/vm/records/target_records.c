// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/records/target_records.h"

#include <stdint.h>

static const loom_target_snapshot_t kVmCoreSnapshot = {
    .name = IREE_SVL("vm-core"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_VM,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .subgroup_size = 1,
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

static const loom_target_export_plan_t kVmCoreExportPlan = {
    .name = IREE_SVL("vm-function"),
    .abi_kind = LOOM_TARGET_ABI_VM_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kVmCoreConfig = {
    .name = IREE_SVL("vm.core"),
    .contract_set_key = IREE_SVL("vm.core"),
};

const loom_target_bundle_t loom_vm_target_bundle_core = {
    .name = IREE_SVL("vm-core"),
    .snapshot = &kVmCoreSnapshot,
    .export_plan = &kVmCoreExportPlan,
    .config = &kVmCoreConfig,
};

bool loom_vm_target_bundle_is_core(const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->config != NULL &&
         iree_string_view_equal(bundle->config->contract_set_key,
                                IREE_SV("vm.core"));
}

static const loom_target_bundle_t* const kVmTargetBundleValues[] = {
    NULL,
    &loom_vm_target_bundle_core,
};

const loom_target_bundle_table_t loom_vm_target_bundles = {
    .values = kVmTargetBundleValues,
    .count = IREE_ARRAYSIZE(kVmTargetBundleValues),
};
