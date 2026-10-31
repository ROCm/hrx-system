// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/records.h"

#define LOOM_VM_CORE_SCHEDULE_MODEL_STABLE_ID UINT64_C(0x32476f14332ccb19)
#define LOOM_VM_CORE_CONTRACT_SET_STABLE_ID UINT64_C(0x32476f14332ccb19)

static const loom_target_snapshot_t kCoreSnapshot = {
    .name = IREE_SVL("vm.core"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_VM,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BINARY,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .maximum_index_bitwidth = 64,
    .offset_bitwidth = 64,
};

static const loom_target_export_plan_t kCoreExportPlan = {
    .name = IREE_SVL("vm-function"),
    .abi_kind = LOOM_TARGET_ABI_VM_FUNCTION,
};

static const loom_target_config_t kCoreConfig = {
    .name = IREE_SVL("vm.core"),
    .schedule_model_stable_id = LOOM_VM_CORE_SCHEDULE_MODEL_STABLE_ID,
    .contract_set =
        {
            .stable_id = LOOM_VM_CORE_CONTRACT_SET_STABLE_ID,
            .spelling = IREE_SVL("vm.core"),
        },
};

const loom_target_bundle_t loom_vm_core_target_bundle = {
    .name = IREE_SVL("vm-core"),
    .snapshot = &kCoreSnapshot,
    .export_plan = &kCoreExportPlan,
    .config = &kCoreConfig,
};

static const loom_target_bundle_t* const kBundles[] = {
    NULL, &loom_vm_core_target_bundle};

const loom_target_bundle_table_t loom_vm_target_bundles = {
    .values = kBundles,
    .count = IREE_ARRAYSIZE(kBundles),
};
