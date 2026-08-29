// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/records/target_records.h"

#include <stdint.h>

static const loom_target_snapshot_t kAie2pCoreSnapshot = {
    .name = IREE_SVL("amd-xdna-aie2p-core"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 32,
    .index_bitwidth = 32,
    .offset_bitwidth = 32,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = 0,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kAie2pCoreExportPlan = {
    .name = IREE_SVL("aie2p-object-function"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kAie2pCoreConfig = {
    .name = IREE_SVL("amd.xdna.aie2p.core"),
    .contract_set_key = IREE_SVL("amd.xdna.aie2p.core"),
    .contract_feature_bits = 0,
};

static const loom_target_bundle_t kAie2pCoreBundle = {
    .name = IREE_SVL("aie2p-core"),
    .snapshot = &kAie2pCoreSnapshot,
    .export_plan = &kAie2pCoreExportPlan,
    .config = &kAie2pCoreConfig,
};

static const loom_target_bundle_t* const kAie2pTargetBundleValues[] = {
    NULL,
    &kAie2pCoreBundle,
};

const loom_target_bundle_table_t loom_aie2p_target_bundles = {
    .values = kAie2pTargetBundleValues,
    .count = IREE_ARRAYSIZE(kAie2pTargetBundleValues),
};
