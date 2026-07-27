// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/records/target_records.h"

#include <stdint.h>

static const loom_target_snapshot_t kWasmCoreSimd128Snapshot = {
    .name = IREE_SVL("wasm32-simd128"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_WASM,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    .default_pointer_bitwidth = 32,
    .index_bitwidth = 32,
    .offset_bitwidth = 32,
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

static const loom_target_export_plan_t kWasmCoreSimd128ExportPlan = {
    .name = IREE_SVL("wasm-function"),
    .abi_kind = LOOM_TARGET_ABI_WASM_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kWasmCoreSimd128Config = {
    .name = IREE_SVL("wasm.core.simd128"),
    .contract_set_key = IREE_SVL("wasm.core.simd128"),
};

const loom_target_bundle_t loom_wasm_low_target_bundle_core_simd128 = {
    .name = IREE_SVL("wasm-simd128"),
    .snapshot = &kWasmCoreSimd128Snapshot,
    .export_plan = &kWasmCoreSimd128ExportPlan,
    .config = &kWasmCoreSimd128Config,
};

static const loom_target_bundle_t* const kWasmTargetBundleValues[] = {
    NULL,
    &loom_wasm_low_target_bundle_core_simd128,
};

const loom_target_bundle_table_t loom_wasm_target_bundles = {
    .values = kWasmTargetBundleValues,
    .count = IREE_ARRAYSIZE(kWasmTargetBundleValues),
};
