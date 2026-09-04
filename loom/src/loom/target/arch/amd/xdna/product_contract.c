// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/product_contract.h"

const loom_target_product_contract_t loom_xdna_kernel_product_contract = {
    .name = IREE_SVL("xdna"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .abi_kind = LOOM_TARGET_ABI_ARRAY_PROGRAM,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};
