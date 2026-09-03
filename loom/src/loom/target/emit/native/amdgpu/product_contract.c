// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/product_contract.h"

const loom_target_product_contract_t loom_amdgpu_hsaco_kernel_product_contract =
    {
        .name = IREE_SVL("amdgpu-hsaco"),
        .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
        .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
        .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};
