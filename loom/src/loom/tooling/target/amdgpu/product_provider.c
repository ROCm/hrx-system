// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/product_provider.h"

#include "loom/product/kernel.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/emit/native/amdgpu/product_contract.h"
#include "loom/tooling/compile/artifact_product.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"

static const loom_product_artifact_schema_t
    kLoomAmdgpuHsacoProductArtifactSchemas[] = {
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
            .format = IREE_SVL(LOOM_AMDGPU_PRODUCT_FORMAT_HSACO),
            .minimum_count = 1,
            .maximum_count = 1,
        },
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_LISTING),
            .format = IREE_SVL(LOOM_AMDGPU_PRODUCT_ARTIFACT_FORMAT_ASSEMBLY),
            .minimum_count = 0,
            .maximum_count = 1,
        },
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_ARTIFACT_MANIFEST),
            .format = IREE_SVL(LOOM_PRODUCT_ARTIFACT_FORMAT_JSON),
            .minimum_count = 0,
            .maximum_count = 1,
        },
};

const loom_product_format_t loom_amdgpu_hsaco_product_format = {
    .operation = &loom_kernel_product_operation,
    .name = IREE_SVL(LOOM_AMDGPU_PRODUCT_FORMAT_HSACO),
    .persistence = LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE,
    .single_file =
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
            .extension = IREE_SVL(".hsaco"),
        },
    .artifact_schemas = kLoomAmdgpuHsacoProductArtifactSchemas,
    .artifact_schema_count =
        IREE_ARRAYSIZE(kLoomAmdgpuHsacoProductArtifactSchemas),
};

static iree_status_t loom_amdgpu_hsaco_product_provider_build(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  return loom_artifact_provider_build_kernel_product(
      provider, &loom_amdgpu_artifact_provider, request, out_product);
}

const loom_product_format_provider_t loom_amdgpu_hsaco_product_provider = {
    .name = IREE_SVL("amdgpu-hal"),
    .operation = &loom_kernel_product_operation,
    .format = &loom_amdgpu_hsaco_product_format,
    .target_profile_type = &loom_amdgpu_target_profile_type,
    .flags = LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    .target_product_contract = &loom_amdgpu_hsaco_kernel_product_contract,
    .build = loom_amdgpu_hsaco_product_provider_build,
};
