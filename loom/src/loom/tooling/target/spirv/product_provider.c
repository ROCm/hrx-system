// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/product_provider.h"

#include "loom/product/kernel.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/emit/spirv/product_contract.h"
#include "loom/tooling/compile/artifact_product.h"
#include "loom/tooling/target/spirv/artifact_provider.h"

static const loom_product_artifact_schema_t
    kLoomSpirvBinaryProductArtifactSchemas[] = {
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
            .format = IREE_SVL(LOOM_SPIRV_PRODUCT_FORMAT_BINARY),
            .minimum_count = 1,
            .maximum_count = 1,
        },
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_ARTIFACT_MANIFEST),
            .format = IREE_SVL(LOOM_PRODUCT_ARTIFACT_FORMAT_JSON),
            .minimum_count = 0,
            .maximum_count = 1,
        },
};

const loom_product_format_t loom_spirv_binary_product_format = {
    .operation = &loom_kernel_product_operation,
    .name = IREE_SVL(LOOM_SPIRV_PRODUCT_FORMAT_BINARY),
    .persistence = LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE,
    .single_file =
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
            .extension = IREE_SVL(".spv"),
        },
    .artifact_schemas = kLoomSpirvBinaryProductArtifactSchemas,
    .artifact_schema_count =
        IREE_ARRAYSIZE(kLoomSpirvBinaryProductArtifactSchemas),
};

static iree_status_t loom_spirv_binary_product_provider_build(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  return loom_artifact_provider_build_kernel_product(
      provider, &loom_spirv_vulkan_artifact_provider, request, out_product);
}

const loom_product_format_provider_t loom_spirv_binary_product_provider = {
    .name = IREE_SVL("spirv-vulkan-hal"),
    .operation = &loom_kernel_product_operation,
    .format = &loom_spirv_binary_product_format,
    .target_profile_type = &loom_spirv_target_profile_type,
    .flags = LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    .target_product_contract = &loom_spirv_binary_kernel_product_contract,
    .default_pipeline_options =
        &loom_spirv_vulkan_artifact_provider.default_pipeline_options,
    .build = loom_spirv_binary_product_provider_build,
};
