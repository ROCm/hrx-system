// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/product.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

typedef struct TestProduct {
  loom_product_t base;
  loom_product_artifact_t artifact;
} TestProduct;

static void DestroyTestProduct(loom_product_t* base_product) {
  TestProduct* product = reinterpret_cast<TestProduct*>(base_product);
  iree_byte_sequence_release(product->artifact.contents);
  delete product;
}

static const loom_product_descriptor_t kProductDescriptor = {
    /*.name=*/IREE_SV("test"),
    /*.destroy=*/DestroyTestProduct,
};

static const loom_product_root_match_t kRootMatches[] = {
    {IREE_SV("test.root"), LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED},
};

static const loom_product_operation_t kOperation = {
    /*.name=*/IREE_SV("test"),
    /*.product_descriptor=*/&kProductDescriptor,
    /*.root_matches=*/kRootMatches,
    /*.root_match_count=*/IREE_ARRAYSIZE(kRootMatches),
};

static const loom_product_artifact_schema_t kArtifactSchemas[] = {
    {
        /*.role=*/IREE_SV("payload"),
        /*.format=*/IREE_SV("test-binary"),
        /*.minimum_count=*/1,
        /*.maximum_count=*/1,
    },
};

static const loom_product_format_t kFormat = {
    /*.operation=*/&kOperation,
    /*.name=*/IREE_SV("test-binary"),
    /*.persistence=*/LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE,
    /*.single_file=*/
    {
        /*.role=*/IREE_SV("payload"),
        /*.extension=*/IREE_SV(".test"),
    },
    /*.artifact_schemas=*/kArtifactSchemas,
    /*.artifact_schema_count=*/IREE_ARRAYSIZE(kArtifactSchemas),
};

static iree_status_t BuildTestProduct(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  (void)provider;
  TestProduct* product = new TestProduct{};
  iree_byte_span_t data = iree_byte_span_empty();
  iree_status_t status = iree_allocator_malloc_uninitialized(
      request->allocator, 1, reinterpret_cast<void**>(&data.data));
  if (iree_status_is_ok(status)) {
    data.data_length = 1;
    data.data[0] = 42;
    status = iree_byte_sequence_create_from_span_move(
        &data, request->allocator, &product->artifact.contents);
  }
  if (iree_status_is_ok(status)) {
    product->artifact.role = IREE_SV("payload");
    product->artifact.format = IREE_SV("test-binary");
    product->artifact.identifier = IREE_SV("test.bin");
    loom_product_initialize(&kProductDescriptor, &product->artifact, 1, 1, 0,
                            &product->base);
    *out_product = &product->base;
  } else {
    iree_allocator_free(request->allocator, data.data);
    delete product;
  }
  return status;
}

static const loom_product_format_provider_t kProvider = {
    /*.name=*/IREE_SV("test-provider"),
    /*.operation=*/&kOperation,
    /*.format=*/&kFormat,
    /*.target_profile_type=*/nullptr,
    /*.accepts_target=*/nullptr,
    /*.flags=*/LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    /*.target_product_contract=*/nullptr,
    /*.default_pipeline_options=*/nullptr,
    /*.build=*/BuildTestProduct,
};

TEST(ProductBuildTest, ValidatesProviderProduct) {
  loom_module_t module = {};
  loom_compile_options_t compile_options = {};
  iree_arena_block_pool_t block_pool = {};
  const loom_product_build_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/nullptr,
      /*.module=*/&module,
      /*.target_profile=*/nullptr,
      /*.target_key=*/{},
      /*.artifact_identifier=*/IREE_SV("test.bin"),
      /*.export_count=*/1,
      /*.compile_options=*/&compile_options,
      /*.block_pool=*/&block_pool,
      /*.option_chain=*/nullptr,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_product_t* product = nullptr;
  IREE_ASSERT_OK(
      loom_product_format_provider_build(&kProvider, &request, &product));
  ASSERT_NE(product, nullptr);
  EXPECT_TRUE(loom_product_isa(product, &kProductDescriptor));
  loom_product_release(product);
}

}  // namespace
}  // namespace loom
