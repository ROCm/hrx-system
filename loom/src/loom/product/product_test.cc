// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/product/product.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

typedef struct test_product_t {
  // Generic immutable product interface under test.
  loom_product_t base;

  // Allocator used for test product storage.
  iree_allocator_t allocator;

  // Caller-owned destruction observation flag.
  bool* destroyed;

  // Owned artifact byte sequence.
  iree_byte_sequence_t* contents;

  // Product-owned artifact view.
  loom_product_artifact_t artifact;
} test_product_t;

static void DestroyTestProduct(loom_product_t* base_product) {
  test_product_t* product = (test_product_t*)base_product;
  *product->destroyed = true;
  iree_byte_sequence_release(product->contents);
  iree_allocator_free(product->allocator, product);
}

static const loom_product_descriptor_t kTestProductDescriptor = {
    /*.name=*/IREE_SV("test-product"),
    /*.destroy=*/DestroyTestProduct,
};

static test_product_t* CreateTestProduct(bool* destroyed) {
  iree_allocator_t allocator = iree_allocator_system();
  test_product_t* product = nullptr;
  IREE_CHECK_OK(
      iree_allocator_malloc(allocator, sizeof(*product), (void**)&product));
  product->allocator = allocator;
  product->destroyed = destroyed;
  static const uint8_t kContents[] = {1, 2, 3, 4};
  iree_byte_span_t contents = iree_byte_span_empty();
  contents.data_length = sizeof(kContents);
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      allocator, contents.data_length, (void**)&contents.data));
  memcpy(contents.data, kContents, sizeof(kContents));
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(&contents, allocator,
                                                         &product->contents));
  product->artifact = (loom_product_artifact_t){
      /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
      /*.format=*/IREE_SV("test-format"),
      /*.identifier=*/IREE_SV("test.bin"),
      /*.contents=*/product->contents,
  };
  loom_product_initialize(&kTestProductDescriptor, &product->artifact,
                          /*artifact_count=*/1, /*export_count=*/2,
                          /*requirement_count=*/3, &product->base);
  return product;
}

TEST(ProductTest, PreservesOpenArtifactIdentityAndOrdinalSpaces) {
  bool destroyed = false;
  test_product_t* product = CreateTestProduct(&destroyed);
  ASSERT_NE(product, nullptr);

  EXPECT_TRUE(loom_product_isa(&product->base, &kTestProductDescriptor));
  EXPECT_EQ(loom_product_descriptor(&product->base), &kTestProductDescriptor);
  EXPECT_EQ(loom_product_artifact_count(&product->base), 1u);
  EXPECT_EQ(loom_product_export_count(&product->base), 2u);
  EXPECT_EQ(loom_product_requirement_count(&product->base), 3u);
  const loom_product_artifact_t* artifact =
      loom_product_artifact_at(&product->base, 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      artifact->role, IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL)));
  EXPECT_TRUE(iree_string_view_equal(artifact->format, IREE_SV("test-format")));
  EXPECT_EQ(iree_byte_sequence_length(artifact->contents), 4u);
  EXPECT_EQ(loom_product_artifact_at(&product->base, 1), nullptr);

  loom_product_retain(&product->base);
  loom_product_release(&product->base);
  EXPECT_FALSE(destroyed);
  loom_product_release(&product->base);
  EXPECT_TRUE(destroyed);
}

}  // namespace
}  // namespace loom
