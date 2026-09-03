// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/product/product.h"

void loom_product_initialize(const loom_product_descriptor_t* descriptor,
                             const loom_product_artifact_t* artifacts,
                             iree_host_size_t artifact_count,
                             iree_host_size_t export_count,
                             iree_host_size_t requirement_count,
                             loom_product_t* out_product) {
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(descriptor->destroy);
  IREE_ASSERT_ARGUMENT(out_product);
  IREE_ASSERT(artifact_count == 0 || artifacts != NULL);
  iree_atomic_ref_count_init(&out_product->ref_count);
  out_product->descriptor = descriptor;
  out_product->artifacts.values = artifacts;
  out_product->artifacts.count = artifact_count;
  out_product->export_count = export_count;
  out_product->requirement_count = requirement_count;
}

bool loom_product_isa(const loom_product_t* product,
                      const loom_product_descriptor_t* descriptor) {
  return product != NULL && product->descriptor == descriptor;
}

void loom_product_retain(loom_product_t* product) {
  if (product == NULL) return;
  iree_atomic_ref_count_inc(&product->ref_count);
}

void loom_product_release(loom_product_t* product) {
  if (product == NULL) return;
  if (iree_atomic_ref_count_dec(&product->ref_count) == 1) {
    product->descriptor->destroy(product);
  }
}

const loom_product_descriptor_t* loom_product_descriptor(
    const loom_product_t* product) {
  return product ? product->descriptor : NULL;
}

iree_host_size_t loom_product_artifact_count(const loom_product_t* product) {
  return product ? product->artifacts.count : 0;
}

const loom_product_artifact_t* loom_product_artifact_at(
    const loom_product_t* product, iree_host_size_t ordinal) {
  if (product == NULL || ordinal >= product->artifacts.count) return NULL;
  return &product->artifacts.values[ordinal];
}

iree_host_size_t loom_product_export_count(const loom_product_t* product) {
  return product ? product->export_count : 0;
}

iree_host_size_t loom_product_requirement_count(const loom_product_t* product) {
  return product ? product->requirement_count : 0;
}
