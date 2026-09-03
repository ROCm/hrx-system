// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/product.h"

static iree_status_t loom_product_build_request_validate(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request) {
  if (provider == NULL || provider->operation == NULL ||
      provider->format == NULL || provider->build == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "product format provider is incomplete");
  }
  if (request == NULL || request->module == NULL ||
      request->compile_options == NULL || request->block_pool == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "product build request is missing required compiler state");
  }
  if (iree_allocator_is_null(request->allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "product build allocator is null");
  }
  if (provider->target_profile_type == NULL) {
    if (request->target_profile != NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target-neutral product provider '%.*s' received a target profile",
          (int)provider->name.size, provider->name.data);
    }
  } else if (request->target_profile != NULL &&
             request->target_profile->type != provider->target_profile_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "product provider '%.*s' cannot consume target family '%.*s'",
        (int)provider->name.size, provider->name.data,
        (int)request->target_profile->type->name.size,
        request->target_profile->type->name.data);
  }
  return iree_ok_status();
}

iree_status_t loom_product_format_provider_build(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  IREE_ASSERT_ARGUMENT(out_product);
  *out_product = NULL;
  IREE_RETURN_IF_ERROR(loom_product_build_request_validate(provider, request));

  loom_product_t* product = NULL;
  iree_status_t status = provider->build(provider, request, &product);
  if (iree_status_is_ok(status) && product != NULL) {
    status = loom_product_format_validate_product(provider->format, product);
  }
  if (iree_status_is_ok(status)) {
    *out_product = product;
    product = NULL;
  }
  loom_product_release(product);
  return status;
}
