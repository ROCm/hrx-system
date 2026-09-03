// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/product/kernel.h"

#include <string.h>

typedef struct loom_kernel_product_t {
  // Generic immutable product interface.
  loom_product_t base;

  // Host allocator used for product-owned storage.
  iree_allocator_t allocator;

  // Canonical selected target key retained in |string_storage|.
  iree_string_view_t target_key;

  // Complete target bundle with internal views rebound to this product.
  loom_target_bundle_storage_t target_bundle_storage;

  // Artifact ordinal containing the loadable kernel payload.
  iree_host_size_t loadable_artifact_ordinal;

  // Owned artifact metadata table.
  loom_product_artifact_t* artifacts;

  // Contiguous storage owning all copied string views.
  char* string_storage;
} loom_kernel_product_t;

static const iree_string_view_t kLoomKernelProductRootOperationNames[] = {
    IREE_SVL("kernel.def"),
    IREE_SVL("low.kernel.def"),
};

static void loom_kernel_product_destroy(loom_product_t* base_product) {
  loom_kernel_product_t* product = (loom_kernel_product_t*)base_product;
  const iree_allocator_t allocator = product->allocator;
  for (iree_host_size_t i = 0; i < product->base.artifacts.count; ++i) {
    iree_byte_sequence_release(product->artifacts[i].contents);
  }
  iree_allocator_free(allocator, product->string_storage);
  iree_allocator_free(allocator, product->artifacts);
  iree_allocator_free(allocator, product);
}

const loom_product_descriptor_t loom_kernel_product_descriptor = {
    .name = IREE_SVL("kernel"),
    .destroy = loom_kernel_product_destroy,
};

const loom_product_operation_t loom_kernel_product_operation = {
    .name = IREE_SVL("kernel"),
    .product_descriptor = &loom_kernel_product_descriptor,
    .root_operation_names = kLoomKernelProductRootOperationNames,
    .root_operation_name_count =
        IREE_ARRAYSIZE(kLoomKernelProductRootOperationNames),
};

static const loom_kernel_product_t* loom_kernel_product_cast(
    const loom_product_t* base_product) {
  return loom_product_isa(base_product, &loom_kernel_product_descriptor)
             ? (const loom_kernel_product_t*)base_product
             : NULL;
}

static bool loom_kernel_product_accumulate_string_size(
    iree_string_view_t value, iree_host_size_t* inout_size) {
  return iree_host_size_checked_add(*inout_size, value.size, inout_size);
}

static bool loom_kernel_product_calculate_string_storage_size(
    const loom_kernel_product_options_t* options,
    iree_host_size_t* out_string_storage_size) {
  const loom_target_bundle_t* bundle = options->target_bundle;
  iree_host_size_t size = 0;
  if (!loom_kernel_product_accumulate_string_size(options->target_key, &size) ||
      !loom_kernel_product_accumulate_string_size(bundle->name, &size) ||
      !loom_kernel_product_accumulate_string_size(bundle->snapshot->name,
                                                  &size) ||
      !loom_kernel_product_accumulate_string_size(bundle->export_plan->name,
                                                  &size) ||
      !loom_kernel_product_accumulate_string_size(
          bundle->export_plan->export_symbol, &size) ||
      !loom_kernel_product_accumulate_string_size(bundle->config->name,
                                                  &size) ||
      !loom_kernel_product_accumulate_string_size(
          bundle->config->contract_set_key, &size)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->artifact_count; ++i) {
    if (!loom_kernel_product_accumulate_string_size(options->artifacts[i].role,
                                                    &size) ||
        !loom_kernel_product_accumulate_string_size(
            options->artifacts[i].format, &size) ||
        !loom_kernel_product_accumulate_string_size(
            options->artifacts[i].identifier, &size)) {
      return false;
    }
  }
  *out_string_storage_size = size;
  return true;
}

static iree_string_view_t loom_kernel_product_copy_string(
    iree_string_view_t source, char** inout_cursor) {
  if (iree_string_view_is_empty(source)) return source;
  char* target = *inout_cursor;
  memcpy(target, source.data, source.size);
  *inout_cursor += source.size;
  return iree_make_string_view(target, source.size);
}

static iree_status_t loom_kernel_product_validate_options(
    const loom_kernel_product_options_t* options) {
  if (options == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product options are NULL");
  }
  if (iree_string_view_is_empty(options->target_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product target key must not be empty");
  }
  if (options->target_bundle == NULL ||
      options->target_bundle->snapshot == NULL ||
      options->target_bundle->export_plan == NULL ||
      options->target_bundle->config == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel product requires a complete resolved target bundle");
  }
  if (options->artifact_count == 0 || options->artifacts == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product requires at least one artifact");
  }
  if (options->loadable_artifact_ordinal >= options->artifact_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "kernel product loadable artifact ordinal is out of range");
  }
  const loom_product_artifact_t* loadable_artifact =
      &options->artifacts[options->loadable_artifact_ordinal];
  if (!iree_string_view_equal(loadable_artifact->role,
                              IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel product loadable artifact must have the 'kernel' role");
  }
  for (iree_host_size_t i = 0; i < options->artifact_count; ++i) {
    const loom_product_artifact_t* artifact = &options->artifacts[i];
    if (iree_string_view_is_empty(artifact->role) ||
        iree_string_view_is_empty(artifact->format) ||
        iree_string_view_is_empty(artifact->identifier) ||
        artifact->contents == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel product artifact metadata must be complete");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_product_create(
    const loom_kernel_product_options_t* options, iree_allocator_t allocator,
    loom_product_t** out_product) {
  IREE_ASSERT_ARGUMENT(out_product);
  *out_product = NULL;
  if (iree_allocator_is_null(allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product allocator is null");
  }
  IREE_RETURN_IF_ERROR(loom_kernel_product_validate_options(options));

  iree_host_size_t string_storage_size = 0;
  if (!loom_kernel_product_calculate_string_storage_size(
          options, &string_storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel product string metadata is too large");
  }

  const loom_target_bundle_t* source_bundle = options->target_bundle;
  loom_kernel_product_t* product = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*product), (void**)&product);
  if (iree_status_is_ok(status)) {
    *product = (loom_kernel_product_t){
        .allocator = allocator,
        .loadable_artifact_ordinal = options->loadable_artifact_ordinal,
    };
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, options->artifact_count,
                                         sizeof(*product->artifacts),
                                         (void**)&product->artifacts);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, string_storage_size,
                                   (void**)&product->string_storage);
  }

  if (iree_status_is_ok(status)) {
    char* string_cursor = product->string_storage;
    product->target_key =
        loom_kernel_product_copy_string(options->target_key, &string_cursor);
    product->target_bundle_storage = (loom_target_bundle_storage_t){
        .snapshot = *source_bundle->snapshot,
        .export_plan = *source_bundle->export_plan,
        .config = *source_bundle->config,
        .bundle = *source_bundle,
    };
    product->target_bundle_storage.bundle.name =
        loom_kernel_product_copy_string(source_bundle->name, &string_cursor);
    product->target_bundle_storage.snapshot.name =
        loom_kernel_product_copy_string(source_bundle->snapshot->name,
                                        &string_cursor);
    product->target_bundle_storage.export_plan.name =
        loom_kernel_product_copy_string(source_bundle->export_plan->name,
                                        &string_cursor);
    product->target_bundle_storage.export_plan.export_symbol =
        loom_kernel_product_copy_string(
            source_bundle->export_plan->export_symbol, &string_cursor);
    product->target_bundle_storage.config.name =
        loom_kernel_product_copy_string(source_bundle->config->name,
                                        &string_cursor);
    product->target_bundle_storage.config.contract_set_key =
        loom_kernel_product_copy_string(source_bundle->config->contract_set_key,
                                        &string_cursor);
    loom_target_bundle_storage_rebind(&product->target_bundle_storage);

    for (iree_host_size_t i = 0; i < options->artifact_count; ++i) {
      product->artifacts[i] = (loom_product_artifact_t){
          .role = loom_kernel_product_copy_string(options->artifacts[i].role,
                                                  &string_cursor),
          .format = loom_kernel_product_copy_string(
              options->artifacts[i].format, &string_cursor),
          .identifier = loom_kernel_product_copy_string(
              options->artifacts[i].identifier, &string_cursor),
          .contents = options->artifacts[i].contents,
      };
      iree_byte_sequence_retain(product->artifacts[i].contents);
    }
    loom_product_initialize(&loom_kernel_product_descriptor, product->artifacts,
                            options->artifact_count, options->export_count,
                            options->requirement_count, &product->base);
    *out_product = &product->base;
    product = NULL;
  }

  if (product != NULL) {
    iree_allocator_free(allocator, product->string_storage);
    iree_allocator_free(allocator, product->artifacts);
    iree_allocator_free(allocator, product);
  }
  return status;
}

iree_string_view_t loom_kernel_product_target_key(
    const loom_product_t* base_product) {
  const loom_kernel_product_t* product = loom_kernel_product_cast(base_product);
  return product ? product->target_key : iree_string_view_empty();
}

const loom_target_bundle_t* loom_kernel_product_target_bundle(
    const loom_product_t* base_product) {
  const loom_kernel_product_t* product = loom_kernel_product_cast(base_product);
  return product ? &product->target_bundle_storage.bundle : NULL;
}

const loom_product_artifact_t* loom_kernel_product_loadable_artifact(
    const loom_product_t* base_product) {
  const loom_kernel_product_t* product = loom_kernel_product_cast(base_product);
  return product ? loom_product_artifact_at(&product->base,
                                            product->loadable_artifact_ordinal)
                 : NULL;
}
