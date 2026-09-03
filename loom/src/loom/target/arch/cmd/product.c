// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/product.h"

#include <string.h>

typedef struct loom_cmd_product_t {
  // Generic immutable product interface.
  loom_product_t base;

  // Allocator used for this product and its generic artifact table.
  iree_allocator_t allocator;

  // Owned serialized programs and copied requirement metadata.
  loom_cmd_program_artifact_set_t artifact_set;
} loom_cmd_product_t;

static const iree_string_view_t kLoomCmdProductRootOperationNames[] = {
    IREE_SVL("command.program.def"),
};

static const loom_product_artifact_schema_t kLoomCmdProductArtifactSchemas[] = {
    {
        .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_COMMAND_PROGRAM),
        .format = IREE_SVL(LOOM_CMD_PRODUCT_FORMAT_LOOM_COMMAND),
        .minimum_count = 1,
        .maximum_count = IREE_HOST_SIZE_MAX,
    },
};

static void loom_cmd_product_destroy(loom_product_t* base_product) {
  loom_cmd_product_t* product = (loom_cmd_product_t*)base_product;
  const iree_allocator_t allocator = product->allocator;
  loom_cmd_program_artifact_set_deinitialize(&product->artifact_set);
  iree_allocator_free(allocator, product);
}

const loom_product_descriptor_t loom_cmd_product_descriptor = {
    .name = IREE_SVL("command"),
    .destroy = loom_cmd_product_destroy,
};

const loom_product_operation_t loom_cmd_product_operation = {
    .name = IREE_SVL("command"),
    .product_descriptor = &loom_cmd_product_descriptor,
    .root_operation_names = kLoomCmdProductRootOperationNames,
    .root_operation_name_count =
        IREE_ARRAYSIZE(kLoomCmdProductRootOperationNames),
};

const loom_product_format_t loom_cmd_product_format = {
    .operation = &loom_cmd_product_operation,
    .name = IREE_SVL(LOOM_CMD_PRODUCT_FORMAT_LOOM_COMMAND),
    .persistence = LOOM_PRODUCT_PERSISTENCE_ARTIFACT_SET,
    .artifact_schemas = kLoomCmdProductArtifactSchemas,
    .artifact_schema_count = IREE_ARRAYSIZE(kLoomCmdProductArtifactSchemas),
};

static const loom_cmd_product_t* loom_cmd_product_cast(
    const loom_product_t* base_product) {
  return loom_product_isa(base_product, &loom_cmd_product_descriptor)
             ? (const loom_cmd_product_t*)base_product
             : NULL;
}

static iree_status_t loom_cmd_product_validate_artifact_set(
    const loom_cmd_program_artifact_set_t* artifact_set) {
  if (artifact_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command artifact set is NULL");
  }
  if (iree_allocator_is_null(artifact_set->host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command artifact set has no host allocator");
  }
  if (artifact_set->programs.count == 0 ||
      artifact_set->programs.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command product requires at least one program");
  }
  if (artifact_set->entries.count != 0 &&
      artifact_set->entries.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command product entry table is missing its storage");
  }
  for (iree_host_size_t i = 0; i < artifact_set->entries.count; ++i) {
    if (iree_string_view_is_empty(artifact_set->entries.values[i].symbol)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command product entry requirement has an empty symbol");
    }
  }
  for (iree_host_size_t i = 0; i < artifact_set->programs.count; ++i) {
    const loom_cmd_program_artifact_t* program =
        &artifact_set->programs.values[i];
    if (iree_string_view_is_empty(program->symbol) || program->data == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command product program metadata must be complete");
    }
    if (program->entry_requirement_count != 0 &&
        program->entry_requirement_indices == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command product program entry projection is missing its storage");
    }
    for (uint32_t j = 0; j < program->entry_requirement_count; ++j) {
      if (program->entry_requirement_indices[j] >=
          artifact_set->entries.count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "command product program entry projection is out of range");
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_product_create(
    loom_cmd_program_artifact_set_t* inout_artifact_set,
    iree_allocator_t allocator, loom_product_t** out_product) {
  IREE_ASSERT_ARGUMENT(out_product);
  *out_product = NULL;
  if (iree_allocator_is_null(allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command product allocator is null");
  }
  IREE_RETURN_IF_ERROR(
      loom_cmd_product_validate_artifact_set(inout_artifact_set));

  iree_host_size_t artifact_storage_size = 0;
  iree_host_size_t allocation_size = sizeof(loom_cmd_product_t);
  if (!iree_host_size_checked_mul(inout_artifact_set->programs.count,
                                  sizeof(loom_product_artifact_t),
                                  &artifact_storage_size) ||
      !iree_host_size_checked_add(allocation_size, artifact_storage_size,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command product metadata is too large");
  }

  loom_cmd_product_t* product = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&product));
  memset(product, 0, sizeof(*product));
  product->allocator = allocator;
  product->artifact_set = *inout_artifact_set;
  *inout_artifact_set = (loom_cmd_program_artifact_set_t){0};

  loom_product_artifact_t* artifacts = (loom_product_artifact_t*)(product + 1);
  for (iree_host_size_t i = 0; i < product->artifact_set.programs.count; ++i) {
    const loom_cmd_program_artifact_t* program =
        &product->artifact_set.programs.values[i];
    artifacts[i] = (loom_product_artifact_t){
        .role = IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_COMMAND_PROGRAM),
        .format = IREE_SV(LOOM_CMD_PRODUCT_FORMAT_LOOM_COMMAND),
        .identifier = program->symbol,
        .contents = program->data,
    };
  }
  loom_product_initialize(&loom_cmd_product_descriptor, artifacts,
                          product->artifact_set.programs.count,
                          product->artifact_set.programs.count,
                          product->artifact_set.entries.count, &product->base);
  *out_product = &product->base;
  return iree_ok_status();
}

const loom_cmd_program_artifact_set_t* loom_cmd_product_artifact_set(
    const loom_product_t* base_product) {
  const loom_cmd_product_t* product = loom_cmd_product_cast(base_product);
  return product ? &product->artifact_set : NULL;
}
