// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/prepared_package.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "loom/util/json.h"

static iree_string_view_t loom_run_hal_testbench_prepared_package_schema(void) {
  return IREE_SV("loom.testbench.prepared_actuals.v0");
}

typedef struct loom_run_hal_testbench_prepared_package_decoder_t {
  // Next unused byte of the input-sized decoded string storage.
  char* next_string;
} loom_run_hal_testbench_prepared_package_decoder_t;

static iree_status_t loom_run_hal_testbench_prepared_package_decode_string(
    loom_run_hal_testbench_prepared_package_decoder_t* decoder,
    iree_string_view_t escaped, iree_string_view_t* out_value) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      escaped, escaped.size, decoder->next_string, &length));
  const iree_string_view_t value =
      iree_make_string_view(decoder->next_string, length);
  if (iree_string_view_is_empty(value) || memchr(value.data, 0, value.size) ||
      !iree_unicode_utf8_validate(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared-package strings must be nonempty UTF-8 without NUL");
  }
  decoder->next_string += length;
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_prepared_package_validate_string(
    iree_string_view_t value, const char* field_name) {
  if (iree_string_view_is_empty(value) || memchr(value.data, 0, value.size) ||
      !iree_unicode_utf8_validate(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared-package field '%s' must be nonempty UTF-8 without NUL",
        field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_prepared_package_validate_path(
    iree_string_view_t path) {
  if (iree_string_view_is_empty(path) || path.data[0] == '/' ||
      path.data[path.size - 1] == '/') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "package path '%.*s' must be relative and nonempty",
                            (int)path.size, path.data);
  }
  iree_host_size_t component_start = 0;
  for (iree_host_size_t i = 0; i <= path.size; ++i) {
    const bool at_end = i == path.size;
    const uint8_t c = at_end ? '/' : (uint8_t)path.data[i];
    if (!at_end && (c < 0x21 || c > 0x7E || c == '\\' || c == ':')) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "package path '%.*s' is not a portable relative path", (int)path.size,
          path.data);
    }
    if (c != '/') {
      continue;
    }
    const iree_host_size_t component_length = i - component_start;
    const iree_string_view_t component =
        iree_make_string_view(path.data + component_start, component_length);
    if (component_length == 0 ||
        iree_string_view_equal(component, IREE_SV(".")) ||
        iree_string_view_equal(component, IREE_SV(".."))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "package path '%.*s' contains an empty or traversal component",
          (int)path.size, path.data);
    }
    component_start = i + 1;
  }
  return iree_ok_status();
}

enum {
  LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_SCHEMA = 0,
  LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_TARGET_PROFILE = 1,
  LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_ARTIFACT_FORMAT = 2,
  LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_PRODUCTS = 3,
  LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_FIELD_COUNT = 4,
};

static const char* const
    loom_run_hal_testbench_prepared_package_root_field_names[] = {
        "schema", "target_profile", "artifact_format", "products"};

typedef struct loom_run_hal_testbench_prepared_package_root_fields_t {
  // Raw JSON string contents or the products array; null data marks omission.
  iree_string_view_t values[LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_FIELD_COUNT];
} loom_run_hal_testbench_prepared_package_root_fields_t;

static iree_status_t loom_run_hal_testbench_prepared_package_root_field(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_run_hal_testbench_prepared_package_root_fields_t* fields = user_data;
  iree_host_size_t index = 0;
  while (index < LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_FIELD_COUNT &&
         !iree_string_view_equal(
             key, iree_make_cstring_view(
                      loom_run_hal_testbench_prepared_package_root_field_names
                          [index]))) {
    ++index;
  }
  if (index == LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_FIELD_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown prepared-package field '%.*s'",
                            (int)key.size, key.data);
  }
  const iree_json_value_type_t expected_type =
      index == LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_PRODUCTS
          ? IREE_JSON_VALUE_TYPE_ARRAY
          : IREE_JSON_VALUE_TYPE_STRING;
  if (fields->values[index].data != NULL || type != expected_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate or wrongly typed prepared-package field '%s'",
        loom_run_hal_testbench_prepared_package_root_field_names[index]);
  }
  fields->values[index] = value;
  return iree_ok_status();
}

enum {
  LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_SUBJECT = 0,
  LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_PREPARED_MODULE = 1,
  LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_LAUNCH_CONFIG = 2,
  LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_FIELD_COUNT = 3,
};

static const char* const
    loom_run_hal_testbench_prepared_package_product_field_names[] = {
        "subject", "prepared_module", "launch_config"};

typedef struct loom_run_hal_testbench_prepared_package_product_fields_t {
  // Raw JSON string contents; null data marks omission.
  iree_string_view_t values[LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_FIELD_COUNT];
} loom_run_hal_testbench_prepared_package_product_fields_t;

static iree_status_t loom_run_hal_testbench_prepared_package_product_field(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_run_hal_testbench_prepared_package_product_fields_t* fields = user_data;
  iree_host_size_t index = 0;
  while (
      index < LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_FIELD_COUNT &&
      !iree_string_view_equal(
          key, iree_make_cstring_view(
                   loom_run_hal_testbench_prepared_package_product_field_names
                       [index]))) {
    ++index;
  }
  if (index == LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_FIELD_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown prepared-package product field '%.*s'",
                            (int)key.size, key.data);
  }
  if (fields->values[index].data != NULL ||
      type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate or wrongly typed prepared-package product field '%s'",
        loom_run_hal_testbench_prepared_package_product_field_names[index]);
  }
  fields->values[index] = value;
  return iree_ok_status();
}

typedef struct loom_run_hal_testbench_prepared_package_product_parser_t {
  // Shared decoded-string cursor.
  loom_run_hal_testbench_prepared_package_decoder_t* decoder;
  // Manifest whose product rows are populated in array order.
  loom_run_hal_testbench_prepared_package_manifest_t* manifest;
} loom_run_hal_testbench_prepared_package_product_parser_t;

static iree_status_t loom_run_hal_testbench_prepared_package_product(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_run_hal_testbench_prepared_package_product_parser_t* parser = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      index >= parser->manifest->product_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared-package products must be objects");
  }
  loom_run_hal_testbench_prepared_package_product_fields_t fields = {0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      value, loom_run_hal_testbench_prepared_package_product_field, &fields));
  for (iree_host_size_t i = 0;
       i < LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_FIELD_COUNT; ++i) {
    if (fields.values[i].data == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepared-package product %zu requires field '%s'", index,
          loom_run_hal_testbench_prepared_package_product_field_names[i]);
    }
  }

  loom_run_hal_testbench_prepared_package_product_t* product =
      &parser->manifest->products[index];
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_decode_string(
      parser->decoder,
      fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_SUBJECT],
      &product->subject));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_decode_string(
      parser->decoder,
      fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_PREPARED_MODULE],
      &product->prepared_module_path));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_decode_string(
      parser->decoder,
      fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_PRODUCT_LAUNCH_CONFIG],
      &product->launch_config_path));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_validate_path(
      product->prepared_module_path));
  return loom_run_hal_testbench_prepared_package_validate_path(
      product->launch_config_path);
}

static iree_status_t loom_run_hal_testbench_prepared_package_validate(
    const loom_run_hal_testbench_prepared_package_manifest_t* manifest) {
  if (manifest == NULL || manifest->product_count == 0 ||
      manifest->products == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared-package manifest requires a target, format, and products");
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_validate_string(
      manifest->target_profile, "target_profile"));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_validate_string(
      manifest->artifact_format, "artifact_format"));
  for (iree_host_size_t i = 0; i < manifest->product_count; ++i) {
    const loom_run_hal_testbench_prepared_package_product_t* product =
        &manifest->products[i];
    IREE_RETURN_IF_ERROR(
        loom_run_hal_testbench_prepared_package_validate_string(
            product->subject, "subject"));
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_validate_path(
        product->prepared_module_path));
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_prepared_package_validate_path(
        product->launch_config_path));
    if (iree_string_view_equal(product->prepared_module_path,
                               product->launch_config_path)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepared-package prepared and launch artifacts must have distinct "
          "paths");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(product->subject,
                                 manifest->products[j].subject)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate prepared-package subject '%.*s'",
                                (int)product->subject.size,
                                product->subject.data);
      }
      if (iree_string_view_equal(product->prepared_module_path,
                                 manifest->products[j].launch_config_path) ||
          iree_string_view_equal(product->launch_config_path,
                                 manifest->products[j].prepared_module_path)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "prepared-package prepared and launch artifact paths must not "
            "alias");
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_prepared_package_manifest_parse(
    iree_string_view_t json, iree_allocator_t host_allocator,
    loom_run_hal_testbench_prepared_package_manifest_t* out_manifest) {
  *out_manifest = (loom_run_hal_testbench_prepared_package_manifest_t){0};
  iree_string_view_t cursor = json;
  iree_string_view_t root = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_consume_value(&cursor, &root));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (iree_string_view_is_empty(root) || root.data[0] != '{' ||
      !iree_string_view_is_empty(cursor)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared-package manifest must be one JSON object");
  }

  loom_run_hal_testbench_prepared_package_root_fields_t fields = {0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      root, loom_run_hal_testbench_prepared_package_root_field, &fields));
  for (iree_host_size_t i = 0;
       i < LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_FIELD_COUNT; ++i) {
    if (fields.values[i].data == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepared-package manifest requires field '%s'",
          loom_run_hal_testbench_prepared_package_root_field_names[i]);
    }
  }

  iree_host_size_t product_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(
      fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_PRODUCTS],
      &product_count));
  if (product_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared-package manifest requires products");
  }

  loom_run_hal_testbench_prepared_package_manifest_t manifest = {
      .host_allocator = host_allocator,
      .product_count = product_count,
  };
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, product_count, sizeof(*manifest.products),
      (void**)&manifest.products);
  if (iree_status_is_ok(status)) {
    memset(manifest.products, 0, product_count * sizeof(*manifest.products));
    status =
        iree_allocator_malloc(host_allocator, json.size, &manifest.storage);
  }
  loom_run_hal_testbench_prepared_package_decoder_t decoder = {
      .next_string = manifest.storage,
  };
  if (iree_status_is_ok(status)) {
    iree_string_view_t schema = iree_string_view_empty();
    status = loom_run_hal_testbench_prepared_package_decode_string(
        &decoder, fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_SCHEMA],
        &schema);
    if (iree_status_is_ok(status) &&
        !iree_string_view_equal(
            schema, loom_run_hal_testbench_prepared_package_schema())) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported prepared-package schema '%.*s'",
                                (int)schema.size, schema.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_decode_string(
        &decoder,
        fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_TARGET_PROFILE],
        &manifest.target_profile);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_decode_string(
        &decoder,
        fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_ARTIFACT_FORMAT],
        &manifest.artifact_format);
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_testbench_prepared_package_product_parser_t parser = {
        .decoder = &decoder,
        .manifest = &manifest,
    };
    status = iree_json_enumerate_array_typed(
        fields.values[LOOM_RUN_HAL_PREPARED_PACKAGE_ROOT_PRODUCTS],
        loom_run_hal_testbench_prepared_package_product, &parser);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_validate(&manifest);
  }
  if (iree_status_is_ok(status)) {
    *out_manifest = manifest;
  } else {
    loom_run_hal_testbench_prepared_package_manifest_deinitialize(&manifest);
  }
  return status;
}

void loom_run_hal_testbench_prepared_package_manifest_deinitialize(
    loom_run_hal_testbench_prepared_package_manifest_t* manifest) {
  if (manifest == NULL) {
    return;
  }
  iree_allocator_free(manifest->host_allocator, manifest->storage);
  iree_allocator_free(manifest->host_allocator, manifest->products);
  *manifest = (loom_run_hal_testbench_prepared_package_manifest_t){0};
}

iree_status_t loom_run_hal_testbench_prepared_package_manifest_write(
    const loom_run_hal_testbench_prepared_package_manifest_t* manifest,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_prepared_package_validate(manifest));
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("schema"),
      loom_run_hal_testbench_prepared_package_schema()));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("target_profile"), manifest->target_profile));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("artifact_format"), manifest->artifact_format));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("products")));
  loom_json_array_writer_t products;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &products));
  for (iree_host_size_t i = 0; i < manifest->product_count; ++i) {
    const loom_run_hal_testbench_prepared_package_product_t* product =
        &manifest->products[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&products));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("subject"), product->subject));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("prepared_module"), product->prepared_module_path));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("launch_config"), product->launch_config_path));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&products));
  return loom_json_object_end(&root);
}

const loom_run_hal_testbench_prepared_package_product_t*
loom_run_hal_testbench_prepared_package_manifest_find(
    const loom_run_hal_testbench_prepared_package_manifest_t* manifest,
    iree_string_view_t subject) {
  if (manifest == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < manifest->product_count; ++i) {
    if (iree_string_view_equal(manifest->products[i].subject, subject)) {
      return &manifest->products[i];
    }
  }
  return NULL;
}
