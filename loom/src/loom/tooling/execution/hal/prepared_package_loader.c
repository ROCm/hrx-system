// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/prepared_package_loader.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/tooling/execution/hal/prepared_package.h"
#include "loom/tooling/io/file.h"
#include "loomc/iree.h"

typedef struct loom_run_hal_testbench_prepared_package_payload_t {
  // Prepared-low module used as the target emitter input.
  loomc_module_t* module;
  // Owned launch-config byte storage referenced by |launch_config|.
  loomc_byte_sequence_t* launch_config_contents;
  // Borrowing launch-config artifact passed to the prepared actual provider.
  loomc_artifact_t launch_config;
} loom_run_hal_testbench_prepared_package_payload_t;

struct loom_run_hal_testbench_prepared_package_t {
  // Allocator owning this package and its arrays and paths.
  iree_allocator_t host_allocator;
  // Parsed manifest owning all manifest strings and product rows.
  loom_run_hal_testbench_prepared_package_manifest_t manifest;
  // Full filesystem path of the untouched authored module.
  char* authored_module_path;
  // Exact target profile selected from |manifest.target_profile|.
  loomc_target_profile_t* target_profile;
  // Unique prepared-module and launch-config payload pairs.
  loom_run_hal_testbench_prepared_package_payload_t* payloads;
  // Number of initialized entries in |payloads|.
  iree_host_size_t payload_count;
  // Payload index for each manifest product row.
  iree_host_size_t* product_payload_indices;
};

static iree_status_t loom_run_hal_testbench_prepared_package_join_path(
    iree_string_view_t package_directory, iree_string_view_t relative_path,
    iree_allocator_t host_allocator, char** out_path) {
  return loom_tooling_file_path_join(package_directory, relative_path,
                                     host_allocator, out_path);
}

static iree_status_t loom_run_hal_testbench_prepared_package_load_module(
    iree_string_view_t path, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_allocator_t host_allocator,
    loomc_module_t** out_module) {
  *out_module = NULL;
  const loomc_source_load_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
  };
  loomc_source_t* source = NULL;
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_source_create_from_path(
      loomc_string_view_from_iree(path), &source_options,
      loomc_allocator_from_iree(host_allocator), &source));
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        context, workspace, source, NULL,
        loomc_allocator_from_iree(host_allocator), out_module, &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_require_successful_result(
        result, IREE_SV("prepared package module could not be deserialized"));
  }
  loomc_result_release(result);
  loomc_source_release(source);
  return status;
}

static iree_status_t loom_run_hal_testbench_prepared_package_load_launch(
    iree_string_view_t path, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_prepared_package_payload_t* payload) {
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(path, host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_byte_sequence_create_copy(
        loomc_byte_span_from_iree(iree_make_const_byte_span(
            contents->const_buffer.data, contents->const_buffer.data_length)),
        loomc_allocator_from_iree(host_allocator),
        &payload->launch_config_contents));
  }
  iree_io_file_contents_free(contents);
  if (iree_status_is_ok(status)) {
    payload->launch_config = (loomc_artifact_t){
        .kind = LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        .format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
        .identifier = loomc_string_view_from_iree(identifier),
        .contents = payload->launch_config_contents,
    };
  }
  return status;
}

static iree_status_t loom_run_hal_testbench_prepared_package_load_payload(
    iree_string_view_t package_directory,
    const loom_run_hal_testbench_prepared_package_product_t* product,
    loomc_context_t* context, loomc_workspace_t* workspace,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_prepared_package_payload_t* out_payload) {
  *out_payload = (loom_run_hal_testbench_prepared_package_payload_t){0};
  char* module_path = NULL;
  char* launch_path = NULL;
  iree_status_t status = loom_run_hal_testbench_prepared_package_join_path(
      package_directory, product->prepared_module_path, host_allocator,
      &module_path);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_join_path(
        package_directory, product->launch_config_path, host_allocator,
        &launch_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_load_module(
        iree_make_cstring_view(module_path), context, workspace, host_allocator,
        &out_payload->module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_load_launch(
        iree_make_cstring_view(launch_path), product->launch_config_path,
        host_allocator, out_payload);
  }
  iree_allocator_free(host_allocator, launch_path);
  iree_allocator_free(host_allocator, module_path);
  return status;
}

static void loom_run_hal_testbench_prepared_package_payload_deinitialize(
    loom_run_hal_testbench_prepared_package_payload_t* payload) {
  loomc_byte_sequence_release(payload->launch_config_contents);
  loomc_module_release(payload->module);
  *payload = (loom_run_hal_testbench_prepared_package_payload_t){0};
}

static iree_host_size_t
loom_run_hal_testbench_prepared_package_find_shared_payload(
    const loom_run_hal_testbench_prepared_package_manifest_t* manifest,
    iree_host_size_t product_index,
    const iree_host_size_t* product_payload_indices) {
  const loom_run_hal_testbench_prepared_package_product_t* product =
      &manifest->products[product_index];
  for (iree_host_size_t i = 0; i < product_index; ++i) {
    const loom_run_hal_testbench_prepared_package_product_t* candidate =
        &manifest->products[i];
    if (iree_string_view_equal(product->prepared_module_path,
                               candidate->prepared_module_path) &&
        iree_string_view_equal(product->launch_config_path,
                               candidate->launch_config_path)) {
      return product_payload_indices[i];
    }
  }
  return IREE_HOST_SIZE_MAX;
}

iree_status_t loom_run_hal_testbench_prepared_package_open(
    iree_string_view_t package_directory,
    loomc_target_environment_t* target_environment, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_allocator_t host_allocator,
    loom_run_hal_testbench_prepared_package_t** out_package) {
  *out_package = NULL;
  if (iree_string_view_is_empty(package_directory) ||
      target_environment == NULL || context == NULL || workspace == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared package requires a directory and compiler environment");
  }

  loom_run_hal_testbench_prepared_package_t* package = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*package),
                                             (void**)&package));
  *package = (loom_run_hal_testbench_prepared_package_t){
      .host_allocator = host_allocator,
  };
  char* manifest_path = NULL;
  iree_io_file_contents_t* manifest_contents = NULL;
  iree_status_t status = loom_run_hal_testbench_prepared_package_join_path(
      package_directory, IREE_SV("manifest.json"), host_allocator,
      &manifest_path);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(iree_make_cstring_view(manifest_path),
                                          host_allocator, &manifest_contents);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_manifest_parse(
        loom_tooling_file_contents_string_view(manifest_contents),
        host_allocator, &package->manifest);
  }
  iree_io_file_contents_free(manifest_contents);
  iree_allocator_free(host_allocator, manifest_path);

  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_prepared_package_join_path(
        package_directory, package->manifest.authored_module_path,
        host_allocator, &package->authored_module_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        target_environment,
        loomc_string_view_from_iree(package->manifest.target_profile),
        loomc_allocator_from_iree(host_allocator), &package->target_profile));
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, package->manifest.product_count,
        sizeof(*package->payloads), (void**)&package->payloads);
  }
  if (iree_status_is_ok(status)) {
    memset(package->payloads, 0,
           package->manifest.product_count * sizeof(*package->payloads));
    status = iree_allocator_malloc_array(
        host_allocator, package->manifest.product_count,
        sizeof(*package->product_payload_indices),
        (void**)&package->product_payload_indices);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < package->manifest.product_count; ++i) {
    const iree_host_size_t shared_index =
        loom_run_hal_testbench_prepared_package_find_shared_payload(
            &package->manifest, i, package->product_payload_indices);
    if (shared_index != IREE_HOST_SIZE_MAX) {
      package->product_payload_indices[i] = shared_index;
      continue;
    }
    const iree_host_size_t payload_index = package->payload_count++;
    package->product_payload_indices[i] = payload_index;
    status = loom_run_hal_testbench_prepared_package_load_payload(
        package_directory, &package->manifest.products[i], context, workspace,
        host_allocator, &package->payloads[payload_index]);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_testbench_prepared_package_release(package);
    return status;
  }
  *out_package = package;
  return iree_ok_status();
}

void loom_run_hal_testbench_prepared_package_release(
    loom_run_hal_testbench_prepared_package_t* package) {
  if (package == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < package->payload_count; ++i) {
    loom_run_hal_testbench_prepared_package_payload_deinitialize(
        &package->payloads[i]);
  }
  iree_allocator_free(package->host_allocator,
                      package->product_payload_indices);
  iree_allocator_free(package->host_allocator, package->payloads);
  loomc_target_profile_release(package->target_profile);
  iree_allocator_free(package->host_allocator, package->authored_module_path);
  loom_run_hal_testbench_prepared_package_manifest_deinitialize(
      &package->manifest);
  iree_allocator_t host_allocator = package->host_allocator;
  iree_allocator_free(host_allocator, package);
}

iree_string_view_t loom_run_hal_testbench_prepared_package_authored_module_path(
    const loom_run_hal_testbench_prepared_package_t* package) {
  return package != NULL && package->authored_module_path != NULL
             ? iree_make_cstring_view(package->authored_module_path)
             : iree_string_view_empty();
}

iree_string_view_t loom_run_hal_testbench_prepared_package_target_profile(
    const loom_run_hal_testbench_prepared_package_t* package) {
  return package != NULL ? package->manifest.target_profile
                         : iree_string_view_empty();
}

static iree_status_t loom_run_hal_testbench_prepared_package_resolve(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    loom_run_hal_testbench_prepared_actual_t* out_prepared_actual) {
  *out_prepared_actual = (loom_run_hal_testbench_prepared_actual_t){0};
  loom_run_hal_testbench_prepared_package_t* package = user_data;
  if (configuration->slot_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "prepared package v0 does not support configured scenarios");
  }
  if (invocation->callee_ref.module_id != 0 ||
      invocation->callee_ref.symbol_id >= invocation->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared package received an invalid subject");
  }
  const loom_symbol_t* symbol =
      &invocation->module->symbols.entries[invocation->callee_ref.symbol_id];
  if (symbol->name_id >= invocation->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared package subject has an invalid name");
  }
  const iree_string_view_t subject =
      loom_string_table_get(&invocation->module->strings, symbol->name_id);
  const loom_run_hal_testbench_prepared_package_product_t* product =
      loom_run_hal_testbench_prepared_package_manifest_find(&package->manifest,
                                                            subject);
  if (product == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "prepared package has no product for scenario subject '%.*s'",
        (int)subject.size, subject.data);
  }
  const iree_host_size_t product_index =
      (iree_host_size_t)(product - package->manifest.products);
  const iree_host_size_t payload_index =
      package->product_payload_indices[product_index];
  const loom_run_hal_testbench_prepared_package_payload_t* payload =
      &package->payloads[payload_index];
  *out_prepared_actual = (loom_run_hal_testbench_prepared_actual_t){
      .module = payload->module,
      .launch_config = &payload->launch_config,
      .target_profile = package->target_profile,
      .artifact_format =
          loomc_string_view_from_iree(package->manifest.artifact_format),
  };
  return iree_ok_status();
}

loom_run_hal_testbench_prepared_actual_resolver_t
loom_run_hal_testbench_prepared_package_resolver(
    loom_run_hal_testbench_prepared_package_t* package) {
  return (loom_run_hal_testbench_prepared_actual_resolver_t){
      .fn = loom_run_hal_testbench_prepared_package_resolve,
      .user_data = package,
  };
}
