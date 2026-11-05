// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/invocation.h"

#include <stdlib.h>

#include "libamdf/tool/amdf-info/activation.h"
#include "libamdf/tool/amdf-info/platform.h"

static void amdf_info_query_extensions(amdf_info_library_t* library) {
  const void* table = NULL;
  library->gpu_status = library->api->query_extension(
      AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
      AMDF_GPU_EXTENSION_VERSION_LATEST, &table);
  if (amdf_status_is_ok(library->gpu_status)) {
    library->gpu = (const amdf_gpu_api_t*)table;
  }
  table = NULL;
  library->xdna_status = library->api->query_extension(
      AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
      AMDF_XDNA_EXTENSION_VERSION_LATEST, &table);
  if (amdf_status_is_ok(library->xdna_status)) {
    library->xdna = (const amdf_xdna_api_t*)table;
  }
}

static void amdf_info_query_pools(const amdf_api_t* api,
                                  amdf_instance_t* instance,
                                  amdf_info_output_t* output,
                                  amdf_info_result_t* result) {
  uint32_t count = 0;
  amdf_status_t status =
      api->instance_enumerate_memory_scopes(instance, 0, NULL, &count);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    status = AMDF_STATUS_OK;
  }
  amdf_memory_scope_t** scopes = NULL;
  if (amdf_status_is_ok(status) && count) {
    scopes = (amdf_memory_scope_t**)calloc(count, sizeof(*scopes));
    status = scopes ? api->instance_enumerate_memory_scopes(instance, count,
                                                            scopes, &count)
                    : amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  if (amdf_info_report_query(output, result, "memory_pools",
                             "instance_enumerate_memory_scopes", status,
                             AMDF_INFO_QUERY_OPTIONAL)) {
    amdf_info_report_pools_begin(output);
    for (uint32_t i = 0; !output->failed && i < count; ++i) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      const amdf_status_t query_status =
          api->memory_scope_query_info(scopes[i], &info);
      if (amdf_info_report_query(output, result, NULL,
                                 "memory_scope_query_info", query_status,
                                 AMDF_INFO_QUERY_OPTIONAL)) {
        amdf_info_report_pool(output, i, &info);
      }
    }
    amdf_info_report_pools_end(output);
  }
  free(scopes);
}

static void amdf_info_query_location(const amdf_api_t* api,
                                     const amdf_info_endpoint_t* endpoint,
                                     amdf_info_output_t* output,
                                     amdf_info_result_t* result) {
  amdf_endpoint_pci_address_t address = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS,
      .structure_size = sizeof(address),
  };
  amdf_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(info),
      .next = &address,
  };
  const amdf_status_t address_status =
      api->endpoint_query_info(endpoint->handle, &info);
  if (amdf_info_report_query(output, result, "pci_address",
                             "endpoint_query_info(PCI address)", address_status,
                             AMDF_INFO_QUERY_OPTIONAL)) {
    amdf_info_report_pci_address(output, &address);
  }
  amdf_endpoint_host_locality_t locality = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_HOST_LOCALITY,
      .structure_size = sizeof(locality),
  };
  info.next = &locality;
  const amdf_status_t locality_status =
      api->endpoint_query_info(endpoint->handle, &info);
  if (amdf_info_report_query(output, result, "host_numa_node",
                             "endpoint_query_info(host locality)",
                             locality_status, AMDF_INFO_QUERY_OPTIONAL)) {
    amdf_info_report_host_locality(output, &locality);
  }
}

static void amdf_info_query_endpoint(const amdf_info_library_t* library,
                                     amdf_info_endpoint_t* endpoint,
                                     const amdf_info_options_t* options,
                                     amdf_info_output_t* output,
                                     amdf_info_result_t* result) {
  amdf_info_report_endpoint_begin(output, endpoint);
  if (endpoint->stage < AMDF_INFO_ENDPOINT_TARGET) {
    const char* operation = endpoint->stage == AMDF_INFO_ENDPOINT_SUMMARY
                                ? "endpoint_open"
                            : endpoint->stage == AMDF_INFO_ENDPOINT_OPEN
                                ? "endpoint_query_info"
                                : "target endpoint_query_info";
    amdf_info_report_query(
        output, result, "qualification", operation, endpoint->status,
        options->activate || options->select || options->device
            ? AMDF_INFO_QUERY_REQUIRED
            : AMDF_INFO_QUERY_OPTIONAL);
  }
  if (endpoint->stage >= AMDF_INFO_ENDPOINT_INFO) {
    amdf_info_report_endpoint_identity(output, &endpoint->info);
    if (endpoint->stage == AMDF_INFO_ENDPOINT_TARGET) {
      if (endpoint->info.engine_kind == AMDF_ENGINE_KIND_GPU) {
        amdf_info_report_gpu_target(output, &endpoint->target.gpu);
      } else {
        amdf_info_report_xdna_target(output, &endpoint->target.xdna);
      }
    }
    amdf_info_query_location(library->api, endpoint, output, result);
    amdf_info_report_queues_begin(output);
    for (uint32_t i = 0;
         !output->failed && i < endpoint->info.queue_family_count; ++i) {
      amdf_queue_family_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
          .structure_size = sizeof(info),
      };
      const amdf_status_t status =
          library->api->endpoint_query_queue_family_info(endpoint->handle, i,
                                                         &info);
      if (amdf_info_report_query(output, result, NULL,
                                 "endpoint_query_queue_family_info", status,
                                 AMDF_INFO_QUERY_OPTIONAL)) {
        amdf_info_report_queue(output, &info);
      }
    }
    amdf_info_report_queues_end(output);
    if (options->activate && endpoint->stage == AMDF_INFO_ENDPOINT_TARGET &&
        !output->failed) {
      amdf_info_activate(library, endpoint, output, result);
    }
  }
  amdf_info_report_endpoint_end(output);
}

static void amdf_info_cleanup(const amdf_api_t* api, amdf_instance_t* instance,
                              uint32_t count, amdf_info_endpoint_t* endpoints,
                              amdf_info_output_t* output,
                              amdf_info_result_t* result) {
  bool retained = false;
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "cleanup_errors", '[');
  }
  for (uint32_t i = 0; i < count; ++i) {
    amdf_info_endpoint_t* endpoint = &endpoints[i];
    if (endpoint->retain_on_exit) {
      retained = true;
    } else if (endpoint->handle) {
      const amdf_status_t status = api->endpoint_close(endpoint->handle);
      if (!amdf_status_is_ok(status)) {
        char uri[48];
        amdf_info_selection_uri(endpoint, uri);
        if (output->format == AMDF_INFO_FORMAT_JSON) {
          amdf_info_json_begin(output, NULL, '{');
          amdf_info_json_string(output, "uri", uri[0] ? uri : NULL);
        } else {
          amdf_info_output_print(output, "\nCleanup: ");
          amdf_info_output_text(output, uri[0] ? uri : endpoint->summary.name);
          amdf_info_output_print(output, "\n");
        }
        amdf_info_report_query(output, result, "endpoint", "endpoint_close",
                               status, AMDF_INFO_QUERY_REQUIRED);
        if (output->format == AMDF_INFO_FORMAT_JSON) {
          amdf_info_json_end(output, '}');
        }
        retained = true;
      }
    }
  }
  if (!retained) {
    amdf_info_report_query(output, result, NULL, "instance_destroy",
                           api->instance_destroy(instance),
                           AMDF_INFO_QUERY_REQUIRED);
  }
  // Failed native teardown retains its owner and borrowed parents until process
  // exit. This one-shot client neither retries nor keeps a cleanup registry.
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, ']');
  }
  free(endpoints);
}

amdf_info_exit_code_t amdf_info_invoke(const amdf_info_options_t* options,
                                       amdf_info_selector_t selector,
                                       amdf_query_api_fn_t query_api,
                                       FILE* output_file) {
  amdf_info_output_t output;
  const bool terminal_color = options->format == AMDF_INFO_FORMAT_TEXT &&
                              options->color == AMDF_INFO_COLOR_AUTO &&
                              amdf_info_platform_has_color(output_file);
  amdf_info_output_initialize(output_file, options, terminal_color, &output);
  amdf_info_result_t result = {.complete = true};
  amdf_info_report_begin(&output, amdf_info_platform_name());
  amdf_info_library_t library = {0};
  amdf_status_t status =
      query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &library.api);
  amdf_instance_t* instance = NULL;
  if (amdf_info_report_query(&output, &result, "library", "amdf_query_api",
                             status, AMDF_INFO_QUERY_REQUIRED) &&
      !output.failed) {
    amdf_info_query_extensions(&library);
    amdf_info_report_library(&output, &library);
    if ((!amdf_status_is_ok(library.gpu_status) &&
         !amdf_info_status_is_optional(library.gpu_status)) ||
        (!amdf_status_is_ok(library.xdna_status) &&
         !amdf_info_status_is_optional(library.xdna_status))) {
      result.complete = false;
      result.exit_code = AMDF_INFO_EXIT_FAILURE;
    }
    if (selector.kind == AMDF_INFO_SELECTOR_NATIVE_PATH) {
      status = amdf_info_platform_resolve_path(selector.path,
                                               &selector.native_identity);
      amdf_info_report_query(&output, &result, "device_path",
                             "resolve device path", status,
                             AMDF_INFO_QUERY_REQUIRED);
    }
    if (amdf_status_is_ok(status) && !output.failed) {
      const amdf_instance_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
          .structure_size = sizeof(create_info),
      };
      status = library.api->instance_create(&create_info, &instance);
      amdf_info_report_query(&output, &result, "instance", "instance_create",
                             status, AMDF_INFO_QUERY_REQUIRED);
    }
  }
  if (instance) {
    amdf_info_query_pools(library.api, instance, &output, &result);
    amdf_info_endpoint_t* endpoints = NULL;
    uint32_t count = 0;
    if (!output.failed) {
      status = amdf_info_inventory_read(&library, instance, &count, &endpoints);
      if (amdf_info_report_query(&output, &result, "enumeration",
                                 "endpoint_enumerate", status,
                                 AMDF_INFO_QUERY_REQUIRED)) {
        const amdf_info_exit_code_t selection =
            amdf_info_selection_apply(options, &selector, count, endpoints);
        if (selection != AMDF_INFO_EXIT_SUCCESS) {
          result.complete = false;
          if (result.exit_code == AMDF_INFO_EXIT_SUCCESS) {
            result.exit_code = selection;
          }
          amdf_info_report_selection_error(&output, selection);
        }
        if (count) {
          qsort(endpoints, count, sizeof(*endpoints),
                amdf_info_endpoint_compare);
        }
        amdf_info_report_endpoints_begin(&output);
        // A failed or ambiguous selection may be displayed, but never
        // activated.
        amdf_info_options_t report_options = *options;
        report_options.activate =
            options->activate && selection == AMDF_INFO_EXIT_SUCCESS;
        for (uint32_t i = 0; !output.failed && i < count; ++i) {
          if (endpoints[i].selected) {
            amdf_info_query_endpoint(&library, &endpoints[i], &report_options,
                                     &output, &result);
          }
        }
        amdf_info_report_endpoints_end(&output);
      }
    }
    amdf_info_cleanup(library.api, instance, count, endpoints, &output,
                      &result);
  }
  amdf_info_report_end(&output, result.complete, result.exit_code,
                       options->activate);
  if (fflush(output_file) != 0 || output.failed) {
    return AMDF_INFO_EXIT_FAILURE;
  }
  return result.exit_code;
}
