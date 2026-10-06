// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/driver.h"

#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/device.h"

typedef struct iree_hal_amd_xdna_driver_t {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Allocator owning the driver and discovery snapshot.
  iree_allocator_t host_allocator;
  // Process-static negotiated libamdf table.
  const amdf_api_t* api;
  // Process-static negotiated XDNA table.
  const amdf_xdna_api_t* xdna;
  // Instance owning native provider connections.
  amdf_instance_t* instance;
  // Borrowed ordinary storage scope owned by instance.
  amdf_memory_scope_t* system_scope;
  // Number of discovered XDNA endpoints.
  uint32_t endpoint_count;
  // Owned passive endpoint snapshot, compacted to XDNA entries.
  amdf_endpoint_summary_t* endpoints;
} iree_hal_amd_xdna_driver_t;

static const iree_hal_driver_vtable_t iree_hal_amd_xdna_driver_vtable;

static void iree_hal_amd_xdna_driver_destroy(iree_hal_driver_t* base) {
  iree_hal_amd_xdna_driver_t* driver = (iree_hal_amd_xdna_driver_t*)base;
  if (driver->instance) {
    amdf_status_t status = driver->api->instance_destroy(driver->instance);
    if (!amdf_status_is_ok(status)) {
      // Driver destruction has no device sink. Keep provider ownership intact
      // and report through the standard diagnostic sink.
      const iree_hal_amd_xdna_context_t diagnostic = {
          .event_sink = iree_hal_device_event_sink_stderr(),
      };
      iree_hal_amd_xdna_context_report(&diagnostic, status, "instance_destroy");
      return;
    }
  }
  iree_allocator_free(driver->host_allocator, driver->endpoints);
  iree_allocator_free(driver->host_allocator, driver);
}

static iree_status_t iree_hal_amd_xdna_driver_discover(
    iree_hal_amd_xdna_driver_t* driver) {
  uint32_t count = 0;
  const amdf_status_t count_status =
      driver->api->instance_enumerate_memory_scopes(driver->instance, 0, NULL,
                                                    &count);
  if (count_status != amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        count_status, "instance_enumerate_memory_scopes(count)"));
  }
  iree_host_size_t size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size, IREE_STRUCT_FIELD_FAM(count, amdf_memory_scope_t*)));
  amdf_memory_scope_t** scopes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(driver->host_allocator, size, (void**)&scopes));
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      driver->api->instance_enumerate_memory_scopes(driver->instance, count,
                                                    scopes, &count),
      "instance_enumerate_memory_scopes");
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        driver->api->memory_scope_query_info(scopes[i], &info),
        "memory_scope_query_info");
    if (iree_status_is_ok(status) &&
        info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      driver->system_scope = scopes[i];
      break;
    }
  }
  iree_allocator_free(driver->host_allocator, scopes);
  IREE_RETURN_IF_ERROR(status);
  if (!driver->system_scope) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no system memory scope");
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      driver->api->endpoint_enumerate(driver->instance, 0, NULL, &count),
      "endpoint_enumerate(count)"));
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size, IREE_STRUCT_FIELD_FAM(count, amdf_endpoint_summary_t)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(driver->host_allocator, size,
                                             (void**)&driver->endpoints));
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      driver->api->endpoint_enumerate(driver->instance, count,
                                      driver->endpoints, &count),
      "endpoint_enumerate"));
  for (uint32_t i = 0; i < count; ++i) {
    if (driver->endpoints[i].engine_kind == AMDF_ENGINE_KIND_XDNA) {
      driver->endpoints[driver->endpoint_count++] = driver->endpoints[i];
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_driver_create(iree_allocator_t host_allocator,
                                              iree_hal_driver_t** out_driver) {
  *out_driver = NULL;
  const amdf_api_t* api = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
      "query_api"));
  const void* extension = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->query_extension(AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                           AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension),
      "query_extension(XDNA)"));
  iree_hal_amd_xdna_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*driver), (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_amd_xdna_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  driver->api = api;
  driver->xdna = extension;
  const amdf_instance_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .structure_size = sizeof(create_info),
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->instance_create(&create_info, &driver->instance), "instance_create");
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_driver_discover(driver);
  }
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_driver_query_available_devices(
    iree_hal_driver_t* base, iree_allocator_t host_allocator,
    iree_host_size_t* out_count, iree_hal_device_info_t** out_infos) {
  iree_hal_amd_xdna_driver_t* driver = (iree_hal_amd_xdna_driver_t*)base;
  const iree_host_size_t string_capacity =
      sizeof(driver->endpoints[0].name) + 12;
  iree_host_size_t size = 0, string_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD_FAM(driver->endpoint_count, iree_hal_device_info_t),
      IREE_STRUCT_FIELD(driver->endpoint_count * string_capacity, char,
                        &string_offset)));
  iree_hal_device_info_t* infos = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&infos));
  char* data = (char*)infos + string_offset;
  for (uint32_t i = 0; i < driver->endpoint_count; ++i) {
    const int path_length = iree_snprintf(data, 12, "%u", i);
    infos[i].device_id = i + 1;
    infos[i].path = iree_make_string_view(data, path_length);
    data += 12;
    iree_string_view_append_to_buffer(
        iree_make_cstring_view(driver->endpoints[i].name), &infos[i].name,
        data);
    data += sizeof(driver->endpoints[i].name);
  }
  *out_count = driver->endpoint_count;
  *out_infos = infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_driver_dump_device_info(
    iree_hal_driver_t* base, iree_hal_device_id_t id,
    iree_string_builder_t* builder) {
  iree_hal_amd_xdna_driver_t* driver = (iree_hal_amd_xdna_driver_t*)base;
  const iree_host_size_t ordinal = id ? id - 1 : 0;
  if (ordinal >= driver->endpoint_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND);
  }
  return iree_string_builder_append_format(builder, "XDNA endpoint: %s\n",
                                           driver->endpoints[ordinal].name);
}

static iree_status_t iree_hal_amd_xdna_driver_create_device_by_id(
    iree_hal_driver_t* base, iree_hal_device_id_t id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amd_xdna_driver_t* driver = (iree_hal_amd_xdna_driver_t*)base;
  const iree_host_size_t ordinal = id ? id - 1 : 0;
  if (ordinal >= driver->endpoint_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "XDNA endpoint is unavailable");
  }
  uint32_t columns = 1;
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    if (!iree_string_view_equal(params[i].key, IREE_SV("columns")) ||
        !iree_string_view_atoi_uint32(params[i].value, &columns)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA accepts only the integer device parameter 'columns'");
    }
  }
  iree_hal_amd_xdna_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_context_create(
      base, driver->api, driver->xdna, driver->instance, driver->system_scope,
      &driver->endpoints[ordinal].id, columns, create_params->event_sink,
      host_allocator, &context));
  iree_status_t status = iree_hal_amd_xdna_device_create(
      context, iree_make_cstring_view(driver->endpoints[ordinal].name),
      create_params, host_allocator, out_device);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_context_destroy(context);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_driver_create_device_by_path(
    iree_hal_driver_t* driver, iree_string_view_t driver_name,
    iree_string_view_t path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  uint32_t ordinal = 0;
  if (!iree_string_view_is_empty(path) &&
      !iree_string_view_atoi_uint32(path, &ordinal)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "XDNA device path must be a discovery ordinal");
  }
  return iree_hal_amd_xdna_driver_create_device_by_id(
      driver, (iree_hal_device_id_t)ordinal + 1, param_count, params,
      create_params, host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_amd_xdna_driver_vtable = {
    .destroy = iree_hal_amd_xdna_driver_destroy,
    .query_available_devices = iree_hal_amd_xdna_driver_query_available_devices,
    .dump_device_info = iree_hal_amd_xdna_driver_dump_device_info,
    .create_device_by_id = iree_hal_amd_xdna_driver_create_device_by_id,
    .create_device_by_path = iree_hal_amd_xdna_driver_create_device_by_path,
};
