// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/driver.h"

#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/device.h"

typedef struct iree_hal_amd_driver_t {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Allocator owning this driver.
  iree_allocator_t host_allocator;
  // Shared instance and passive discovery retained by each materialized device.
  iree_hal_amd_fabric_t* fabric;
} iree_hal_amd_driver_t;

static const iree_hal_driver_vtable_t iree_hal_amd_driver_vtable;

static void iree_hal_amd_driver_destroy(iree_hal_driver_t* base) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  iree_hal_amd_fabric_release(driver->fabric);
  iree_allocator_free(driver->host_allocator, driver);
}

iree_status_t iree_hal_amd_driver_create(amdf_native_lifetime_t native_lifetime,
                                         iree_allocator_t host_allocator,
                                         iree_hal_driver_t** out_driver) {
  *out_driver = NULL;
  const amdf_api_t* api = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
      "query_api"));
  iree_hal_amd_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*driver), (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_amd_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_status_t status = iree_hal_amd_fabric_create(
      api, native_lifetime, host_allocator, &driver->fabric);
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

// Schemes select execution implementations, independently of physical endpoint
// enumeration. A driver never advertises devices it has no engine to execute.
static iree_status_t iree_hal_amd_driver_select_engine(
    iree_string_view_t driver_name, amdf_engine_kind_t* out_engine_kind,
    amdf_gpu_architecture_t* out_architecture) {
  *out_architecture = AMDF_GPU_ARCHITECTURE_UNKNOWN;
  if (iree_string_view_is_empty(driver_name) ||
      iree_string_view_equal(driver_name, IREE_SV("xdna"))) {
    *out_engine_kind = AMDF_ENGINE_KIND_XDNA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(driver_name, IREE_SV("rdna"))) {
    *out_engine_kind = AMDF_ENGINE_KIND_GPU;
    *out_architecture = AMDF_GPU_ARCHITECTURE_RDNA;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "AMD execution scheme '%.*s' is not built in",
                          (int)driver_name.size, driver_name.data);
}

// Architecture qualification opens only a query handle. It creates no native
// device, address space, queue, or event and never activates another engine.
static iree_status_t iree_hal_amd_driver_match_endpoint(
    iree_hal_amd_fabric_t* fabric, const amdf_endpoint_summary_t* summary,
    amdf_engine_kind_t engine_kind, amdf_gpu_architecture_t architecture,
    bool* out_matches) {
  *out_matches = false;
  if (summary->engine_kind != engine_kind) {
    return iree_ok_status();
  }
  if (engine_kind != AMDF_ENGINE_KIND_GPU) {
    *out_matches = true;
    return iree_ok_status();
  }
  const void* extension = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->query_extension(
          AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
          AMDF_GPU_EXTENSION_VERSION_LATEST, &extension),
      "query_extension(GPU)"));
  const amdf_gpu_api_t* gpu = extension;
  amdf_endpoint_t* endpoint = NULL;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->endpoint_open(fabric->instance, &summary->id, &endpoint),
      "endpoint_open"));
  amdf_gpu_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      gpu->endpoint_query_info(endpoint, &info), "gpu.endpoint_query_info");
  const amdf_status_t close_status = fabric->api->endpoint_close(endpoint);
  if (!amdf_status_is_ok(close_status)) {
    // The failed close preserves its endpoint. Keep its borrowed instance live
    // along with that native ownership instead of freeing the parent beneath
    // it.
    iree_hal_amd_fabric_retain(fabric);
    status = iree_status_join(
        status, IREE_HAL_AMD_STATUS_FROM_AMDF(close_status, "endpoint_close"));
  }
  if (iree_status_is_ok(status)) {
    *out_matches = amdf_gpu_endpoint_architecture(&info) == architecture;
  }
  return status;
}

static iree_status_t iree_hal_amd_driver_query_available_devices(
    iree_hal_driver_t* base, iree_string_view_t driver_name,
    iree_allocator_t host_allocator, iree_host_size_t* out_count,
    iree_hal_device_info_t** out_infos) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  iree_hal_amd_fabric_t* fabric = driver->fabric;
  amdf_engine_kind_t engine_kind = AMDF_ENGINE_KIND_UNKNOWN;
  amdf_gpu_architecture_t architecture = AMDF_GPU_ARCHITECTURE_UNKNOWN;
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_select_engine(
      driver_name, &engine_kind, &architecture));
  if (!fabric->endpoint_count) {
    return iree_ok_status();
  }
  // An endpoint identity path is "id/" followed by two 16-digit hex words.
  const iree_host_size_t path_capacity = 36;
  const iree_host_size_t string_capacity =
      AMDF_ENDPOINT_NAME_CAPACITY + path_capacity;
  iree_host_size_t size = 0, string_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD_FAM(fabric->endpoint_count, iree_hal_device_info_t),
      IREE_STRUCT_FIELD(fabric->endpoint_count * string_capacity, char,
                        &string_offset)));
  iree_hal_device_info_t* infos = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&infos));
  char* data = (char*)infos + string_offset;
  uint32_t count = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < fabric->endpoint_count && iree_status_is_ok(status);
       ++i) {
    const amdf_endpoint_summary_t* endpoint = &fabric->endpoints[i];
    bool matches = false;
    status = iree_hal_amd_driver_match_endpoint(fabric, endpoint, engine_kind,
                                                architecture, &matches);
    if (!matches) {
      continue;
    }
    const int path_length =
        iree_snprintf(data, path_capacity, "id/%016" PRIx64 "%016" PRIx64,
                      endpoint->id.words[0], endpoint->id.words[1]);
    infos[count].device_id = (iree_hal_device_id_t)i + 1;
    infos[count].path = iree_make_string_view(data, path_length);
    data += path_capacity;
    iree_string_view_append_to_buffer(iree_make_cstring_view(endpoint->name),
                                      &infos[count].name, data);
    data += AMDF_ENDPOINT_NAME_CAPACITY;
    ++count;
  }
  if (iree_status_is_ok(status)) {
    *out_count = count;
    *out_infos = infos;
  } else {
    iree_allocator_free(host_allocator, infos);
  }
  return status;
}

static iree_status_t iree_hal_amd_driver_resolve_id(
    const iree_hal_amd_fabric_t* fabric, iree_hal_device_id_t id,
    const amdf_endpoint_summary_t** out_endpoint) {
  if (id == IREE_HAL_DEVICE_ID_DEFAULT) {
    for (uint32_t i = 0; i < fabric->endpoint_count; ++i) {
      if (fabric->endpoints[i].engine_kind == AMDF_ENGINE_KIND_XDNA) {
        *out_endpoint = &fabric->endpoints[i];
        return iree_ok_status();
      }
    }
  } else if (id - 1 < fabric->endpoint_count) {
    *out_endpoint = &fabric->endpoints[id - 1];
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "AMD endpoint is unavailable");
}

static iree_status_t iree_hal_amd_driver_dump_device_info(
    iree_hal_driver_t* base, iree_hal_device_id_t id,
    iree_string_builder_t* builder) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  const amdf_endpoint_summary_t* endpoint = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_driver_resolve_id(driver->fabric, id, &endpoint));
  return iree_string_builder_append_format(builder, "AMD endpoint: %s\n",
                                           endpoint->name);
}

static iree_status_t iree_hal_amd_driver_create_gpu_device(
    iree_hal_amd_fabric_t* fabric, const amdf_endpoint_summary_t* endpoint,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amd_gpu_device_options_t options;
  iree_hal_amd_gpu_device_options_initialize(&options);
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    if (iree_string_view_equal(params[i].key, IREE_SV("asan_report_policy")) ||
        iree_string_view_equal(params[i].key, IREE_SV("tsan_report_policy"))) {
      iree_hal_amd_gpu_sanitizer_report_policy_t policy;
      if (iree_string_view_equal(params[i].value, IREE_SV("report-only"))) {
        policy = IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_REPORT_ONLY;
      } else if (iree_string_view_equal(params[i].value,
                                        IREE_SV("fail-device"))) {
        policy = IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GPU sanitizer report policy must be "
                                "report-only or fail-device");
      }
      if (iree_string_view_equal(params[i].key,
                                 IREE_SV("asan_report_policy"))) {
        options.asan.report_policy = policy;
      } else {
        options.tsan.report_policy = policy;
      }
      continue;
    }
    uint32_t* value = NULL;
    if (iree_string_view_equal(params[i].key, IREE_SV("queue_count"))) {
      value = &options.queue_count;
    } else if (iree_string_view_equal(params[i].key, IREE_SV("queue_family"))) {
      value = &options.queue_family_ordinal;
    } else if (iree_string_view_equal(params[i].key,
                                      IREE_SV("host_numa_node"))) {
      value = &options.host_numa_node;
    } else if (iree_string_view_equal(params[i].key,
                                      IREE_SV("tsan_memory_granule_shift"))) {
      value = &options.tsan.memory.memory_granule_shift;
    } else if (iree_string_view_equal(
                   params[i].key,
                   IREE_SV("tsan_workgroup_local_memory_size"))) {
      value = &options.tsan.memory.workgroup_local_memory_size;
    } else if (iree_string_view_equal(params[i].key,
                                      IREE_SV("tsan_workgroup_capacity"))) {
      value = &options.tsan.memory.workgroup_capacity;
    } else if (iree_string_view_equal(params[i].key,
                                      IREE_SV("tsan_shadow_slot_count"))) {
      value = &options.tsan.memory.shadow_slot_count;
    }
    if (!value || !iree_string_view_atoi_uint32(params[i].value, value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown GPU device parameter or invalid unsigned integer '%.*s'",
          (int)params[i].key.size, params[i].key.data);
    }
  }
  iree_hal_amd_gpu_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_allocate(
      fabric, &endpoint->id, create_params->event_sink, host_allocator,
      &context));
  iree_status_t status = iree_hal_amd_gpu_device_create(
      context, iree_make_cstring_view(endpoint->name), &options, create_params,
      host_allocator, out_device);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_free(context);
  }
  return status;
}

static iree_status_t iree_hal_amd_driver_create_device_by_id(
    iree_hal_driver_t* base, iree_hal_device_id_t id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  const amdf_endpoint_summary_t* endpoint = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_driver_resolve_id(driver->fabric, id, &endpoint));
  if (endpoint->engine_kind == AMDF_ENGINE_KIND_GPU) {
    return iree_hal_amd_driver_create_gpu_device(
        driver->fabric, endpoint, param_count, params, create_params,
        host_allocator, out_device);
  }
  if (endpoint->engine_kind != AMDF_ENGINE_KIND_XDNA) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMD endpoint has no supported execution engine");
  }
  iree_hal_amd_xdna_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_context_allocate(
      driver->fabric, &endpoint->id, param_count, params,
      create_params->event_sink, host_allocator, &context));
  iree_status_t status = iree_hal_amd_xdna_device_create(
      context, iree_make_cstring_view(endpoint->name), create_params,
      host_allocator, out_device);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_context_free(context);
  }
  return status;
}

static bool iree_hal_amd_driver_parse_identity(iree_string_view_t path,
                                               amdf_endpoint_id_t* out_id) {
  uint8_t bytes[16];
  if (path.size != sizeof(bytes) * 2 ||
      !iree_string_view_parse_hex_bytes(path, sizeof(bytes), bytes)) {
    return false;
  }
  // The URI spells each word most-significant byte first, independently of
  // the host representation of the native identity.
  for (uint32_t i = 0; i < 2; ++i) {
    uint64_t word = 0;
    for (uint32_t j = 0; j < 8; ++j) {
      word = (word << 8) | bytes[i * 8 + j];
    }
    out_id->words[i] = word;
  }
  return true;
}

static bool iree_hal_amd_driver_parse_ordinal(iree_string_view_t path,
                                              uint32_t* out_ordinal) {
  uint32_t ordinal = 0;
  for (iree_host_size_t i = 0; i < path.size; ++i) {
    const uint32_t digit = (uint32_t)(uint8_t)path.data[i] - '0';
    if (digit > 9 || ordinal > (UINT32_MAX - digit) / 10) {
      return false;
    }
    ordinal = ordinal * 10 + digit;
  }
  *out_ordinal = ordinal;
  return true;
}

static iree_status_t iree_hal_amd_driver_create_device_by_path(
    iree_hal_driver_t* base, iree_string_view_t driver_name,
    iree_string_view_t path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  amdf_engine_kind_t engine_kind = AMDF_ENGINE_KIND_UNKNOWN;
  amdf_gpu_architecture_t architecture = AMDF_GPU_ARCHITECTURE_UNKNOWN;
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_select_engine(
      driver_name, &engine_kind, &architecture));
  const bool by_identity =
      iree_string_view_consume_prefix(&path, IREE_SV("id/"));
  uint32_t ordinal = 0;
  amdf_endpoint_id_t endpoint_id = {{0, 0}};
  if (by_identity) {
    if (!iree_hal_amd_driver_parse_identity(path, &endpoint_id)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMD identity path must be id/<32 hexadecimal digits>");
    }
  } else if (!iree_hal_amd_driver_parse_ordinal(path, &ordinal)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMD device path must be a family ordinal or id/<identity>");
  }
  uint32_t family_ordinal = 0;
  iree_hal_device_id_t device_id = IREE_HAL_DEVICE_ID_DEFAULT;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < driver->fabric->endpoint_count &&
       device_id == IREE_HAL_DEVICE_ID_DEFAULT && iree_status_is_ok(status);
       ++i) {
    const amdf_endpoint_summary_t* endpoint = &driver->fabric->endpoints[i];
    bool matches = false;
    status = iree_hal_amd_driver_match_endpoint(
        driver->fabric, endpoint, engine_kind, architecture, &matches);
    if (!matches) {
      continue;
    }
    if (by_identity ? amdf_endpoint_id_is_equal(&endpoint->id, &endpoint_id)
                    : family_ordinal == ordinal) {
      device_id = (iree_hal_device_id_t)i + 1;
    }
    ++family_ordinal;
  }
  IREE_RETURN_IF_ERROR(status);
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no AMD endpoint matches the selected family and path");
  }
  return iree_hal_amd_driver_create_device_by_id(base, device_id, param_count,
                                                 params, create_params,
                                                 host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_amd_driver_vtable = {
    .destroy = iree_hal_amd_driver_destroy,
    .query_available_devices = iree_hal_amd_driver_query_available_devices,
    .dump_device_info = iree_hal_amd_driver_dump_device_info,
    .create_device_by_id = iree_hal_amd_driver_create_device_by_id,
    .create_device_by_path = iree_hal_amd_driver_create_device_by_path,
};
