// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/driver.h"

#include <stdlib.h>

#include "amdf/gpu.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/status.h"
#if IREE_HAL_AMD_GPU_ENABLED
#include "iree/hal/drivers/amd/gpu/device.h"
#endif
#if IREE_HAL_AMD_XDNA_ENABLED
#include "iree/hal/drivers/amd/xdna/device.h"
#endif

typedef struct iree_hal_amd_driver_scheme_t {
  // URI scheme selecting this independent ordinal namespace.
  iree_string_view_t name;
  // Optional ordered list of unmasked scheme ordinals.
  const char* visibility_variable;
  // Native execution engine exposed by this scheme.
  amdf_engine_kind_t engine_kind;
  // GPU architecture filter, or UNKNOWN for all supported architectures.
  amdf_gpu_architecture_t architecture;
} iree_hal_amd_driver_scheme_t;

static const iree_hal_amd_driver_scheme_t iree_hal_amd_driver_schemes[] = {
#if IREE_HAL_AMD_XDNA_ENABLED
    {IREE_SVL("xdna"), "HAL_VISIBLE_XDNA_DEVICES", AMDF_ENGINE_KIND_XDNA,
     AMDF_GPU_ARCHITECTURE_UNKNOWN},
#endif
#if IREE_HAL_AMD_GPU_ENABLED
    {IREE_SVL("rdna"), "HAL_VISIBLE_RDNA_DEVICES", AMDF_ENGINE_KIND_GPU,
     AMDF_GPU_ARCHITECTURE_RDNA},
    {IREE_SVL("cdna"), "HAL_VISIBLE_CDNA_DEVICES", AMDF_ENGINE_KIND_GPU,
     AMDF_GPU_ARCHITECTURE_CDNA},
    {IREE_SVL("amdgpu"), "HAL_VISIBLE_AMDGPU_DEVICES", AMDF_ENGINE_KIND_GPU,
     AMDF_GPU_ARCHITECTURE_UNKNOWN},
#endif
};

typedef struct iree_hal_amd_driver_view_t {
  // Number of visible endpoints, with public ordinals in [0, count).
  uint32_t count;
  // Ordered native endpoint indices owned by the driver's selection allocation.
  uint32_t* indices;
} iree_hal_amd_driver_view_t;

typedef struct iree_hal_amd_driver_t {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Allocator owning this driver.
  iree_allocator_t host_allocator;
  // Shared instance and passive discovery retained by each materialized device.
  iree_hal_amd_fabric_t* fabric;
  // Unmasked scheme membership bits per native endpoint. This allocation also
  // owns all view indices; zero membership means unsupported execution.
  uint32_t* endpoint_schemes;
  // Immutable selections captured with the driver, independent for each scheme.
  iree_hal_amd_driver_view_t views[IREE_ARRAYSIZE(iree_hal_amd_driver_schemes)];
  // Native-order union of endpoints exposed by at least one selected scheme.
  iree_hal_amd_driver_view_t all_view;
} iree_hal_amd_driver_t;

static const iree_hal_driver_vtable_t iree_hal_amd_driver_vtable;
static iree_status_t iree_hal_amd_driver_initialize_views(
    iree_hal_amd_driver_t* driver);

static void iree_hal_amd_driver_destroy(iree_hal_driver_t* base) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  iree_hal_amd_fabric_release(driver->fabric);
  iree_allocator_free(driver->host_allocator, driver->endpoint_schemes);
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
    status = iree_hal_amd_driver_initialize_views(driver);
  }
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

// Schemes select execution implementations, independently of physical endpoint
// enumeration. GPU schemes require usable native compute queue families.
static iree_status_t iree_hal_amd_driver_select_view(
    const iree_hal_amd_driver_t* driver, iree_string_view_t driver_name,
    const iree_hal_amd_driver_view_t** out_view) {
  if (iree_string_view_is_empty(driver_name)) {
    *out_view = &driver->all_view;
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(driver->views); ++i) {
    if (iree_string_view_equal(driver_name,
                               iree_hal_amd_driver_schemes[i].name)) {
      *out_view = &driver->views[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "AMD execution scheme '%.*s' is not built in",
                          (int)driver_name.size, driver_name.data);
}

#if IREE_HAL_AMD_GPU_ENABLED
// Captures architecture and compute support through query-only handles. This
// creates no native device, address space, queue, or event.
static iree_status_t iree_hal_amd_driver_query_gpu_support(
    iree_hal_amd_fabric_t* fabric, const amdf_gpu_api_t* gpu,
    const amdf_endpoint_summary_t* summary,
    amdf_gpu_architecture_t* out_architecture, bool* out_supported) {
  *out_architecture = AMDF_GPU_ARCHITECTURE_UNKNOWN;
  *out_supported = false;
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
  bool supported = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_device_query_support(fabric->api, endpoint,
                                                   &supported);
  }
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
    *out_architecture = amdf_gpu_endpoint_architecture(&info);
    *out_supported = supported;
  }
  return status;
}
#endif  // IREE_HAL_AMD_GPU_ENABLED

static iree_status_t iree_hal_amd_driver_query_available_devices(
    iree_hal_driver_t* base, iree_string_view_t driver_name,
    iree_allocator_t host_allocator, iree_host_size_t* out_count,
    iree_hal_device_info_t** out_infos) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  iree_hal_amd_fabric_t* fabric = driver->fabric;
  const iree_hal_amd_driver_view_t* view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_driver_select_view(driver, driver_name, &view));
  if (!view->count) {
    return iree_ok_status();
  }
  // An endpoint identity path is "id/" followed by two 16-digit hex words.
  const iree_host_size_t path_capacity = 36;
  const iree_host_size_t string_capacity =
      AMDF_ENDPOINT_NAME_CAPACITY + path_capacity;
  iree_host_size_t size = 0, string_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size, IREE_STRUCT_FIELD_FAM(view->count, iree_hal_device_info_t),
      IREE_STRUCT_FIELD(view->count * string_capacity, char, &string_offset)));
  iree_hal_device_info_t* infos = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&infos));
  char* data = (char*)infos + string_offset;
  for (uint32_t i = 0; i < view->count; ++i) {
    const uint32_t endpoint_index = view->indices[i];
    const amdf_endpoint_summary_t* endpoint =
        &fabric->endpoints[endpoint_index];
    const int path_length =
        iree_snprintf(data, path_capacity, "id/%016" PRIx64 "%016" PRIx64,
                      endpoint->id.words[0], endpoint->id.words[1]);
    infos[i].device_id = (iree_hal_device_id_t)endpoint_index + 1;
    infos[i].path = iree_make_string_view(data, path_length);
    data += path_capacity;
    iree_string_view_append_to_buffer(iree_make_cstring_view(endpoint->name),
                                      &infos[i].name, data);
    data += AMDF_ENDPOINT_NAME_CAPACITY;
  }
  *out_count = view->count;
  *out_infos = infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_driver_resolve_id(
    const iree_hal_amd_driver_t* driver, iree_hal_device_id_t id,
    const amdf_endpoint_summary_t** out_endpoint) {
  const iree_hal_amd_fabric_t* fabric = driver->fabric;
  if (id == IREE_HAL_DEVICE_ID_DEFAULT) {
    const iree_hal_amd_driver_view_t* view = &driver->all_view;
    if (view->count) {
      *out_endpoint = &fabric->endpoints[view->indices[0]];
      return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_resolve_id(driver, id, &endpoint));
  return iree_string_builder_append_format(builder, "AMD endpoint: %s\n",
                                           endpoint->name);
}

#if IREE_HAL_AMD_GPU_ENABLED
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
    if (iree_string_view_equal(params[i].key, IREE_SV("tsan_shadow_size"))) {
      if (!iree_string_view_atoi_uint64(params[i].value,
                                        &options.tsan.memory.shadow_size)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GPU TSAN shadow size must be an unsigned "
                                "byte count");
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
                                      IREE_SV("completion_spin_ns"))) {
      value = &options.completion_spin_ns;
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
  return iree_hal_amd_gpu_device_create(
      context, iree_make_cstring_view(endpoint->name), &options, create_params,
      host_allocator, out_device);
}
#endif  // IREE_HAL_AMD_GPU_ENABLED

static iree_status_t iree_hal_amd_driver_create_device(
    iree_hal_amd_driver_t* driver, const amdf_endpoint_summary_t* endpoint,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  switch (endpoint->engine_kind) {
#if IREE_HAL_AMD_GPU_ENABLED
    case AMDF_ENGINE_KIND_GPU:
      return iree_hal_amd_driver_create_gpu_device(
          driver->fabric, endpoint, param_count, params, create_params,
          host_allocator, out_device);
#endif
#if IREE_HAL_AMD_XDNA_ENABLED
    case AMDF_ENGINE_KIND_XDNA: {
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
#endif
    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "AMD endpoint has no built-in execution engine");
  }
}

static iree_status_t iree_hal_amd_driver_create_device_by_id(
    iree_hal_driver_t* base, iree_hal_device_id_t id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  const amdf_endpoint_summary_t* endpoint = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_resolve_id(driver, id, &endpoint));
  if (!driver->endpoint_schemes[endpoint - driver->fabric->endpoints]) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMD endpoint '%s' has no supported execution "
                            "queue family",
                            endpoint->name);
  }
  return iree_hal_amd_driver_create_device(driver, endpoint, param_count,
                                           params, create_params,
                                           host_allocator, out_device);
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

// Applies one scheme's visibility without changing another scheme or the native
// snapshot. The temporary copy permits arbitrary reordering of the source list.
static iree_status_t iree_hal_amd_driver_apply_visibility(
    const char* variable, iree_allocator_t host_allocator,
    iree_hal_amd_driver_view_t* view) {
  const char* value = getenv(variable);
  if (!value) {
    return iree_ok_status();
  }
  iree_string_view_t remaining =
      iree_string_view_trim(iree_make_cstring_view(value));
  const uint32_t unmasked_count = view->count;
  view->count = 0;
  if (iree_string_view_is_empty(remaining)) {
    return iree_ok_status();
  }
  if (!unmasked_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s selects from a scheme with no supported devices", variable);
  }

  uint32_t* unmasked_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(view->indices,
                                unmasked_count * sizeof(*view->indices)),
      (void**)&unmasked_indices));
  iree_status_t status = iree_ok_status();
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t token;
    const intptr_t separator =
        iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_string_view_trim(token);
    uint32_t ordinal = 0;
    if (iree_string_view_is_empty(token) ||
        !iree_hal_amd_driver_parse_ordinal(token, &ordinal) ||
        ordinal >= unmasked_count ||
        (separator != -1 && iree_string_view_is_empty(remaining))) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%s must be a comma-separated list of ordinals in [0, %u)", variable,
          unmasked_count);
      continue;
    }
    const uint32_t endpoint_index = unmasked_indices[ordinal];
    for (uint32_t i = 0; i < view->count; ++i) {
      if (view->indices[i] == endpoint_index) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "%s repeats ordinal %u", variable, ordinal);
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      view->indices[view->count++] = endpoint_index;
    }
  }
  iree_allocator_free(host_allocator, unmasked_indices);
  return status;
}

static iree_status_t iree_hal_amd_driver_initialize_views(
    iree_hal_amd_driver_t* driver) {
  const iree_host_size_t endpoint_count = driver->fabric->endpoint_count;
  if (endpoint_count) {
    iree_host_size_t size = 0, indices_offset = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &size, IREE_STRUCT_FIELD_FAM(endpoint_count, uint32_t),
        IREE_STRUCT_FIELD(endpoint_count * (IREE_ARRAYSIZE(driver->views) + 1),
                          uint32_t, &indices_offset)));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        driver->host_allocator, size, (void**)&driver->endpoint_schemes));
    uint32_t* indices =
        (uint32_t*)((uint8_t*)driver->endpoint_schemes + indices_offset);
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(driver->views); ++i) {
      driver->views[i].indices = indices + endpoint_count * i;
    }
    driver->all_view.indices =
        indices + endpoint_count * IREE_ARRAYSIZE(driver->views);
  }

  iree_status_t status = iree_ok_status();
#if IREE_HAL_AMD_GPU_ENABLED
  const amdf_gpu_api_t* gpu = NULL;
  for (uint32_t i = 0; i < endpoint_count && !gpu && iree_status_is_ok(status);
       ++i) {
    if (driver->fabric->endpoints[i].engine_kind != AMDF_ENGINE_KIND_GPU) {
      continue;
    }
    const void* extension = NULL;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        driver->fabric->api->query_extension(
            AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
            AMDF_GPU_EXTENSION_VERSION_LATEST, &extension),
        "query_extension(GPU)");
    gpu = extension;
  }
#endif
  for (uint32_t i = 0; i < endpoint_count && iree_status_is_ok(status); ++i) {
    const amdf_endpoint_summary_t* endpoint = &driver->fabric->endpoints[i];
    amdf_gpu_architecture_t architecture = AMDF_GPU_ARCHITECTURE_UNKNOWN;
    bool supported = false;
#if IREE_HAL_AMD_GPU_ENABLED
    if (endpoint->engine_kind == AMDF_ENGINE_KIND_GPU) {
      status = iree_hal_amd_driver_query_gpu_support(
          driver->fabric, gpu, endpoint, &architecture, &supported);
    }
#endif
#if IREE_HAL_AMD_XDNA_ENABLED
    if (endpoint->engine_kind == AMDF_ENGINE_KIND_XDNA) {
      supported = true;
    }
#endif
    if (!supported) {
      continue;
    }
    for (uint32_t j = 0; j < IREE_ARRAYSIZE(driver->views); ++j) {
      const iree_hal_amd_driver_scheme_t* scheme =
          &iree_hal_amd_driver_schemes[j];
      if (endpoint->engine_kind == scheme->engine_kind &&
          (scheme->architecture == AMDF_GPU_ARCHITECTURE_UNKNOWN ||
           architecture == scheme->architecture)) {
        driver->endpoint_schemes[i] |= 1u << j;
        iree_hal_amd_driver_view_t* view = &driver->views[j];
        view->indices[view->count++] = i;
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(driver->views) && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_driver_apply_visibility(
        iree_hal_amd_driver_schemes[i].visibility_variable,
        driver->host_allocator, &driver->views[i]);
  }
  if (iree_status_is_ok(status)) {
    // Mark every endpoint retained by any independent scheme, then compact the
    // markers in native order. The output prefix never reaches an unread
    // marker.
    for (uint32_t i = 0; i < endpoint_count; ++i) {
      driver->all_view.indices[i] = 0;
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(driver->views); ++i) {
      const iree_hal_amd_driver_view_t* view = &driver->views[i];
      for (uint32_t j = 0; j < view->count; ++j) {
        driver->all_view.indices[view->indices[j]] = 1;
      }
    }
    for (uint32_t i = 0; i < endpoint_count; ++i) {
      if (driver->all_view.indices[i]) {
        driver->all_view.indices[driver->all_view.count++] = i;
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_amd_driver_create_device_by_path(
    iree_hal_driver_t* base, iree_string_view_t driver_name,
    iree_string_view_t path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amd_driver_t* driver = (iree_hal_amd_driver_t*)base;
  const iree_hal_amd_driver_view_t* view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_driver_select_view(driver, driver_name, &view));
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
  const amdf_endpoint_summary_t* selected_endpoint = NULL;
  if (by_identity) {
    for (uint32_t i = 0; i < view->count; ++i) {
      const amdf_endpoint_summary_t* endpoint =
          &driver->fabric->endpoints[view->indices[i]];
      if (amdf_endpoint_id_is_equal(&endpoint->id, &endpoint_id)) {
        selected_endpoint = endpoint;
        break;
      }
    }
    if (!selected_endpoint) {
      for (uint32_t i = 0; i < driver->fabric->endpoint_count; ++i) {
        const amdf_endpoint_summary_t* endpoint = &driver->fabric->endpoints[i];
        if (amdf_endpoint_id_is_equal(&endpoint->id, &endpoint_id)) {
          return iree_make_status(
              IREE_STATUS_UNAVAILABLE,
              "AMD endpoint '%s' is not available in scheme '%.*s'",
              endpoint->name, (int)driver_name.size, driver_name.data);
        }
      }
    }
  } else if (ordinal < view->count) {
    selected_endpoint = &driver->fabric->endpoints[view->indices[ordinal]];
  }
  if (!selected_endpoint) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no AMD endpoint matches the selected family and path");
  }
  return iree_hal_amd_driver_create_device(driver, selected_endpoint,
                                           param_count, params, create_params,
                                           host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_amd_driver_vtable = {
    .destroy = iree_hal_amd_driver_destroy,
    .query_available_devices = iree_hal_amd_driver_query_available_devices,
    .dump_device_info = iree_hal_amd_driver_dump_device_info,
    .create_device_by_id = iree_hal_amd_driver_create_device_by_id,
    .create_device_by_path = iree_hal_amd_driver_create_device_by_path,
};
