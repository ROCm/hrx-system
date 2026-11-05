// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only implementation of the public library dependency. The executable
// uses the shipping main and invocation, including their parser and cleanup.
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "amdf/xdna.h"

struct amdf_instance_t {
  // Whether the invocation owns the instance.
  bool live;
};
struct amdf_endpoint_t {
  // Native enumeration index, deliberately distinct from sorted identity order.
  uint32_t index;
  // Whether endpoint_open has published this handle.
  bool live;
  // Whether a created device still borrows the endpoint.
  bool device_live;
};
struct amdf_device_t {
  // Endpoint borrowed until successful device destruction.
  amdf_endpoint_t* endpoint;
};
struct amdf_memory_scope_t {
  // Instance owner of the borrowed system scope.
  amdf_instance_t* instance;
};

static const char* scenario;
static amdf_instance_t instance;
static amdf_endpoint_t endpoints[] = {{.index = 0}, {.index = 1}, {.index = 2}};
static amdf_device_t devices[3];
static amdf_memory_scope_t scope = {.instance = &instance};
static const amdf_endpoint_id_t identities[] = {{{2, 9}}, {{1, 7}}, {{3, 5}}};

static bool is_scenario(const char* name) {
  return strcmp(scenario, name) == 0;
}

static void require(bool condition) {
  if (!condition) {
    fputs("invalid public API lifecycle\n", stderr);
    abort();
  }
}

static void trace(const char* operation, const amdf_endpoint_t* endpoint) {
  if (endpoint) {
    fprintf(stderr, "%s:%u\n", operation, endpoint->index);
  } else {
    fprintf(stderr, "%s\n", operation);
  }
}

static amdf_status_t failure(void) {
  return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
}

static amdf_status_t AMDF_CALL instance_create(
    const amdf_instance_create_info_t* info, amdf_instance_t** out_instance) {
  require(info->type == AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  trace("instance_create", NULL);
  if (is_scenario("instance_failure")) {
    return failure();
  }
  instance.live = true;
  *out_instance = &instance;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL instance_destroy(amdf_instance_t* owner) {
  require(owner == &instance && owner->live);
  for (size_t i = 0; i < 3; ++i) {
    require(!endpoints[i].live);
  }
  trace("instance_destroy", NULL);
  if (is_scenario("instance_destroy_failure")) {
    return failure();
  }
  owner->live = false;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL
endpoint_enumerate(amdf_instance_t* owner, uint32_t capacity,
                   amdf_endpoint_summary_t* summaries, uint32_t* out_count) {
  require(owner->live);
  trace(capacity ? "enumerate_fill" : "enumerate_count", NULL);
  if (is_scenario("enumeration_failure")) {
    return failure();
  }
  *out_count = is_scenario("empty") ? 0 : 3;
  if (!capacity) {
    return AMDF_STATUS_OK;
  }
  require(capacity == 3);
  if (is_scenario("enumeration_growth")) {
    *out_count = 4;
    return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    summaries[i] = (amdf_endpoint_summary_t){
        .id = identities[i],
        .engine_kind = i < 2 ? AMDF_ENGINE_KIND_GPU : AMDF_ENGINE_KIND_XDNA,
        .type_flags =
            AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY | UINT32_C(0x80000000),
    };
    snprintf(summaries[i].name, sizeof(summaries[i].name),
             "device %u \"\\\n\033[31m \xcf\x80 \xc2\x9b", i);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL endpoint_open(amdf_instance_t* owner,
                                             const amdf_endpoint_id_t* id,
                                             amdf_endpoint_t** out_endpoint) {
  require(owner->live);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!amdf_endpoint_id_is_equal(id, &identities[i])) {
      continue;
    }
    trace("endpoint_open", &endpoints[i]);
    if (i == 0 && is_scenario("endpoint_denied")) {
      return amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
    }
    require(!endpoints[i].live);
    endpoints[i].live = true;
    *out_endpoint = &endpoints[i];
    return AMDF_STATUS_OK;
  }
  abort();
}

static amdf_status_t AMDF_CALL endpoint_close(amdf_endpoint_t* endpoint) {
  require(endpoint->live && !endpoint->device_live);
  trace("endpoint_close", endpoint);
  if (endpoint->index == 0 && is_scenario("endpoint_close_failure")) {
    return failure();
  }
  endpoint->live = false;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL endpoint_query_info(amdf_endpoint_t* endpoint,
                                                   amdf_endpoint_info_t* info) {
  require(endpoint->live);
  if (info->next) {
    amdf_endpoint_pci_address_t* extension = info->next;
    if (extension->type == AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS) {
      if (endpoint->index == 0 && is_scenario("location_denied")) {
        return amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                                UINT32_C(0xc0000022));
      }
      extension->available = endpoint->index != 1;
      extension->domain = 2;
      extension->bus = 0xab;
      extension->device = 3;
      extension->function = 1;
    } else {
      require(extension->type == AMDF_STRUCTURE_TYPE_ENDPOINT_HOST_LOCALITY);
      amdf_endpoint_host_locality_t* locality = info->next;
      locality->numa_node = endpoint->index == 0 ? 4 : UINT32_MAX;
    }
  }
  info->id = identities[endpoint->index];
  info->engine_kind =
      endpoint->index < 2 ? AMDF_ENGINE_KIND_GPU : AMDF_ENGINE_KIND_XDNA;
  info->queue_family_count = 1;
  info->pci.vendor_id = 0x1002;
  info->pci.device_id = 0x1234;
  if (endpoint->index == 1) {
    info->native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER;
    info->native_identity.value.windows_adapter.luid =
        UINT64_C(0xffffffffabcdef01);
    info->native_identity.value.windows_adapter.physical_adapter_index = 7;
  } else {
    info->native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE;
    info->native_identity.value.linux_device.major =
        endpoint->index == 0 ? 226 : 261;
    info->native_identity.value.linux_device.minor =
        endpoint->index == 0 ? 130 : 0;
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL gpu_endpoint_query_info(
    amdf_endpoint_t* endpoint, amdf_gpu_endpoint_info_t* info) {
  require(endpoint->live && endpoint->index < 2);
  if (endpoint->index == 0 && is_scenario("target_denied")) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EACCES);
  }
  info->gfx_ip.major = 11;
  info->gfx_ip.minor = 5;
  info->gfx_ip.stepping = 1;
  info->unique_id = UINT64_MAX - endpoint->index;
  info->compute.compute_unit_count = 40;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL xdna_endpoint_query_info(
    amdf_endpoint_t* endpoint, amdf_xdna_endpoint_info_t* info) {
  require(endpoint->live && endpoint->index == 2);
  info->architecture = AMDF_XDNA_ARCHITECTURE_AIE2P;
  snprintf(info->target_id, sizeof(info->target_id), "amd.xdna.test");
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL
queue_query_info(amdf_endpoint_t* endpoint, uint32_t ordinal,
                 amdf_queue_family_info_t* info) {
  require(endpoint->live && ordinal == 0);
  if (endpoint->index == 0 && is_scenario("queue_failure")) {
    return failure();
  }
  if (endpoint->index == 0 && is_scenario("unknown_domain")) {
    return amdf_make_status(123, UINT32_MAX);
  }
  info->ordinal = ordinal;
  info->command_type = endpoint->index < 2 ? AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                                           : AMDF_QUEUE_COMMAND_TYPE_XDNA;
  info->format_version = 1;
  info->format_features = UINT64_C(0xfedcba9876543210);
  info->maximum_ring_byte_length = UINT64_MAX;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL enumerate_scopes(amdf_instance_t* owner,
                                                uint32_t capacity,
                                                amdf_memory_scope_t** scopes,
                                                uint32_t* out_count) {
  require(owner->live);
  *out_count = 1;
  if (!capacity) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  }
  scopes[0] = &scope;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL scope_query_info(
    amdf_memory_scope_t* memory_scope, amdf_memory_scope_info_t* info) {
  require(memory_scope->instance->live);
  info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
  info->capacity_byte_length = UINT64_C(9007199254740993);
  info->memory_profile_count = 1;
  info->flags = AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED;
  return AMDF_STATUS_OK;
}

static amdf_status_t create_device(amdf_endpoint_t* endpoint,
                                   amdf_device_t** out_device) {
  require(endpoint->live && !endpoint->device_live);
  trace("device_create", endpoint);
  if (is_scenario("device_create_failure")) {
    return failure();
  }
  endpoint->device_live = true;
  devices[endpoint->index].endpoint = endpoint;
  *out_device = &devices[endpoint->index];
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL gpu_device_create(
    amdf_endpoint_t* endpoint, const amdf_gpu_device_create_info_t* info,
    amdf_device_t** out_device) {
  require(endpoint->index < 2 &&
          info->type == AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  return create_device(endpoint, out_device);
}

static amdf_status_t AMDF_CALL xdna_device_create(
    amdf_endpoint_t* endpoint, const amdf_xdna_device_create_info_t* info,
    amdf_device_t** out_device) {
  require(endpoint->index == 2 &&
          info->type == AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO);
  return create_device(endpoint, out_device);
}

static amdf_status_t AMDF_CALL
gpu_device_query_info(amdf_device_t* device, amdf_gpu_device_info_t* info) {
  require(device->endpoint->device_live);
  trace("device_query_info", device->endpoint);
  if (is_scenario("device_query_failure")) {
    return failure();
  }
  info->id.words[0] = 9;
  info->reset_epoch = UINT64_MAX;
  info->features = AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL
xdna_device_query_info(amdf_device_t* device, amdf_xdna_device_info_t* info) {
  require(device->endpoint->device_live);
  trace("device_query_info", device->endpoint);
  info->id.words[0] = 10;
  info->array.column_count = 4;
  info->array.row_count = 6;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL
placement_query_info(amdf_device_t* device, uint32_t ordinal,
                     amdf_queue_family_placement_info_t* info) {
  require(device->endpoint->device_live && ordinal == 0);
  trace("placement_query_info", device->endpoint);
  if (is_scenario("placement_failure")) {
    return failure();
  }
  info->ordinal = ordinal;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL device_destroy(amdf_device_t* device) {
  require(device->endpoint->device_live);
  trace("device_destroy", device->endpoint);
  if (is_scenario("device_destroy_failure")) {
    return failure();
  }
  device->endpoint->device_live = false;
  return AMDF_STATUS_OK;
}

static const amdf_gpu_api_t gpu_api = {
    .structure_size = sizeof(gpu_api),
    .extension_version = AMDF_GPU_EXTENSION_VERSION_LATEST,
    .endpoint_query_info = gpu_endpoint_query_info,
    .device_create = gpu_device_create,
    .device_query_info = gpu_device_query_info,
};
static const amdf_xdna_api_t xdna_api = {
    .structure_size = sizeof(xdna_api),
    .extension_version = AMDF_XDNA_EXTENSION_VERSION_LATEST,
    .endpoint_query_info = xdna_endpoint_query_info,
    .device_create = xdna_device_create,
    .device_query_info = xdna_device_query_info,
};

static amdf_status_t AMDF_CALL query_extension(amdf_extension_id_t extension_id,
                                               uint32_t minimum_version,
                                               uint32_t maximum_version,
                                               const void** out_api) {
  require(minimum_version <= maximum_version);
  if (is_scenario("extension_failure")) {
    return failure();
  }
  if (extension_id == AMDF_EXTENSION_GPU) {
    if (is_scenario("no_gpu_extension")) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    *out_api = &gpu_api;
  } else {
    require(extension_id == AMDF_EXTENSION_XDNA);
    *out_api = &xdna_api;
  }
  return AMDF_STATUS_OK;
}

static const amdf_api_t api = {
    .structure_size = sizeof(api),
    .abi_version = AMDF_ABI_VERSION_LATEST,
    .instance_create = instance_create,
    .instance_destroy = instance_destroy,
    .endpoint_enumerate = endpoint_enumerate,
    .endpoint_open = endpoint_open,
    .endpoint_query_info = endpoint_query_info,
    .endpoint_close = endpoint_close,
    .query_extension = query_extension,
    .endpoint_query_queue_family_info = queue_query_info,
    .device_query_queue_family_placement_info = placement_query_info,
    .device_destroy = device_destroy,
    .instance_enumerate_memory_scopes = enumerate_scopes,
    .memory_scope_query_info = scope_query_info,
};

amdf_status_t AMDF_CALL amdf_query_api(amdf_abi_version_t minimum_version,
                                       amdf_abi_version_t maximum_version,
                                       const amdf_api_t** out_api) {
  scenario = getenv("AMDF_INFO_TEST_SCENARIO");
  if (!scenario) {
    scenario = "normal";
  }
  trace("query_api", NULL);
  require(minimum_version <= maximum_version);
  if (is_scenario("abi_failure")) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  *out_api = &api;
  return AMDF_STATUS_OK;
}
