// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "util/device_cache.h"

#include "amdf/xdna.h"
#include "util/provider.h"

amdf_status_t CtsDeviceCache::GetInstance(amdf_instance_t** out_instance) {
  if (!initialized_) {
    initialized_ = true;
    initialization_status_ = amdf_cts_provider_query_api()(
        AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api_);
    if (amdf_status_is_ok(initialization_status_)) {
      amdf_instance_create_info_t create_info = {};
      create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      create_info.structure_size = sizeof(create_info);
      create_info.native_lifetime = native_lifetime_;
      initialization_status_ = api_->instance_create(&create_info, &instance_);
    }
  }
  if (amdf_status_is_ok(initialization_status_)) {
    *out_instance = instance_;
  }
  return initialization_status_;
}

amdf_status_t CtsDeviceCache::OpenEndpoint(const amdf_endpoint_id_t& id,
                                           amdf_endpoint_t** out_endpoint) {
  amdf_instance_t* instance = nullptr;
  const amdf_status_t status = GetInstance(&instance);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (const Endpoint& endpoint : endpoints_) {
    if (!amdf_endpoint_id_is_equal(&endpoint.id, &id)) {
      continue;
    }
    if (amdf_status_is_ok(endpoint.status)) {
      *out_endpoint = endpoint.handle;
    }
    return endpoint.status;
  }
  endpoints_.push_back({id});
  Endpoint& endpoint = endpoints_.back();
  endpoint.status = api_->endpoint_open(instance, &id, &endpoint.handle);
  if (amdf_status_is_ok(endpoint.status)) {
    *out_endpoint = endpoint.handle;
  }
  return endpoint.status;
}

amdf_status_t CtsDeviceCache::ResolveGpuEndpoint(
    const amdf_endpoint_native_identity_t& native_identity) {
  amdf_instance_t* instance = nullptr;
  amdf_status_t status = GetInstance(&instance);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t count = 0;
  status = api_->endpoint_enumerate(instance, 0, nullptr, &count);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  std::vector<amdf_endpoint_summary_t> summaries(count);
  if (count != 0) {
    status =
        api_->endpoint_enumerate(instance, count, summaries.data(), &count);
  }
  std::optional<amdf_endpoint_id_t> selected;
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < count && !selected;
       ++i) {
    if (summaries[i].engine_kind != AMDF_ENGINE_KIND_GPU) {
      continue;
    }
    amdf_endpoint_t* endpoint = nullptr;
    status = OpenEndpoint(summaries[i].id, &endpoint);
    amdf_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    if (amdf_status_is_ok(status)) {
      status = api_->endpoint_query_info(endpoint, &info);
    }
    if (!amdf_status_is_ok(status) ||
        info.native_identity.type != native_identity.type) {
      continue;
    }
    const auto& actual = info.native_identity.value;
    const auto& requested = native_identity.value;
    if ((native_identity.type ==
             AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE &&
         actual.linux_device.major == requested.linux_device.major &&
         actual.linux_device.minor == requested.linux_device.minor) ||
        (native_identity.type ==
             AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER &&
         actual.windows_adapter.luid == requested.windows_adapter.luid &&
         actual.windows_adapter.physical_adapter_index ==
             requested.windows_adapter.physical_adapter_index)) {
      selected = info.id;
    }
  }
  if (amdf_status_is_ok(status)) {
    if (!selected.has_value()) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
    } else if (!IsGpuEndpointSelected(*selected)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    } else {
      gpu_endpoint_id_ = *selected;
    }
  }
  return status;
}

amdf_status_t CtsDeviceCache::GetDevice(amdf_endpoint_t* endpoint,
                                        amdf_engine_kind_t engine_kind,
                                        amdf_device_t** out_device) {
  for (const Device& device : devices_) {
    if (device.endpoint != endpoint || device.engine_kind != engine_kind) {
      continue;
    }
    if (amdf_status_is_ok(device.status)) {
      *out_device = device.handle;
    }
    return device.status;
  }
  devices_.push_back({endpoint, engine_kind});
  Device& device = devices_.back();
  const void* extension = nullptr;
  if (engine_kind == AMDF_ENGINE_KIND_GPU) {
    device.status =
        api_->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                              AMDF_GPU_EXTENSION_VERSION_LATEST, &extension);
    if (amdf_status_is_ok(device.status)) {
      const auto* gpu_api = static_cast<const amdf_gpu_api_t*>(extension);
      amdf_gpu_device_create_info_t create_info = {};
      create_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO;
      create_info.structure_size = sizeof(create_info);
      device.status =
          gpu_api->device_create(endpoint, &create_info, &device.handle);
    }
  } else {
    device.status = api_->query_extension(
        AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
        AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension);
    if (amdf_status_is_ok(device.status)) {
      const auto* xdna_api = static_cast<const amdf_xdna_api_t*>(extension);
      amdf_xdna_device_create_info_t create_info = {};
      create_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO;
      create_info.structure_size = sizeof(create_info);
      device.status =
          xdna_api->device_create(endpoint, &create_info, &device.handle);
    }
  }
  if (amdf_status_is_ok(device.status)) {
    *out_device = device.handle;
  }
  return device.status;
}

amdf_status_t CtsDeviceCache::GetGpuDevice(amdf_endpoint_t* endpoint,
                                           amdf_device_t** out_device) {
  return GetDevice(endpoint, AMDF_ENGINE_KIND_GPU, out_device);
}

amdf_status_t CtsDeviceCache::GetXdnaDevice(amdf_endpoint_t* endpoint,
                                            amdf_device_t** out_device) {
  return GetDevice(endpoint, AMDF_ENGINE_KIND_XDNA, out_device);
}

amdf_status_t CtsDeviceCache::Deinitialize() {
  for (auto device = devices_.rbegin(); device != devices_.rend(); ++device) {
    if (device->handle == nullptr) {
      continue;
    }
    const amdf_status_t status = api_->device_destroy(device->handle);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    device->handle = nullptr;
  }
  for (auto endpoint = endpoints_.rbegin(); endpoint != endpoints_.rend();
       ++endpoint) {
    if (endpoint->handle == nullptr) {
      continue;
    }
    const amdf_status_t status = api_->endpoint_close(endpoint->handle);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    endpoint->handle = nullptr;
  }
  if (instance_ != nullptr) {
    const amdf_status_t status = api_->instance_destroy(instance_);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    instance_ = nullptr;
  }
  return AMDF_STATUS_OK;
}

CtsDeviceCache& GetCtsDeviceCache() {
  static CtsDeviceCache cache;
  return cache;
}
