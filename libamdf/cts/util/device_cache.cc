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
        AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api_);
    if (amdf_status_is_ok(initialization_status_)) {
      amdf_instance_create_info_t create_info = {};
      create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      create_info.structure_size = sizeof(create_info);
      create_info.native_lifetime = native_lifetime_;
      initialization_status_ = api_->instance_create(&create_info, &instance_);
    }
  }
  if (amdf_status_is_ok(initialization_status_)) *out_instance = instance_;
  return initialization_status_;
}

amdf_status_t CtsDeviceCache::OpenEndpoint(const amdf_endpoint_id_t& id,
                                           amdf_endpoint_t** out_endpoint) {
  amdf_instance_t* instance = nullptr;
  const amdf_status_t status = GetInstance(&instance);
  if (!amdf_status_is_ok(status)) return status;
  for (const Endpoint& endpoint : endpoints_) {
    if (!amdf_endpoint_id_is_equal(&endpoint.id, &id)) continue;
    if (amdf_status_is_ok(endpoint.status)) *out_endpoint = endpoint.handle;
    return endpoint.status;
  }
  endpoints_.push_back({id});
  Endpoint& endpoint = endpoints_.back();
  endpoint.status = api_->endpoint_open(instance, &id, &endpoint.handle);
  if (amdf_status_is_ok(endpoint.status)) *out_endpoint = endpoint.handle;
  return endpoint.status;
}

amdf_status_t CtsDeviceCache::GetDevice(amdf_endpoint_t* endpoint,
                                        amdf_engine_kind_t engine_kind,
                                        amdf_device_t** out_device) {
  for (const Device& device : devices_) {
    if (device.endpoint != endpoint || device.engine_kind != engine_kind)
      continue;
    if (amdf_status_is_ok(device.status)) *out_device = device.handle;
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
  if (amdf_status_is_ok(device.status)) *out_device = device.handle;
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
    if (device->handle == nullptr) continue;
    const amdf_status_t status = api_->device_destroy(device->handle);
    if (!amdf_status_is_ok(status)) return status;
    device->handle = nullptr;
  }
  for (auto endpoint = endpoints_.rbegin(); endpoint != endpoints_.rend();
       ++endpoint) {
    if (endpoint->handle == nullptr) continue;
    const amdf_status_t status = api_->endpoint_close(endpoint->handle);
    if (!amdf_status_is_ok(status)) return status;
    endpoint->handle = nullptr;
  }
  if (instance_ != nullptr) {
    const amdf_status_t status = api_->instance_destroy(instance_);
    if (!amdf_status_is_ok(status)) return status;
    instance_ = nullptr;
  }
  return AMDF_STATUS_OK;
}

CtsDeviceCache& GetCtsDeviceCache() {
  static CtsDeviceCache cache;
  return cache;
}
