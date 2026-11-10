// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "util/device_cache.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>

#include "amdf/rocjitsu.h"
#include "amdf/xdna.h"
#include "libamdf/cts/util/native_identity.h"
#include "util/provider.h"

amdf_status_t CtsDeviceCache::GetInstance(amdf_instance_t** out_instance) {
  if (!initialized_) {
    initialized_ = true;
    initialization_status_ = amdf_cts_provider_query_api()(
        AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api_);
    if (amdf_status_is_ok(initialization_status_)) {
      amdf_instance_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
          .structure_size = sizeof(create_info),
          .native_lifetime = native_lifetime_,
      };
      initialization_status_ = api_->instance_create(&create_info, &instance_);
    }
    if (amdf_status_is_ok(initialization_status_) &&
        has_rocjitsu_configurations()) {
      initialization_status_ = ConfigureRocjitsu();
    }
  }
  if (amdf_status_is_ok(initialization_status_)) {
    *out_instance = instance_;
  }
  return initialization_status_;
}

amdf_status_t CtsDeviceCache::ConfigureRocjitsu() {
  const void* extension = nullptr;
  amdf_status_t status = api_->query_extension(
      AMDF_EXTENSION_ROCJITSU, AMDF_ROCJITSU_EXTENSION_VERSION_1,
      AMDF_ROCJITSU_EXTENSION_VERSION_LATEST, &extension);
  if (!amdf_status_is_ok(status)) {
    std::fprintf(stderr,
                 "cannot query rocjitsu API: status domain=%u code=%u\n",
                 amdf_status_domain(status), amdf_status_code(status));
    return status;
  }
  const auto* rocjitsu = static_cast<const amdf_rocjitsu_api_t*>(extension);
  for (size_t i = 0;
       amdf_status_is_ok(status) && i < rocjitsu_configurations_.size(); ++i) {
    const std::string& path = rocjitsu_configurations_[i];
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      std::fprintf(stderr, "cannot read rocjitsu configuration: %s\n",
                   path.c_str());
      status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
      continue;
    }
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    if (input.bad()) {
      std::fprintf(stderr, "failed reading rocjitsu configuration: %s\n",
                   path.c_str());
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      continue;
    }
    const amdf_rocjitsu_machine_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_ROCJITSU_MACHINE_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .library_path = rocjitsu_library_.c_str(),
        .configuration_json = json.c_str(),
    };
    status = rocjitsu->instance_add_machine(instance_, &create_info);
    if (!amdf_status_is_ok(status)) {
      std::fprintf(stderr,
                   "cannot configure rocjitsu machine %zu from %s with %s: "
                   "status domain=%u code=%u\n",
                   i, path.c_str(), rocjitsu_library_.c_str(),
                   amdf_status_domain(status), amdf_status_code(status));
    }
  }
  return status;
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
    amdf_endpoint_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
        .structure_size = sizeof(info),
    };
    if (amdf_status_is_ok(status)) {
      status = api_->endpoint_query_info(endpoint, &info);
    }
    if (!amdf_status_is_ok(status)) {
      continue;
    }
    if (amdf::cts::NativeIdentityIsEqual(info.native_identity,
                                         native_identity)) {
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

amdf_status_t CtsDeviceCache::ResolveRocjitsuGpuSelection() {
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
  status = api_->endpoint_enumerate(instance, count, summaries.data(), &count);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const void* extension = nullptr;
  status =
      api_->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                            AMDF_GPU_EXTENSION_VERSION_LATEST, &extension);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
  std::optional<amdf_endpoint_id_t> primary;
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < count && !primary;
       ++i) {
    const auto& summary = summaries[i];
    if (summary.engine_kind != AMDF_ENGINE_KIND_GPU ||
        !IsGpuEndpointSelected(summary.id)) {
      continue;
    }
    amdf_endpoint_t* endpoint = nullptr;
    status = OpenEndpoint(summary.id, &endpoint);
    amdf_gpu_endpoint_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
        .structure_size = sizeof(info),
    };
    if (amdf_status_is_ok(status)) {
      status = gpu->endpoint_query_info(endpoint, &info);
    }
    if (!amdf_status_is_ok(status)) {
      continue;
    }
    char target[32];
    std::snprintf(target, sizeof(target), "gfx%u%x%x", info.gfx_ip.major,
                  info.gfx_ip.minor, info.gfx_ip.stepping);
    if (gpu_targets_.empty() ||
        std::find(gpu_targets_.begin(), gpu_targets_.end(), target) !=
            gpu_targets_.end()) {
      primary = summary.id;
    }
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (!primary) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  gpu_endpoint_id_ = *primary;
  if (!gpu_peer_endpoint_id_) {
    for (const auto& summary : summaries) {
      if (summary.engine_kind == AMDF_ENGINE_KIND_GPU &&
          !amdf_endpoint_id_is_equal(&summary.id, &*primary)) {
        gpu_peer_endpoint_id_ = summary.id;
        break;
      }
    }
  }
  return AMDF_STATUS_OK;
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
      amdf_gpu_device_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
          .structure_size = sizeof(create_info),
      };
      device.status =
          gpu_api->device_create(endpoint, &create_info, &device.handle);
    }
  } else {
    device.status = api_->query_extension(
        AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
        AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension);
    if (amdf_status_is_ok(device.status)) {
      const auto* xdna_api = static_cast<const amdf_xdna_api_t*>(extension);
      amdf_xdna_device_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
          .structure_size = sizeof(create_info),
      };
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
