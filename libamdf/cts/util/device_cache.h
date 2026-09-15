// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_UTIL_DEVICE_CACHE_H_
#define AMDF_CTS_UTIL_DEVICE_CACHE_H_

#include <vector>

#include "amdf/amdf.h"
#include "amdf/gpu.h"

// Process-lifetime native owners borrowed by the serial CTS corpus. Resources
// are created lazily so discovery-only tests do not activate devices. Failed
// creation is cached too; subsequent cases report the failure without retrying
// native activation. Workload children remain owned by individual test cases.
class CtsDeviceCache {
 public:
  // Configures the serial corpus before any instance is requested.
  void SetNativeLifetime(amdf_native_lifetime_t native_lifetime) {
    native_lifetime_ = native_lifetime;
  }
  amdf_native_lifetime_t native_lifetime() const { return native_lifetime_; }

  amdf_status_t GetInstance(amdf_instance_t** out_instance);
  amdf_status_t OpenEndpoint(const amdf_endpoint_id_t& id,
                             amdf_endpoint_t** out_endpoint);
  // Device requests borrow an endpoint opened by this cache.
  amdf_status_t GetGpuDevice(amdf_endpoint_t* endpoint,
                             amdf_device_t** out_device);
  // Device requests borrow an endpoint opened by this cache.
  amdf_status_t GetXdnaDevice(amdf_endpoint_t* endpoint,
                              amdf_device_t** out_device);

  // Releases devices before endpoints and the instance. A failure retains
  // remaining resources and requires the caller to keep the provider loaded.
  amdf_status_t Deinitialize();

 private:
  struct Endpoint {
    // Enumerated identity used to share one opened endpoint.
    amdf_endpoint_id_t id;
    // Owned endpoint, null when opening failed.
    amdf_endpoint_t* handle = nullptr;
    // Result of the single endpoint-open attempt.
    amdf_status_t status = AMDF_STATUS_OK;
  };

  struct Device {
    // Cached endpoint borrowed by this device.
    amdf_endpoint_t* endpoint;
    // Device family determining the creation API.
    amdf_engine_kind_t engine_kind;
    // Owned device, null when creation failed.
    amdf_device_t* handle = nullptr;
    // Result of the single device-creation attempt.
    amdf_status_t status = AMDF_STATUS_OK;
  };

  amdf_status_t GetDevice(amdf_endpoint_t* endpoint,
                          amdf_engine_kind_t engine_kind,
                          amdf_device_t** out_device);

  // Core table borrowed from the initialized CTS provider.
  const amdf_api_t* api_ = nullptr;
  // Native lifetime selected once by the test executable's arguments.
  amdf_native_lifetime_t native_lifetime_ = AMDF_NATIVE_LIFETIME_PROCESS;
  // Shared instance owned until all cached descendants have been released.
  amdf_instance_t* instance_ = nullptr;
  // Whether the single instance-creation attempt has occurred.
  bool initialized_ = false;
  // Result of API negotiation and instance creation.
  amdf_status_t initialization_status_ = AMDF_STATUS_OK;
  // Owned endpoints, opened only when requested by a test.
  std::vector<Endpoint> endpoints_;
  // Owned devices, keyed by endpoint and family.
  std::vector<Device> devices_;
};

// Test-only cache; main explicitly releases it before unloading the provider.
CtsDeviceCache& GetCtsDeviceCache();

#endif  // AMDF_CTS_UTIL_DEVICE_CACHE_H_
