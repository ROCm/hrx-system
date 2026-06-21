// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_HAL_INTEGRATION_UTIL_H_
#define EXPERIMENTAL_ID4_STAGES_HAL_INTEGRATION_UTIL_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

namespace id4::test {

template <typename T, void (*Release)(T*)>
class OwningRef {
 public:
  OwningRef() = default;
  OwningRef(const OwningRef&) = delete;
  OwningRef& operator=(const OwningRef&) = delete;

  ~OwningRef() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  // Owned reference released by this wrapper.
  T* value_ = nullptr;
};

using FrontierTrackerRef = OwningRef<iree_async_frontier_tracker_t,
                                     iree_async_frontier_tracker_release>;
using HalDeviceRef = OwningRef<iree_hal_device_t, iree_hal_device_release>;
using HalDeviceGroupRef =
    OwningRef<iree_hal_device_group_t, iree_hal_device_group_release>;
using ProactorPoolRef =
    OwningRef<iree_async_proactor_pool_t, iree_async_proactor_pool_release>;

typedef struct LiveHalDevice {
  // Proactor pool used by the live HAL device.
  ProactorPoolRef proactor_pool;
  // Frontier tracker retained by the HAL device group.
  FrontierTrackerRef frontier_tracker;
  // Device group passed through the ID4 stage API.
  HalDeviceGroupRef device_group;
  // HAL device selected from the requested device URI.
  HalDeviceRef device;
} LiveHalDevice;

typedef struct StageDiagnostics {
  // Number of diagnostic events observed.
  iree_host_size_t event_count;
  // Number of kernel diagnostic events observed.
  iree_host_size_t kernel_event_count;
} StageDiagnostics;

// Returns a diagnostics sink that counts lifecycle and kernel events.
id4_pipeline_diagnostics_sink_t DiagnosticsSink(StageDiagnostics* diagnostics);

// Creates a live HAL device and device group for integration tests.
iree_status_t CreateLiveHalDevice(iree_string_view_t device_uri,
                                  LiveHalDevice* out_device);

// Creates a parameter provider for |scope| from parsed --parameters flags.
iree_status_t CreateParameterProviderFromFlags(
    iree_string_view_t scope, iree_io_parameter_provider_t** out_provider);

}  // namespace id4::test

#endif  // EXPERIMENTAL_ID4_STAGES_HAL_INTEGRATION_UTIL_H_
