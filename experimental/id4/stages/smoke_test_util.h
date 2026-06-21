// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_SMOKE_TEST_UTIL_H_
#define EXPERIMENTAL_ID4_STAGES_SMOKE_TEST_UTIL_H_

#include <string>
#include <vector>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

namespace id4::test {

extern const char kSmokeKernelSource[];

typedef struct SmokeDiagnostics {
  // Number of diagnostic events observed.
  iree_host_size_t count;
  // Event keys observed in order.
  std::vector<std::string> keys;
  // Number of kernel diagnostic events observed.
  iree_host_size_t kernel_event_count;
} SmokeDiagnostics;

typedef struct SmokeExecutableCache {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Host allocator used for executable cache storage.
  iree_allocator_t host_allocator;
  // Number of infer-format calls observed.
  iree_host_size_t infer_count;
  // Number of can-prepare calls observed.
  iree_host_size_t can_prepare_count;
  // Number of prepare calls observed.
  iree_host_size_t prepare_count;
  // Caching mode from the latest prepare call.
  iree_hal_executable_caching_mode_t last_caching_mode;
} SmokeExecutableCache;

std::string ToString(iree_string_view_t value);

bool ContainsKey(const std::vector<std::string>& keys, const char* key);

id4_pipeline_diagnostics_sink_t DiagnosticsSink(SmokeDiagnostics* diagnostics);

iree_hal_device_group_t* CreateLocalSyncDeviceGroup();

iree_hal_semaphore_t* CreateSemaphore(iree_hal_device_t* device);

iree_status_t CreateExecutableCache(iree_allocator_t host_allocator,
                                    SmokeExecutableCache** out_cache);

iree_status_t CreateKernelCache(iree_allocator_t host_allocator,
                                id4_pipeline_kernel_cache_t** out_kernel_cache);

iree_status_t CreateSmokeStage(iree_hal_device_group_t* device_group,
                               SmokeExecutableCache* executable_cache,
                               id4_pipeline_kernel_cache_t* kernel_cache,
                               iree_allocator_t host_allocator,
                               id4_pipeline_stage_t** out_stage);

iree_io_parameter_provider_t* CreateSmokeParameterProvider();

}  // namespace id4::test

#endif  // EXPERIMENTAL_ID4_STAGES_SMOKE_TEST_UTIL_H_
