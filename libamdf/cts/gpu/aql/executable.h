// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_AQL_EXECUTABLE_H_
#define AMDF_CTS_GPU_AQL_EXECUTABLE_H_

#include "libamdf/cts/gpu/kernels/kernel.h"
#include "libamdf/cts/gpu/util/memory.h"
#include "libamdf/cts/gpu/util/user_queue.h"

namespace aql {

// Includes speculative instruction fetch and compiler-requested prefetch.
// Allocation granularity belongs to the executable storage owner.
uint64_t ExecutableByteLength(const amdf_gpu_endpoint_info_t& endpoint,
                              const kernels::Kernel& kernel);

// One device's case-owned image, cache-publication carrier and setup signal.
// The caller retains the device and removes every queue using this image
// before Release. Destruction performs no implicit native cleanup or retry.
class Executable {
 public:
  // Completes instruction-cache publication and ring consumption, then
  // publishes the descriptor address. Initialization advances the caller's
  // packet index at reservation; partially initialized storage stays owned.
  void Initialize(const amdf_api_t* api, amdf_memory_scope_t* system_scope,
                  amdf_device_t* device,
                  const amdf_gpu_endpoint_info_t& endpoint,
                  const kernels::Kernel& kernel, const char* property_prefix,
                  GpuUserQueue& queue, uint64_t* next_packet_index);
  bool Release(const amdf_api_t* api);

  uint64_t descriptor_address() const { return descriptor_address_; }

 private:
  // Executable backing retained through the final queue using its descriptor.
  GpuMemory code_;
  // Immutable native cache-publication commands.
  GpuMemory commands_;
  // Complete native signal owned independently of workload dependencies.
  GpuMemory completion_;
  // Device address published only after initialization completes.
  uint64_t descriptor_address_ = 0;
};

}  // namespace aql

#endif  // AMDF_CTS_GPU_AQL_EXECUTABLE_H_
