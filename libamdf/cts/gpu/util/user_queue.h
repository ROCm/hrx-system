// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_UTIL_USER_QUEUE_H_
#define AMDF_CTS_GPU_UTIL_USER_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <thread>

#include "amdf/amdf.h"
#include "amdf/gpu.h"

// These native producer witnesses require x86-64 and naturally aligned device
// memory. Lock-free acquire/release accesses preserve host-thread atomicity and
// lower to plain loads/stores. Doorbells never use locked read-modify-writes.
template <typename T>
T GpuLoadAcquire(uint64_t address) {
  static_assert(std::atomic_ref<T>::is_always_lock_free);
  return std::atomic_ref<T>(*reinterpret_cast<T*>(address))
      .load(std::memory_order_acquire);
}

template <typename T>
void GpuStoreRelease(uint64_t address, T value) {
  static_assert(std::atomic_ref<T>::is_always_lock_free);
  std::atomic_ref<T>(*reinterpret_cast<T*>(address))
      .store(value, std::memory_order_release);
}

// Waits on a coherent completion line without maintaining any payload cache.
// The command sequence defines what completion means; consumption is separate.
template <typename T>
void GpuWaitEqual(uint64_t address, T value) {
  while (GpuLoadAcquire<T>(address) != value) {
    std::this_thread::yield();
  }
}

// Case-owned native queue and its producer views. Packet encoding, index
// units, completion and cache operations belong to the individual engine case.
struct GpuUserQueue {
  GpuUserQueue() = default;
  GpuUserQueue(const GpuUserQueue&) = delete;
  GpuUserQueue& operator=(const GpuUserQueue&) = delete;
  void Initialize(const amdf_api_t* api, const amdf_gpu_api_t* gpu_api,
                  amdf_device_t* device, const amdf_queue_family_info_t& family,
                  amdf_queue_producer_mode_t producer_mode,
                  const amdf_gpu_queue_scratch_t& scratch,
                  amdf_user_queue_capabilities_t required_capabilities =
                      AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
                  uint64_t ring_byte_length = 0);
  bool Release(const amdf_api_t* api);

  // Publishes an already-written PM4/SDMA stream, with the engine's index
  // units. AQL instead reserves indices before publishing each packet's valid
  // header.
  void PublishStream(uint64_t producer_index);
  void WaitConsumed(const amdf_api_t* api, uint64_t producer_index);

  // Queue retained until all submitted work and borrowed memory uses complete.
  amdf_user_queue_t* queue = nullptr;
  // Host view released before queue destruction.
  amdf_user_queue_mapping_t* mapping = nullptr;
  // Native format, capabilities and ring geometry.
  amdf_user_queue_info_t info = {};
  // Borrowed producer addresses valid until mapping is destroyed.
  amdf_user_queue_mapping_info_t host = {};
  // Optional exact-owning-GPU view, retired before its queue can be destroyed.
  struct {
    // Borrow released only after the case's device publisher has stopped.
    amdf_user_queue_mapping_t* mapping = nullptr;
    // GPU addresses, independent of the host mapping's virtual addresses.
    amdf_user_queue_mapping_info_t info = {};
  } producer;
};

#endif  // AMDF_CTS_GPU_UTIL_USER_QUEUE_H_
