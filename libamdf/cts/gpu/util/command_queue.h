// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_UTIL_COMMAND_QUEUE_H_
#define AMDF_CTS_GPU_UTIL_COMMAND_QUEUE_H_

#include <span>

#include "libamdf/cts/gpu/util/user_queue.h"
#include "util/mapped_memory.h"

// Selects an admitted host publication mode, preferring a directly mapped
// single-producer queue. Zero means neither permitted mode is available.
amdf_queue_publication_modes_t SelectGpuHostPublication(
    const amdf_queue_family_info_t& family,
    amdf_queue_publication_modes_t permitted_modes);

// Case-owned finite PM4/SDMA stream. USER writes directly into the native ring;
// KERNEL writes executable command memory and submits successive ranges. The
// caller encodes complete packets, explicit cache work and completion markers,
// and observes payloads before retiring command storage. No stream wraps or
// reuses the owner's storage, and each publication retires before the next.
// KERNEL callers may also submit their own already-published command buffers.
class GpuCommandQueue {
 public:
  // Zero command length retains default storage. A positive length requests
  // that native USER ring capacity or at least that much KERNEL IB backing.
  void Initialize(const amdf_api_t* api, const amdf_gpu_api_t* gpu_api,
                  amdf_device_t* device, amdf_memory_scope_t* system_scope,
                  const amdf_queue_family_info_t& family,
                  amdf_queue_publication_modes_t publication_mode,
                  uint64_t command_byte_length = 0);

  std::span<uint32_t> words() const { return words_; }
  // PM4 USER commands enter on the primary ring; KERNEL commands enter in an
  // IB. The case uses this distinction when encoding control-flow packets.
  amdf_queue_publication_modes_t publication_mode() const {
    return kernel_queue_ ? AMDF_QUEUE_PUBLICATION_MODE_KERNEL
                         : AMDF_QUEUE_PUBLICATION_MODE_USER;
  }
  const amdf_device_id_t& device_id() const { return device_id_; }
  // Native object identity, without inventing a USER queue ID for KERNEL.
  const void* native_handle() const;

  // Publishes only the newly appended complete commands through word_count.
  // Command-cache publication is independent of caller-owned payload edges.
  void Publish(const amdf_api_t* api, const amdf_gpu_api_t* gpu_api,
               size_t word_count);
  // KERNEL entry for an already-published caller-owned command buffer. The
  // caller retains its memory and every referenced allocation through native
  // retirement and queue removal, just as for commands written through words().
  void Submit(const amdf_gpu_api_t* gpu_api,
              const amdf_gpu_kernel_command_t& command);
  // Retires the last publication. USER consumption permits command-storage
  // reuse only; it does not establish shader completion or payload visibility.
  void WaitRetired(const amdf_api_t* api);
  // Removes the queue before command backing; failure retains all backing.
  bool Release(const amdf_api_t* api);

 private:
  // Directly published queue/mapping when USER publication was selected.
  GpuUserQueue user_queue_;
  // Kernel-mediated queue when KERNEL publication was selected.
  amdf_kernel_queue_t* kernel_queue_ = nullptr;
  // Executable command backing owned only by the KERNEL path.
  CtsMappedMemory commands_;
  // Writable command storage borrowed from the selected native owner.
  std::span<uint32_t> words_;
  // Exact native device identity reported by the selected queue.
  amdf_device_id_t device_id_ = {};
  // Native command format, defining USER producer index units.
  amdf_queue_command_type_t command_type_ = 0;
  // Cumulative number of words successfully published from words_.
  size_t published_word_count_ = 0;
  // Last accepted KERNEL point, used without assuming dense numbering.
  uint64_t submission_ = 0;
};

#endif  // AMDF_CTS_GPU_UTIL_COMMAND_QUEUE_H_
