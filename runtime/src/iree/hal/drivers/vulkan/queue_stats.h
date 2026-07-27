// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_QUEUE_STATS_H_
#define IREE_HAL_DRIVERS_VULKAN_QUEUE_STATS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Snapshot of Vulkan queue-owned BDA publication cache counters.
typedef struct iree_hal_vulkan_bda_publication_cache_stats_t {
  // Number of BDA publication blocks currently owned by the queue.
  uint64_t block_count;
} iree_hal_vulkan_bda_publication_cache_stats_t;

// Snapshot of Vulkan queue-owned native BDA replay cache counters.
typedef struct iree_hal_vulkan_native_replay_cache_stats_t {
  // Number of native replay instances currently cached.
  uint64_t instance_count;
  // Maximum cached native BDA replay instances retained by the queue.
  uint64_t max_instance_count;
  // Idle native replay instances retained per command buffer by queue trim.
  uint64_t retained_instance_count;
  // BDA publication bytes retained by cached replay instances.
  uint64_t publication_bytes;
  // Maximum BDA publication bytes retained by cached replay instances.
  uint64_t max_publication_bytes;
  // Peak native replay instances cached at once.
  uint64_t peak_instance_count;
  // Peak BDA publication bytes retained by cached replay instances.
  uint64_t peak_publication_bytes;
  // Cached replay acquisitions using an idle native command buffer.
  uint64_t hit_count;
  // Cached replay acquisitions that had to record a native command buffer.
  uint64_t miss_count;
  // Cached replay instances successfully recorded and retained.
  uint64_t create_count;
  // Cached replay creations for command buffers already busy on the queue.
  uint64_t fork_count;
  // Cached replay hits that reused existing BDA publication bytes.
  uint64_t publication_skip_count;
  // Cached replay hits that republished changed BDA table bytes.
  uint64_t publication_update_count;
  // Cached replay acquisitions bypassed because descriptors were required.
  uint64_t descriptor_bypass_count;
  // Cached replay acquisitions bypassed because profiling was active.
  uint64_t profile_bypass_count;
  // Cached replay acquisitions bypassed because the command buffer is one-shot.
  uint64_t one_shot_bypass_count;
  // Cached replay acquisitions bypassed by configured capacity limits.
  uint64_t capacity_bypass_count;
  // Cached replay instances destroyed by queue trim.
  uint64_t trim_count;
} iree_hal_vulkan_native_replay_cache_stats_t;

// Samples aggregate BDA publication cache counters across all queue lanes.
void iree_hal_vulkan_logical_device_sample_bda_publication_cache_stats(
    iree_hal_device_t* device,
    iree_hal_vulkan_bda_publication_cache_stats_t* out_stats);

// Samples aggregate native BDA replay cache counters across all queue lanes.
void iree_hal_vulkan_logical_device_sample_native_replay_cache_stats(
    iree_hal_device_t* device,
    iree_hal_vulkan_native_replay_cache_stats_t* out_stats);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_QUEUE_STATS_H_
