// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-global cache of prepared single-dispatch native commands (ERT_START_CU
// / START_NPU path), reused across one-shot command-buffer instances and
// rebound/rewritten in place to match a freshly recorded dispatch.
enum { kAmdxdnaSingleCommandCacheCapacity = 8 };

typedef struct iree_hal_amdxdna_single_command_cache_entry_t {
  iree_hal_amdxdna_native_queue_t* queue;
  uint32_t cu_index;
  uint32_t* ctrl_words;
  iree_host_size_t ctrl_word_count;
  iree_hal_amdxdna_native_buffer_t** binding_buffers;
  uint64_t* binding_device_addrs;
  iree_device_size_t* binding_offsets;
  iree_device_size_t* binding_lengths;
  iree_host_size_t binding_count;
  iree_hal_amdxdna_native_buffer_t* ctrl_code_buffer;
  iree_hal_amdxdna_native_command_t* command;
  uint64_t last_use;
} iree_hal_amdxdna_single_command_cache_entry_t;

typedef struct iree_hal_amdxdna_device_single_command_cache_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_single_command_cache_entry_t
      entries[kAmdxdnaSingleCommandCacheCapacity];
  iree_host_size_t entry_count;
  uint64_t use_clock;
} iree_hal_amdxdna_device_single_command_cache_t;

typedef struct iree_hal_amdxdna_device iree_hal_amdxdna_device;

iree_hal_amdxdna_device_single_command_cache_t*
iree_hal_amdxdna_get_single_command_cache(iree_hal_amdxdna_device* device);

iree_status_t iree_hal_amdxdna_find_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count,
    iree_hal_amdxdna_single_command_cache_entry_t** out_entry);

iree_hal_amdxdna_single_command_cache_entry_t*
iree_hal_amdxdna_store_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count,
    iree_hal_amdxdna_native_buffer_t* ctrl_code_buffer,
    iree_hal_amdxdna_native_command_t* command);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_
