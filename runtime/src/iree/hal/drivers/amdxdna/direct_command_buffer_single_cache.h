// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
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
  void* ctrl_code_mapped_ptr;
  iree_hal_amdxdna_native_command_t* command;
  // Static descriptor identity for late-bound START_NPU template reuse. The
  // cached command owns the mutable control-code BO/native command, while these
  // fields identify which immutable executable-owned run template it came from.
  const iree_hal_amdxdna_u32_list_t* src_asm_inst;
  const iree_hal_amdxdna_u32_list_t* src_patches;
  iree_host_size_t src_constant_count;
  bool src_use_native_partial_elf;
  uint64_t last_use;
  // Protected by the parent cache mutex. Non-zero means this mutable native
  // command BO has been issued and must not be rewritten, reused, or evicted
  // until the completion queue releases it.
  uint32_t in_flight_count;
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

iree_status_t
iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches, iree_host_size_t constant_count,
    bool use_native_partial_elf, iree_host_size_t binding_count,
    iree_hal_amdxdna_single_command_cache_entry_t** out_entry);

iree_status_t iree_hal_amdxdna_update_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count);

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

void iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
    iree_hal_amdxdna_single_command_cache_entry_t* entry,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches, iree_host_size_t constant_count,
    bool use_native_partial_elf, void* ctrl_code_mapped_ptr);

void iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
    iree_hal_amdxdna_single_command_cache_entry_t* entry);

void iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry);

// Discards a non-in-flight entry after a failed in-place rewrite. The caller
// must hold cache->mutex.
void iree_hal_amdxdna_single_command_cache_entry_discard(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_
