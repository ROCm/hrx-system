// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_H_

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/completion_queue.h"
#include "iree/hal/drivers/amdxdna/device.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// `out_command_buffer` must be released by the caller (see
// iree_hal_command_buffer_release).
iree_status_t iree_hal_amdxdna_direct_command_buffer_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Static dispatch lowering computed during command-buffer recording. All
// pointer fields borrow executable-owned storage; recorded command buffers
// retain the executable, so these stay valid until apply. Binding-table-derived
// BO handles, device addresses, and native command objects are intentionally
// absent: those are late-bound by the queue worker after queue_execute provides
// the final binding table.
typedef struct iree_hal_amdxdna_dispatch_plan_t {
  iree_hal_amdxdna_executable* executable;
  iree_hal_amdxdna_kernel_params_t* kernel_params;
  uint32_t entry_point;
  iree_host_size_t control_code_count;
  const iree_hal_amdxdna_u32_list_t* control_codes;
  iree_host_size_t patch_table_count;
  const iree_hal_amdxdna_u32_list_t* patch_tables;
  iree_host_size_t constant_patch_table_count;
  const iree_hal_amdxdna_write32_constant_patch_list_t* constant_patch_tables;
  iree_host_size_t data_payload_count;
  const iree_hal_amdxdna_u32_list_t* data_payloads;
  uint32_t data_payload_run_count;
  iree_const_byte_span_t pdi_span;
  iree_const_byte_span_t xclbin_span;
  iree_string_view_t kernel_name;
  bool use_native_partial_elf_context;
  bool has_host_patch_table;
  bool multi_control_code_or_pdi;
  bool use_chain_accumulation_policy;
} iree_hal_amdxdna_dispatch_plan_t;

iree_status_t iree_hal_amdxdna_dispatch_plan_initialize(
    const iree_hal_amdxdna_native_c_device_caps_t* native_caps,
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_amdxdna_dispatch_plan_t* out_plan);

iree_status_t iree_hal_amdxdna_direct_command_buffer_dispatch_plan(
    iree_hal_command_buffer_t* base_command_buffer,
    const iree_hal_amdxdna_dispatch_plan_t* plan,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags);

void iree_hal_amdxdna_direct_command_buffer_set_completion_batch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_amdxdna_completion_batch_t* completion_batch);

void iree_hal_amdxdna_device_destroy_single_command_cache(
    iree_hal_amdxdna_device* device);

void iree_hal_amdxdna_device_destroy_chain_command_cache(
    iree_hal_amdxdna_device* device);

// Invalidates non-in-flight cached native commands tied to |queue| and marks
// in-flight entries for discard once native completion releases them. The queue
// belongs to a native context that is about to be released by the context
// cache, so cache entries keyed by it must not outlive that context identity.
void iree_hal_amdxdna_device_invalidate_command_caches_for_queue(
    iree_hal_amdxdna_device* device, iree_hal_amdxdna_native_queue_t* queue);

// Returns true when a cached chain command's already staged control-code words
// must be rewritten to match a freshly recorded command. Generic allocator
// caching may legally change HAL buffer wrapper/native BO identity between
// otherwise equivalent submissions; native instruction-code update decisions
// must key on the effective patched control-code bytes, not object identity.
bool iree_hal_amdxdna_direct_command_buffer_control_words_changed(
    const uint32_t* cached_words, iree_host_size_t cached_word_count,
    const uint32_t* fresh_words, iree_host_size_t fresh_word_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_H_
