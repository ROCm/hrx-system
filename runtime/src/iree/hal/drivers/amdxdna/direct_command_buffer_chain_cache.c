// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/amdxdna/device.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

static iree_host_size_t iree_hal_amdxdna_max_host_size(iree_host_size_t lhs,
                                                       iree_host_size_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static iree_status_t iree_hal_amdxdna_reserve_array(
    iree_allocator_t host_allocator, iree_host_size_t element_size,
    iree_host_size_t min_capacity, void** inout_data,
    iree_host_size_t* inout_capacity) {
  if (*inout_capacity >= min_capacity) return iree_ok_status();
  iree_host_size_t new_capacity = *inout_capacity ? *inout_capacity : 4;
  while (new_capacity < min_capacity) {
    if (new_capacity > (IREE_HOST_SIZE_MAX / 2)) {
      new_capacity = min_capacity;
      break;
    }
    new_capacity *= 2;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      host_allocator, new_capacity, element_size, inout_data));
  *inout_capacity = new_capacity;
  return iree_ok_status();
}

static bool iree_hal_amdxdna_u32_span_equal(const uint32_t* lhs,
                                            iree_host_size_t lhs_count,
                                            const uint32_t* rhs,
                                            iree_host_size_t rhs_count) {
  return lhs_count == rhs_count &&
         (lhs_count == 0 ||
          memcmp(lhs, rhs, lhs_count * sizeof(uint32_t)) == 0);
}

static bool iree_hal_amdxdna_u8_span_equal(const uint8_t* lhs,
                                           iree_host_size_t lhs_count,
                                           const uint8_t* rhs,
                                           iree_host_size_t rhs_count) {
  return lhs_count == rhs_count &&
         (lhs_count == 0 || memcmp(lhs, rhs, lhs_count) == 0);
}

static bool iree_hal_amdxdna_ptr_span_equal(const void* const* lhs,
                                            const void* const* rhs,
                                            iree_host_size_t count) {
  return count == 0 || memcmp(lhs, rhs, count * sizeof(void*)) == 0;
}

static bool iree_hal_amdxdna_u64_span_equal(const uint64_t* lhs,
                                            const uint64_t* rhs,
                                            iree_host_size_t count) {
  return count == 0 || memcmp(lhs, rhs, count * sizeof(uint64_t)) == 0;
}

static bool iree_hal_amdxdna_device_size_span_equal(
    const iree_device_size_t* lhs, const iree_device_size_t* rhs,
    iree_host_size_t count) {
  return count == 0 ||
         memcmp(lhs, rhs, count * sizeof(iree_device_size_t)) == 0;
}

static void iree_hal_amdxdna_chain_cmd_free_signature(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd) {
  iree_allocator_free(host_allocator, cmd->ctrl_words);
  iree_allocator_free(host_allocator, cmd->binding_buffers);
  iree_allocator_free(host_allocator, cmd->binding_device_addrs);
  iree_allocator_free(host_allocator, cmd->binding_offsets);
  iree_allocator_free(host_allocator, cmd->binding_lengths);
  cmd->ctrl_words = NULL;
  cmd->ctrl_word_count = 0;
  cmd->binding_buffers = NULL;
  cmd->binding_device_addrs = NULL;
  cmd->binding_offsets = NULL;
  cmd->binding_lengths = NULL;
  cmd->binding_count = 0;
}

static void iree_hal_amdxdna_chain_cmd_free_deferred(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd) {
  iree_allocator_free(host_allocator, cmd->src_constants);
  cmd->src_asm_inst = NULL;
  cmd->src_patches = NULL;
  cmd->src_constants = NULL;
  cmd->src_constant_count = 0;
  memset(&cmd->src_cu_idx, 0, sizeof(cmd->src_cu_idx));
  cmd->src_use_native_partial_elf = false;
  cmd->repeat_count = 1;
}

void iree_hal_amdxdna_chain_cmd_initialize(iree_hal_amdxdna_chain_cmd_t* cmd) {
  memset(cmd, 0, sizeof(*cmd));
  cmd->repeat_count = 1;
  cmd->native_bindings_current = true;
}

void iree_hal_amdxdna_chain_cmd_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd) {
  if (!cmd) return;
  iree_hal_amdxdna_native_command_c_destroy(cmd->command);
  iree_hal_amdxdna_native_buffer_c_destroy(cmd->ctrl_code);
  iree_hal_amdxdna_chain_cmd_free_signature(host_allocator, cmd);
  iree_hal_amdxdna_chain_cmd_free_deferred(host_allocator, cmd);
  iree_hal_amdxdna_chain_cmd_initialize(cmd);
}

void iree_hal_amdxdna_chain_cmd_move(iree_hal_amdxdna_chain_cmd_t* dst,
                                     iree_hal_amdxdna_chain_cmd_t* src) {
  *dst = *src;
  iree_hal_amdxdna_chain_cmd_initialize(src);
}

iree_status_t iree_hal_amdxdna_chain_cmd_set_signature(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  uint32_t* new_ctrl_words = NULL;
  iree_hal_amdxdna_native_buffer_t** new_binding_buffers = NULL;
  uint64_t* new_binding_device_addrs = NULL;
  iree_device_size_t* new_binding_offsets = NULL;
  iree_device_size_t* new_binding_lengths = NULL;
  iree_status_t status = iree_ok_status();
  // Inputs may alias the command being rewritten (deferred descriptors are
  // materialized in place), so copy first and only then release the old arrays.
  if (ctrl_word_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, ctrl_word_count,
                                         sizeof(*new_ctrl_words),
                                         (void**)&new_ctrl_words);
    if (iree_status_is_ok(status)) {
      memcpy(new_ctrl_words, ctrl_words,
             ctrl_word_count * sizeof(*new_ctrl_words));
    }
  }
  if (iree_status_is_ok(status) && binding_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, binding_count,
                                         sizeof(*new_binding_buffers),
                                         (void**)&new_binding_buffers);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(host_allocator, binding_count,
                                           sizeof(*new_binding_device_addrs),
                                           (void**)&new_binding_device_addrs);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(host_allocator, binding_count,
                                           sizeof(*new_binding_offsets),
                                           (void**)&new_binding_offsets);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(host_allocator, binding_count,
                                           sizeof(*new_binding_lengths),
                                           (void**)&new_binding_lengths);
    }
    if (iree_status_is_ok(status)) {
      memcpy(new_binding_buffers, binding_buffers,
             binding_count * sizeof(*new_binding_buffers));
      memcpy(new_binding_device_addrs, binding_device_addrs,
             binding_count * sizeof(*new_binding_device_addrs));
      memcpy(new_binding_offsets, binding_offsets,
             binding_count * sizeof(*new_binding_offsets));
      memcpy(new_binding_lengths, binding_lengths,
             binding_count * sizeof(*new_binding_lengths));
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_chain_cmd_free_signature(host_allocator, cmd);
    cmd->ctrl_words = new_ctrl_words;
    cmd->ctrl_word_count = ctrl_word_count;
    cmd->binding_buffers = new_binding_buffers;
    cmd->binding_device_addrs = new_binding_device_addrs;
    cmd->binding_offsets = new_binding_offsets;
    cmd->binding_lengths = new_binding_lengths;
    cmd->binding_count = binding_count;
    return iree_ok_status();
  }

  iree_allocator_free(host_allocator, new_binding_lengths);
  iree_allocator_free(host_allocator, new_binding_offsets);
  iree_allocator_free(host_allocator, new_binding_device_addrs);
  iree_allocator_free(host_allocator, new_binding_buffers);
  iree_allocator_free(host_allocator, new_ctrl_words);
  return status;
}

iree_status_t iree_hal_amdxdna_chain_cmd_set_deferred_descriptor(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_chain_cmd_set_signature(
      host_allocator, cmd, NULL, 0, binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, binding_count));
  iree_hal_amdxdna_chain_cmd_free_deferred(host_allocator, cmd);
  if (constants.data_length != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, constants.data_length, sizeof(*cmd->src_constants),
        (void**)&cmd->src_constants));
    memcpy(cmd->src_constants, constants.data, constants.data_length);
  }
  cmd->src_constant_count = constants.data_length;
  cmd->src_asm_inst = asm_inst;
  cmd->src_patches = patches;
  cmd->src_cu_idx = cu_idx;
  cmd->src_use_native_partial_elf = use_native_partial_elf;
  cmd->repeat_count = 1;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_chain_cmd_clone_unbuilt_descriptor(
    iree_allocator_t host_allocator, const iree_hal_amdxdna_chain_cmd_t* src,
    iree_hal_amdxdna_chain_cmd_t* out_cmd) {
  iree_hal_amdxdna_chain_cmd_initialize(out_cmd);
  return iree_hal_amdxdna_chain_cmd_set_deferred_descriptor(
      host_allocator, out_cmd, src->src_asm_inst, src->src_patches,
      src->src_cu_idx,
      iree_make_const_byte_span(src->src_constants, src->src_constant_count),
      src->src_use_native_partial_elf, src->binding_buffers,
      src->binding_device_addrs, src->binding_offsets, src->binding_lengths,
      src->binding_count);
}

void iree_hal_amdxdna_chain_group_initialize(
    iree_hal_amdxdna_chain_group_t* group) {
  memset(group, 0, sizeof(*group));
}

void iree_hal_amdxdna_chain_group_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group) {
  if (!group) return;
  iree_hal_amdxdna_native_context_ref_release(group->context);
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    iree_hal_amdxdna_chain_cmd_deinitialize(host_allocator, &group->cmds[i]);
  }
  for (iree_host_size_t i = 0; i < group->reconf_buffer_count; ++i) {
    iree_hal_amdxdna_native_buffer_c_destroy(group->reconf_buffers[i]);
  }
  iree_allocator_free(host_allocator, group->cmds);
  iree_allocator_free(host_allocator, group->reconf_buffers);
  iree_allocator_free(host_allocator, group->binding_refs);
  iree_hal_amdxdna_chain_group_initialize(group);
}

void iree_hal_amdxdna_chain_group_move(iree_hal_amdxdna_chain_group_t* dst,
                                       iree_hal_amdxdna_chain_group_t* src) {
  *dst = *src;
  iree_hal_amdxdna_chain_group_initialize(src);
}

iree_status_t iree_hal_amdxdna_chain_group_append_cmd_move(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group,
    iree_hal_amdxdna_chain_cmd_t* cmd) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_reserve_array(
      host_allocator, sizeof(*group->cmds), group->cmd_count + 1,
      (void**)&group->cmds, &group->cmd_capacity));
  iree_hal_amdxdna_chain_cmd_move(&group->cmds[group->cmd_count++], cmd);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_chain_group_append_reconf_buffer(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_reserve_array(
      host_allocator, sizeof(*group->reconf_buffers),
      group->reconf_buffer_count + 1, (void**)&group->reconf_buffers,
      &group->reconf_buffer_capacity));
  group->reconf_buffers[group->reconf_buffer_count++] = buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_chain_group_append_binding_ref_unique(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group,
    iree_hal_buffer_ref_t binding_ref) {
  for (iree_host_size_t i = 0; i < group->binding_ref_count; ++i) {
    const iree_hal_buffer_ref_t* existing = &group->binding_refs[i];
    if (existing->buffer == binding_ref.buffer &&
        existing->offset == binding_ref.offset &&
        existing->length == binding_ref.length) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_reserve_array(
      host_allocator, sizeof(*group->binding_refs),
      group->binding_ref_count + 1, (void**)&group->binding_refs,
      &group->binding_ref_capacity));
  group->binding_refs[group->binding_ref_count++] = binding_ref;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_chain_group_take_cmds(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* dst,
    iree_hal_amdxdna_chain_group_t* src) {
  for (iree_host_size_t i = 0; i < dst->cmd_count; ++i) {
    iree_hal_amdxdna_chain_cmd_deinitialize(host_allocator, &dst->cmds[i]);
  }
  iree_allocator_free(host_allocator, dst->cmds);
  dst->cmds = src->cmds;
  dst->cmd_count = src->cmd_count;
  dst->cmd_capacity = src->cmd_capacity;
  src->cmds = NULL;
  src->cmd_count = 0;
  src->cmd_capacity = 0;
  return iree_ok_status();
}

void iree_hal_amdxdna_chain_accum_initialize(
    iree_hal_amdxdna_chain_accum_t* accum) {
  memset(accum, 0, sizeof(*accum));
}

void iree_hal_amdxdna_chain_accum_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_accum_t* accum) {
  iree_hal_amdxdna_chain_accum_clear(host_allocator, accum);
  iree_allocator_free(host_allocator, accum->groups);
  iree_hal_amdxdna_chain_accum_initialize(accum);
}

void iree_hal_amdxdna_chain_accum_clear(iree_allocator_t host_allocator,
                                        iree_hal_amdxdna_chain_accum_t* accum) {
  for (iree_host_size_t i = 0; i < accum->group_count; ++i) {
    iree_hal_amdxdna_chain_group_deinitialize(host_allocator,
                                              &accum->groups[i]);
  }
  accum->group_count = 0;
}

iree_status_t iree_hal_amdxdna_chain_accum_append_group(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_accum_t* accum,
    iree_hal_amdxdna_chain_group_t** out_group) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_reserve_array(
      host_allocator, sizeof(*accum->groups), accum->group_count + 1,
      (void**)&accum->groups, &accum->group_capacity));
  iree_hal_amdxdna_chain_group_t* group = &accum->groups[accum->group_count++];
  iree_hal_amdxdna_chain_group_initialize(group);
  *out_group = group;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_chain_command_cache_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_device_chain_command_cache_t** out_cache) {
  *out_cache = NULL;
  iree_hal_amdxdna_device_chain_command_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  memset(cache, 0, sizeof(*cache));
  cache->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

iree_hal_amdxdna_device_chain_command_cache_t*
iree_hal_amdxdna_get_chain_command_cache(iree_hal_amdxdna_device* device) {
  if (device->chain_command_cache) return device->chain_command_cache;
  iree_slim_mutex_lock(&device->command_cache_mutex);
  if (!device->chain_command_cache) {
    (void)iree_hal_amdxdna_chain_command_cache_create(
        device->host_allocator, &device->chain_command_cache);
  }
  iree_slim_mutex_unlock(&device->command_cache_mutex);
  return device->chain_command_cache;
}

void iree_hal_amdxdna_chain_command_cache_entry_clear_chains(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry) {
  for (iree_host_size_t i = 0; i < entry->chain_count; ++i) {
    iree_hal_amdxdna_native_command_c_destroy(entry->chains[i]);
  }
  entry->chain_count = 0;
}

static void iree_hal_amdxdna_chain_command_cache_entry_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry) {
  iree_hal_amdxdna_chain_command_cache_entry_clear_chains(host_allocator,
                                                          entry);
  iree_allocator_free(host_allocator, entry->chains);
  iree_hal_amdxdna_chain_group_deinitialize(host_allocator, &entry->group);
  memset(entry, 0, sizeof(*entry));
}

void iree_hal_amdxdna_device_destroy_chain_command_cache(
    iree_hal_amdxdna_device* device) {
  iree_hal_amdxdna_device_chain_command_cache_t* cache =
      device->chain_command_cache;
  if (!cache) return;
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_chain_command_cache_entry_deinitialize(
        cache->host_allocator, &cache->entries[i]);
  }
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
  device->chain_command_cache = NULL;
}

bool iree_hal_amdxdna_direct_command_buffer_control_words_changed(
    const uint32_t* cached_words, iree_host_size_t cached_word_count,
    const uint32_t* fresh_words, iree_host_size_t fresh_word_count) {
  if (cached_word_count != fresh_word_count) return true;
  if (cached_word_count == 0) return false;
  return memcmp(cached_words, fresh_words,
                cached_word_count * sizeof(uint32_t)) != 0;
}

static bool iree_hal_amdxdna_chain_cmd_device_signature_matches(
    const iree_hal_amdxdna_chain_cmd_t* lhs,
    const iree_hal_amdxdna_chain_cmd_t* rhs) {
  return iree_hal_amdxdna_u32_span_equal(lhs->ctrl_words, lhs->ctrl_word_count,
                                         rhs->ctrl_words,
                                         rhs->ctrl_word_count) &&
         lhs->binding_count == rhs->binding_count &&
         iree_hal_amdxdna_ptr_span_equal(
             (const void* const*)lhs->binding_buffers,
             (const void* const*)rhs->binding_buffers, lhs->binding_count) &&
         iree_hal_amdxdna_u64_span_equal(lhs->binding_device_addrs,
                                         rhs->binding_device_addrs,
                                         lhs->binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             lhs->binding_offsets, rhs->binding_offsets, lhs->binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             lhs->binding_lengths, rhs->binding_lengths, lhs->binding_count);
}

static bool iree_hal_amdxdna_chain_cmd_shape_matches(
    const iree_hal_amdxdna_chain_cmd_t* lhs,
    const iree_hal_amdxdna_chain_cmd_t* rhs) {
  return lhs->ctrl_word_count == rhs->ctrl_word_count &&
         lhs->binding_count == rhs->binding_count &&
         iree_hal_amdxdna_device_size_span_equal(
             lhs->binding_offsets, rhs->binding_offsets, lhs->binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             lhs->binding_lengths, rhs->binding_lengths, lhs->binding_count);
}

iree_host_size_t iree_hal_amdxdna_chain_group_logical_command_count(
    const iree_hal_amdxdna_chain_group_t* group) {
  iree_host_size_t total = 0;
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    const iree_host_size_t repeat_count =
        iree_hal_amdxdna_max_host_size(group->cmds[i].repeat_count, 1);
    if (repeat_count > IREE_HOST_SIZE_MAX - total) return IREE_HOST_SIZE_MAX;
    total += repeat_count;
  }
  return total;
}

static const iree_hal_amdxdna_chain_cmd_t*
iree_hal_amdxdna_chain_group_cmd_at_logical_index(
    const iree_hal_amdxdna_chain_group_t* group,
    iree_host_size_t logical_index) {
  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    const iree_host_size_t repeat_count =
        iree_hal_amdxdna_max_host_size(group->cmds[i].repeat_count, 1);
    if (logical_index < cursor + repeat_count) return &group->cmds[i];
    cursor += repeat_count;
  }
  return NULL;
}

iree_status_t iree_hal_amdxdna_expand_repeated_chain_descriptors(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group) {
  const iree_host_size_t logical_count =
      iree_hal_amdxdna_chain_group_logical_command_count(group);
  if (logical_count == group->cmd_count) return iree_ok_status();
  iree_hal_amdxdna_chain_cmd_t* expanded = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, logical_count, sizeof(*expanded), (void**)&expanded));
  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    iree_hal_amdxdna_chain_cmd_t* cmd = &group->cmds[i];
    if (cmd->built && cmd->repeat_count > 1) {
      iree_allocator_free(host_allocator, expanded);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna cmd-chain cannot expand a repeated built descriptor");
    }
    const iree_host_size_t repeat_count =
        iree_hal_amdxdna_max_host_size(cmd->repeat_count, 1);
    for (iree_host_size_t j = 1; j < repeat_count; ++j) {
      iree_status_t status =
          iree_hal_amdxdna_chain_cmd_clone_unbuilt_descriptor(
              host_allocator, cmd, &expanded[cursor + j]);
      if (!iree_status_is_ok(status)) {
        for (iree_host_size_t k = 0; k < cursor + j; ++k) {
          iree_hal_amdxdna_chain_cmd_deinitialize(host_allocator, &expanded[k]);
        }
        iree_allocator_free(host_allocator, expanded);
        return status;
      }
    }
    cursor += repeat_count;
  }
  cursor = 0;
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    iree_host_size_t repeat_count =
        iree_hal_amdxdna_max_host_size(group->cmds[i].repeat_count, 1);
    group->cmds[i].repeat_count = 1;
    iree_hal_amdxdna_chain_cmd_move(&expanded[cursor], &group->cmds[i]);
    cursor += repeat_count;
  }
  iree_allocator_free(host_allocator, group->cmds);
  group->cmds = expanded;
  group->cmd_count = logical_count;
  group->cmd_capacity = logical_count;
  return iree_ok_status();
}

bool iree_hal_amdxdna_chain_command_cache_device_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots) {
  if (cache->chain_count == 0 || cache->max_slots != max_slots ||
      cache->group.queue != group->queue ||
      cache->group.native_partial_elf != group->native_partial_elf ||
      cache->group.cmd_count != group->cmd_count ||
      cache->group.reconf_buffer_count != 0 ||
      group->reconf_buffer_count != 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    if (!iree_hal_amdxdna_chain_cmd_device_signature_matches(
            &cache->group.cmds[i], &group->cmds[i])) {
      return false;
    }
  }
  return true;
}

bool iree_hal_amdxdna_chain_command_cache_shape_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots) {
  if (cache->group.cmd_count == 0 || cache->max_slots != max_slots ||
      cache->group.queue != group->queue ||
      cache->group.native_partial_elf != group->native_partial_elf ||
      cache->group.cmd_count != group->cmd_count ||
      cache->group.reconf_buffer_count != 0 ||
      group->reconf_buffer_count != 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < group->cmd_count; ++i) {
    if (!iree_hal_amdxdna_chain_cmd_shape_matches(&cache->group.cmds[i],
                                                  &group->cmds[i])) {
      return false;
    }
  }
  return true;
}

bool iree_hal_amdxdna_chain_cmd_descriptor_matches(
    const iree_hal_amdxdna_chain_cmd_t* lhs,
    const iree_hal_amdxdna_chain_cmd_t* rhs) {
  return lhs->src_asm_inst == rhs->src_asm_inst &&
         lhs->src_patches == rhs->src_patches &&
         lhs->src_use_native_partial_elf == rhs->src_use_native_partial_elf &&
         lhs->src_cu_idx.index == rhs->src_cu_idx.index &&
         iree_hal_amdxdna_u8_span_equal(
             lhs->src_constants, lhs->src_constant_count, rhs->src_constants,
             rhs->src_constant_count) &&
         lhs->binding_count == rhs->binding_count &&
         iree_hal_amdxdna_ptr_span_equal(
             (const void* const*)lhs->binding_buffers,
             (const void* const*)rhs->binding_buffers, lhs->binding_count) &&
         iree_hal_amdxdna_u64_span_equal(lhs->binding_device_addrs,
                                         rhs->binding_device_addrs,
                                         lhs->binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             lhs->binding_offsets, rhs->binding_offsets, lhs->binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             lhs->binding_lengths, rhs->binding_lengths, lhs->binding_count);
}

bool iree_hal_amdxdna_chain_cmd_matches_raw_descriptor(
    const iree_hal_amdxdna_chain_cmd_t* cmd,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    const uint64_t* binding_device_addrs,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  if (cmd->built || cmd->src_asm_inst != asm_inst ||
      cmd->src_patches != patches ||
      cmd->src_use_native_partial_elf != use_native_partial_elf ||
      cmd->src_cu_idx.index != cu_idx.index ||
      cmd->src_constant_count != constants.data_length) {
    return false;
  }
  if (constants.data_length != 0 &&
      memcmp(cmd->src_constants, constants.data, constants.data_length) != 0) {
    return false;
  }
  return cmd->binding_count == binding_count &&
         iree_hal_amdxdna_ptr_span_equal(
             (const void* const*)cmd->binding_buffers,
             (const void* const*)binding_buffers, binding_count) &&
         iree_hal_amdxdna_u64_span_equal(cmd->binding_device_addrs,
                                         binding_device_addrs, binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             cmd->binding_offsets, binding_offsets, binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             cmd->binding_lengths, binding_lengths, binding_count);
}

bool iree_hal_amdxdna_chain_command_cache_descriptor_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots) {
  const iree_host_size_t cache_logical_count =
      iree_hal_amdxdna_chain_group_logical_command_count(&cache->group);
  const iree_host_size_t group_logical_count =
      iree_hal_amdxdna_chain_group_logical_command_count(group);
  if (cache->chain_count == 0 || cache->max_slots != max_slots ||
      cache->group.queue != group->queue ||
      cache->group.native_partial_elf != group->native_partial_elf ||
      cache_logical_count != group_logical_count ||
      cache->group.reconf_buffer_count != 0 ||
      group->reconf_buffer_count != 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < group_logical_count; ++i) {
    const iree_hal_amdxdna_chain_cmd_t* cache_cmd =
        iree_hal_amdxdna_chain_group_cmd_at_logical_index(&cache->group, i);
    const iree_hal_amdxdna_chain_cmd_t* group_cmd =
        iree_hal_amdxdna_chain_group_cmd_at_logical_index(group, i);
    if (!cache_cmd || !group_cmd) return false;
    if (!iree_hal_amdxdna_chain_cmd_descriptor_matches(cache_cmd, group_cmd)) {
      return false;
    }
  }
  return true;
}

iree_status_t iree_hal_amdxdna_update_cached_chain_cmd(
    iree_hal_amdxdna_chain_cmd_t* cached,
    const iree_hal_amdxdna_chain_cmd_t* fresh, bool* out_packet_changed,
    bool* out_code_changed, bool* out_device_bindings_changed,
    bool* out_rebound) {
  if (out_packet_changed) *out_packet_changed = false;
  if (out_code_changed) *out_code_changed = false;
  if (out_device_bindings_changed) *out_device_bindings_changed = false;
  if (out_rebound) *out_rebound = false;
  if (!iree_hal_amdxdna_chain_cmd_shape_matches(cached, fresh)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna cached native chain command shape changed");
  }
  const bool code_changed =
      iree_hal_amdxdna_direct_command_buffer_control_words_changed(
          cached->ctrl_words, cached->ctrl_word_count, fresh->ctrl_words,
          fresh->ctrl_word_count);
  const bool device_bindings_changed =
      !iree_hal_amdxdna_u64_span_equal(cached->binding_device_addrs,
                                       fresh->binding_device_addrs,
                                       cached->binding_count) ||
      !iree_hal_amdxdna_device_size_span_equal(cached->binding_offsets,
                                               fresh->binding_offsets,
                                               cached->binding_count) ||
      !iree_hal_amdxdna_device_size_span_equal(cached->binding_lengths,
                                               fresh->binding_lengths,
                                               cached->binding_count);
  const bool native_bindings_changed =
      !iree_hal_amdxdna_ptr_span_equal(
          (const void* const*)cached->binding_buffers,
          (const void* const*)fresh->binding_buffers, cached->binding_count) ||
      device_bindings_changed || !cached->native_bindings_current;
  if (code_changed) {
    void* ctrl_ptr = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_buffer_c_map(cached->ctrl_code, &ctrl_ptr));
    memcpy(ctrl_ptr, fresh->ctrl_words,
           fresh->ctrl_word_count * sizeof(*fresh->ctrl_words));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_mark_code_dirty(cached->command));
    memcpy(cached->ctrl_words, fresh->ctrl_words,
           fresh->ctrl_word_count * sizeof(*fresh->ctrl_words));
  }
  if (native_bindings_changed) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_reset_bound_buffers(cached->command));
    for (iree_host_size_t i = 0; i < fresh->binding_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_bind_buffer(
          cached->command, i + 1, fresh->binding_buffers[i],
          fresh->binding_offsets[i], fresh->binding_lengths[i]));
    }
    cached->native_bindings_current = true;
    if (out_rebound) *out_rebound = true;
  }
  memcpy(cached->binding_buffers, fresh->binding_buffers,
         fresh->binding_count * sizeof(*cached->binding_buffers));
  memcpy(cached->binding_device_addrs, fresh->binding_device_addrs,
         fresh->binding_count * sizeof(*cached->binding_device_addrs));
  memcpy(cached->binding_offsets, fresh->binding_offsets,
         fresh->binding_count * sizeof(*cached->binding_offsets));
  memcpy(cached->binding_lengths, fresh->binding_lengths,
         fresh->binding_count * sizeof(*cached->binding_lengths));
  if (out_packet_changed) {
    *out_packet_changed =
        code_changed || device_bindings_changed || native_bindings_changed;
  }
  if (out_code_changed) *out_code_changed = code_changed;
  if (out_device_bindings_changed) {
    *out_device_bindings_changed = device_bindings_changed;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_chain_command_cache_entry_append_chain(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry,
    iree_hal_amdxdna_native_command_t* chain) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_reserve_array(
      host_allocator, sizeof(*entry->chains), entry->chain_count + 1,
      (void**)&entry->chains, &entry->chain_capacity));
  entry->chains[entry->chain_count++] = chain;
  return iree_ok_status();
}

iree_hal_amdxdna_chain_command_cache_entry_t*
iree_hal_amdxdna_chain_command_cache_allocate_entry(
    iree_hal_amdxdna_device_chain_command_cache_t* cache) {
  iree_hal_amdxdna_chain_command_cache_entry_t* entry = NULL;
  if (cache->entry_count < kAmdxdnaChainCommandCacheCapacity) {
    entry = &cache->entries[cache->entry_count++];
    memset(entry, 0, sizeof(*entry));
    iree_hal_amdxdna_chain_group_initialize(&entry->group);
    return entry;
  }
  entry = &cache->entries[0];
  for (iree_host_size_t i = 1; i < cache->entry_count; ++i) {
    if (cache->entries[i].last_use < entry->last_use) {
      entry = &cache->entries[i];
    }
  }
  iree_hal_amdxdna_chain_command_cache_entry_deinitialize(cache->host_allocator,
                                                          entry);
  iree_hal_amdxdna_chain_group_initialize(&entry->group);
  return entry;
}
