// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"

#include <string.h>

#include "iree/hal/drivers/amdxdna/device.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

static bool iree_hal_amdxdna_u32_span_equal(const uint32_t* lhs,
                                            iree_host_size_t lhs_count,
                                            const uint32_t* rhs,
                                            iree_host_size_t rhs_count) {
  return lhs_count == rhs_count &&
         (lhs_count == 0 ||
          memcmp(lhs, rhs, lhs_count * sizeof(uint32_t)) == 0);
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

static void iree_hal_amdxdna_single_command_cache_entry_deinitialize(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry) {
  iree_hal_amdxdna_native_command_c_destroy(entry->command);
  iree_hal_amdxdna_native_buffer_c_destroy(entry->ctrl_code_buffer);
  iree_allocator_free(cache->host_allocator, entry->ctrl_words);
  iree_allocator_free(cache->host_allocator, entry->binding_buffers);
  iree_allocator_free(cache->host_allocator, entry->binding_device_addrs);
  iree_allocator_free(cache->host_allocator, entry->binding_offsets);
  iree_allocator_free(cache->host_allocator, entry->binding_lengths);
  iree_allocator_free(cache->host_allocator, entry->owned_src_asm_inst.data);
  iree_allocator_free(cache->host_allocator, entry->owned_src_patches.data);
  memset(entry, 0, sizeof(*entry));
}

void iree_hal_amdxdna_single_command_cache_entry_discard(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(cache);
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT(entry->in_flight_count == 0);
  iree_hal_amdxdna_single_command_cache_entry_deinitialize(cache, entry);
}

// Clones |src| into cache-owned storage. On failure |dst| is left empty.
static iree_status_t iree_hal_amdxdna_single_command_cache_clone_u32_list(
    iree_allocator_t host_allocator, const iree_hal_amdxdna_u32_list_t* src,
    iree_hal_amdxdna_u32_list_t* dst) {
  dst->data = NULL;
  dst->count = 0;
  if (!src || src->count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, src->count * sizeof(*dst->data), (void**)&dst->data));
  memcpy(dst->data, src->data, src->count * sizeof(*dst->data));
  dst->count = src->count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_single_command_cache_copy_signature(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  if (ctrl_word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        cache->host_allocator, ctrl_word_count * sizeof(uint32_t),
        (void**)&entry->ctrl_words));
    memcpy(entry->ctrl_words, ctrl_words, ctrl_word_count * sizeof(uint32_t));
  }
  entry->ctrl_word_count = ctrl_word_count;
  if (binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        cache->host_allocator, binding_count * sizeof(*entry->binding_buffers),
        (void**)&entry->binding_buffers));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        cache->host_allocator,
        binding_count * sizeof(*entry->binding_device_addrs),
        (void**)&entry->binding_device_addrs));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        cache->host_allocator, binding_count * sizeof(*entry->binding_offsets),
        (void**)&entry->binding_offsets));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        cache->host_allocator, binding_count * sizeof(*entry->binding_lengths),
        (void**)&entry->binding_lengths));
    memcpy(entry->binding_buffers, binding_buffers,
           binding_count * sizeof(*entry->binding_buffers));
    memcpy(entry->binding_device_addrs, binding_device_addrs,
           binding_count * sizeof(*entry->binding_device_addrs));
    memcpy(entry->binding_offsets, binding_offsets,
           binding_count * sizeof(*entry->binding_offsets));
    memcpy(entry->binding_lengths, binding_lengths,
           binding_count * sizeof(*entry->binding_lengths));
  }
  entry->binding_count = binding_count;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_single_command_cache_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_device_single_command_cache_t** out_cache) {
  *out_cache = NULL;
  iree_hal_amdxdna_device_single_command_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  memset(cache, 0, sizeof(*cache));
  cache->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

iree_hal_amdxdna_device_single_command_cache_t*
iree_hal_amdxdna_get_single_command_cache(iree_hal_amdxdna_device* device) {
  return device->single_command_cache;
}

void iree_hal_amdxdna_device_destroy_single_command_cache(
    iree_hal_amdxdna_device* device) {
  iree_hal_amdxdna_device_single_command_cache_t* cache =
      device->single_command_cache;
  if (!cache) return;
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_single_command_cache_entry_deinitialize(
        cache, &cache->entries[i]);
  }
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
  device->single_command_cache = NULL;
}

void iree_hal_amdxdna_single_command_cache_invalidate_queue(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue) {
  if (!cache || !queue) return;
  iree_slim_mutex_lock(&cache->mutex);
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_single_command_cache_entry_t* entry = &cache->entries[i];
    if (entry->queue != queue) continue;
    if (entry->in_flight_count != 0) {
      entry->invalidated = true;
    } else {
      iree_hal_amdxdna_single_command_cache_entry_deinitialize(cache, entry);
    }
  }
  iree_slim_mutex_unlock(&cache->mutex);
}

static bool iree_hal_amdxdna_single_command_cache_matches(
    const iree_hal_amdxdna_single_command_cache_entry_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  return cache->command && cache->queue == queue &&
         cache->in_flight_count == 0 && cache->cu_index == cu_index &&
         cache->binding_count == binding_count &&
         iree_hal_amdxdna_u32_span_equal(cache->ctrl_words,
                                         cache->ctrl_word_count, ctrl_words,
                                         ctrl_word_count) &&
         iree_hal_amdxdna_ptr_span_equal(
             (const void* const*)cache->binding_buffers,
             (const void* const*)binding_buffers, binding_count) &&
         iree_hal_amdxdna_u64_span_equal(cache->binding_device_addrs,
                                         binding_device_addrs, binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             cache->binding_offsets, binding_offsets, binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             cache->binding_lengths, binding_lengths, binding_count);
}

static bool iree_hal_amdxdna_single_command_cache_shape_matches(
    const iree_hal_amdxdna_single_command_cache_entry_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    iree_host_size_t ctrl_word_count, const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  return cache->command && cache->queue == queue &&
         cache->in_flight_count == 0 && cache->cu_index == cu_index &&
         cache->src_asm_inst == NULL && cache->src_patches == NULL &&
         cache->ctrl_word_count == ctrl_word_count &&
         cache->binding_count == binding_count &&
         iree_hal_amdxdna_device_size_span_equal(
             cache->binding_offsets, binding_offsets, binding_count) &&
         iree_hal_amdxdna_device_size_span_equal(
             cache->binding_lengths, binding_lengths, binding_count);
}

static bool iree_hal_amdxdna_single_command_cache_descriptor_template_matches(
    const iree_hal_amdxdna_single_command_cache_entry_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches, iree_host_size_t constant_count,
    bool use_native_partial_elf, iree_host_size_t binding_count) {
  // Match by template *content* (against the cache-owned clones), never by the
  // executable-owned pointer identity: those pointers are freed when the
  // recording executable is destroyed and can be reallocated at the same
  // address (ABA), which would otherwise let a different same-shaped kernel
  // falsely reuse this command's on-device control code.
  return cache->command && cache->queue == queue &&
         cache->in_flight_count == 0 && cache->cu_index == cu_index &&
         cache->src_constant_count == constant_count &&
         cache->src_use_native_partial_elf == use_native_partial_elf &&
         cache->ctrl_word_count == (asm_inst ? asm_inst->count : 0) &&
         cache->binding_count == binding_count &&
         iree_hal_amdxdna_u32_span_equal(cache->owned_src_asm_inst.data,
                                         cache->owned_src_asm_inst.count,
                                         asm_inst ? asm_inst->data : NULL,
                                         asm_inst ? asm_inst->count : 0) &&
         iree_hal_amdxdna_u32_span_equal(
             cache->owned_src_patches.data, cache->owned_src_patches.count,
             patches ? patches->data : NULL, patches ? patches->count : 0);
}

iree_status_t iree_hal_amdxdna_update_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count) {
  const bool ctrl_changed = !iree_hal_amdxdna_u32_span_equal(
      entry->ctrl_words, entry->ctrl_word_count, ctrl_words, ctrl_word_count);
  const bool bindings_changed =
      entry->binding_count != binding_count ||
      !iree_hal_amdxdna_ptr_span_equal(
          (const void* const*)entry->binding_buffers,
          (const void* const*)binding_buffers, binding_count) ||
      !iree_hal_amdxdna_device_size_span_equal(
          entry->binding_offsets, binding_offsets, binding_count) ||
      !iree_hal_amdxdna_device_size_span_equal(entry->binding_lengths,
                                               binding_lengths, binding_count);
  if (ctrl_changed) {
    void* ctrl_ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_map(
        entry->ctrl_code_buffer, &ctrl_ptr));
    memcpy(ctrl_ptr, ctrl_words, ctrl_word_count * sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_sync_all(
        entry->ctrl_code_buffer,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_mark_code_dirty(entry->command));
    memcpy(entry->ctrl_words, ctrl_words, ctrl_word_count * sizeof(uint32_t));
  }
  if (bindings_changed) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_reset_bound_buffers(entry->command));
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_bind_buffer(
          entry->command, i + 1, binding_buffers[i], binding_offsets[i],
          binding_lengths[i]));
    }
    memcpy(entry->binding_buffers, binding_buffers,
           binding_count * sizeof(*entry->binding_buffers));
  }
  memcpy(entry->binding_device_addrs, binding_device_addrs,
         binding_count * sizeof(*entry->binding_device_addrs));
  memcpy(entry->binding_offsets, binding_offsets,
         binding_count * sizeof(*entry->binding_offsets));
  memcpy(entry->binding_lengths, binding_lengths,
         binding_count * sizeof(*entry->binding_lengths));
  return iree_ok_status();
}

static bool iree_hal_amdxdna_single_command_cache_partial_elf_shape_matches(
    const iree_hal_amdxdna_single_command_cache_entry_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    iree_host_size_t ctrl_word_count, iree_host_size_t binding_count) {
  return cache->command && cache->ctrl_code_buffer && cache->queue == queue &&
         cache->in_flight_count == 0 && cache->cu_index == cu_index &&
         cache->src_use_native_partial_elf && cache->src_asm_inst != NULL &&
         cache->src_patches != NULL &&
         cache->ctrl_word_count == ctrl_word_count &&
         cache->binding_count == binding_count;
}

iree_status_t
iree_hal_amdxdna_find_single_command_cache_partial_elf_shape_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    iree_host_size_t ctrl_word_count, iree_host_size_t binding_count,
    iree_hal_amdxdna_single_command_cache_entry_t** out_entry) {
  *out_entry = NULL;
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_single_command_cache_entry_t* entry = &cache->entries[i];
    if (iree_hal_amdxdna_single_command_cache_partial_elf_shape_matches(
            entry, queue, cu_index, ctrl_word_count, binding_count)) {
      entry->last_use = ++cache->use_clock;
      *out_entry = entry;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t
iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches, iree_host_size_t constant_count,
    bool use_native_partial_elf, iree_host_size_t binding_count,
    iree_hal_amdxdna_single_command_cache_entry_t** out_entry) {
  *out_entry = NULL;
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_single_command_cache_entry_t* entry = &cache->entries[i];
    if (iree_hal_amdxdna_single_command_cache_descriptor_template_matches(
            entry, queue, cu_index, asm_inst, patches, constant_count,
            use_native_partial_elf, binding_count)) {
      entry->last_use = ++cache->use_clock;
      *out_entry = entry;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_find_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count,
    iree_hal_amdxdna_single_command_cache_entry_t** out_entry) {
  *out_entry = NULL;
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_single_command_cache_entry_t* entry = &cache->entries[i];
    if (iree_hal_amdxdna_single_command_cache_matches(
            entry, queue, cu_index, ctrl_words, ctrl_word_count,
            binding_buffers, binding_device_addrs, binding_offsets,
            binding_lengths, binding_count)) {
      entry->last_use = ++cache->use_clock;
      *out_entry = entry;
      return iree_ok_status();
    }
  }
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_amdxdna_single_command_cache_entry_t* entry = &cache->entries[i];
    if (iree_hal_amdxdna_single_command_cache_shape_matches(
            entry, queue, cu_index, ctrl_word_count, binding_offsets,
            binding_lengths, binding_count)) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_update_single_command_cache_entry(
          cache, entry, ctrl_words, ctrl_word_count, binding_buffers,
          binding_device_addrs, binding_offsets, binding_lengths,
          binding_count));
      entry->last_use = ++cache->use_clock;
      *out_entry = entry;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

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
    iree_hal_amdxdna_native_command_t* command) {
  iree_host_size_t slot = cache->entry_count;
  if (slot >= kAmdxdnaSingleCommandCacheCapacity) {
    slot = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
      if (cache->entries[i].in_flight_count != 0) continue;
      if (slot == IREE_HOST_SIZE_MAX ||
          cache->entries[i].last_use < cache->entries[slot].last_use) {
        slot = i;
      }
    }
    if (slot == IREE_HOST_SIZE_MAX) return NULL;
    iree_hal_amdxdna_single_command_cache_entry_deinitialize(
        cache, &cache->entries[slot]);
  } else {
    ++cache->entry_count;
  }
  iree_hal_amdxdna_single_command_cache_entry_t* entry = &cache->entries[slot];
  entry->queue = queue;
  entry->cu_index = cu_index;
  if (!iree_status_is_ok(iree_hal_amdxdna_single_command_cache_copy_signature(
          cache, entry, ctrl_words, ctrl_word_count, binding_buffers,
          binding_device_addrs, binding_offsets, binding_lengths,
          binding_count))) {
    iree_hal_amdxdna_single_command_cache_entry_deinitialize(cache, entry);
    if (slot == cache->entry_count - 1) --cache->entry_count;
    return NULL;
  }
  entry->ctrl_code_buffer = ctrl_code_buffer;
  entry->command = command;
  entry->last_use = ++cache->use_clock;
  return entry;
}

void iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches, iree_host_size_t constant_count,
    bool use_native_partial_elf, void* ctrl_code_mapped_ptr) {
  IREE_ASSERT_ARGUMENT(cache);
  IREE_ASSERT_ARGUMENT(entry);
  // Clone the executable-owned template lists into cache-owned storage so reuse
  // is matched by content and stays valid after the recording executable dies.
  // Best-effort: if a clone fails the owned list is left empty, so the
  // matcher's content compare simply never hits for this entry (correctness
  // preserved).
  iree_allocator_free(cache->host_allocator, entry->owned_src_asm_inst.data);
  iree_allocator_free(cache->host_allocator, entry->owned_src_patches.data);
  entry->owned_src_asm_inst.data = NULL;
  entry->owned_src_asm_inst.count = 0;
  entry->owned_src_patches.data = NULL;
  entry->owned_src_patches.count = 0;
  iree_status_t clone_status =
      iree_hal_amdxdna_single_command_cache_clone_u32_list(
          cache->host_allocator, asm_inst, &entry->owned_src_asm_inst);
  if (iree_status_is_ok(clone_status)) {
    clone_status = iree_hal_amdxdna_single_command_cache_clone_u32_list(
        cache->host_allocator, patches, &entry->owned_src_patches);
  }
  if (!iree_status_is_ok(clone_status)) {
    iree_status_ignore(clone_status);
    iree_allocator_free(cache->host_allocator, entry->owned_src_asm_inst.data);
    iree_allocator_free(cache->host_allocator, entry->owned_src_patches.data);
    entry->owned_src_asm_inst.data = NULL;
    entry->owned_src_asm_inst.count = 0;
    entry->owned_src_patches.data = NULL;
    entry->owned_src_patches.count = 0;
  }
  entry->src_asm_inst =
      entry->owned_src_asm_inst.data ? &entry->owned_src_asm_inst : NULL;
  entry->src_patches =
      entry->owned_src_patches.data ? &entry->owned_src_patches : NULL;
  entry->src_constant_count = constant_count;
  entry->src_use_native_partial_elf = use_native_partial_elf;
  entry->ctrl_code_mapped_ptr = ctrl_code_mapped_ptr;
}

void iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
    iree_hal_amdxdna_single_command_cache_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(entry);
  ++entry->in_flight_count;
}

void iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry) {
  if (!cache || !entry) return;
  iree_slim_mutex_lock(&cache->mutex);
  IREE_ASSERT(entry->in_flight_count > 0,
              "amdxdna single command cache in-flight underflow");
  --entry->in_flight_count;
  if (entry->in_flight_count == 0 && entry->invalidated) {
    iree_hal_amdxdna_single_command_cache_entry_deinitialize(cache, entry);
  }
  iree_slim_mutex_unlock(&cache->mutex);
}
