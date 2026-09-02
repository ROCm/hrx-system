// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_CHAIN_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_CHAIN_CACHE_H_

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

// Chain (ERT_CMD_CHAIN / runlist) building blocks and the device-global chain
// command cache.
//
// Ownership/threading contract: the chain command cache is intentionally
// DEVICE-GLOBAL, not executable- or command-buffer-owned. A chain spans
// multiple executable entry points (a sequence of dispatches), so no single
// entry point can own it; and its whole value is reuse across one-shot
// command-buffer instances of the same shape, which only a device-scoped cache
// provides. Entries are mutable (control words rewritten / buffers rebound to
// match a freshly recorded command) and are serialized by the cache mutex; a
// cached chain may be reused by one submission lane at a time.

// Bounded device-level cache for realized chain templates. Entry count is only
// a structural guard: each entry may retain many native child command BOs,
// instruction/control BOs, and parent chain BOs. Admission is therefore also
// resource-budgeted in the cache implementation and evicts LRU non-in-flight
// entries before retaining a new template. Child, parent, and instruction
// limits bound distinct native resources until backends expose queryable
// retention limits through native capabilities.
enum { kAmdxdnaChainCommandCacheCapacity = 64 };
enum { kAmdxdnaChainCommandCacheDefaultMaxChildCommands = 896 };
enum { kAmdxdnaChainCommandCacheMaxParentCommands = 96 };
enum { kAmdxdnaChainCommandCacheMaxInstructionBytes = 32 * 1024 * 1024 };

typedef struct iree_hal_amdxdna_device iree_hal_amdxdna_device;

// One chainable command: a host-patched control-code BO + its START_NPU native
// command. The raw native handles are owned by the struct and destroyed by
// iree_hal_amdxdna_chain_cmd_deinitialize().
typedef struct iree_hal_amdxdna_chain_cmd_t {
  iree_hal_amdxdna_native_buffer_t* ctrl_code;
  void* ctrl_code_mapped_ptr;
  iree_hal_amdxdna_native_command_t* command;
  uint32_t* ctrl_words;
  iree_host_size_t ctrl_word_count;
  iree_hal_amdxdna_native_buffer_t** binding_buffers;
  uint64_t* binding_device_addrs;
  iree_device_size_t* binding_offsets;
  iree_device_size_t* binding_lengths;
  iree_host_size_t binding_count;
  // Late-bound inputs (set by accumulation). The ctrl_code BO + native command
  // above are built lazily on a cache miss in flush.
  const iree_hal_amdxdna_u32_list_t* src_asm_inst;
  const iree_hal_amdxdna_u32_list_t* src_patches;
  const iree_hal_amdxdna_write32_constant_patch_list_t* src_constant_patches;
  uint64_t src_executable_identity;
  uint32_t src_entry_point;
  uint32_t src_run_ordinal;
  // Device-cache entries clone executable-owned template lists here before
  // they outlive the executable or one-shot command buffer that recorded them.
  iree_hal_amdxdna_u32_list_t owned_src_asm_inst;
  iree_hal_amdxdna_u32_list_t owned_src_patches;
  iree_hal_amdxdna_write32_constant_patch_list_t owned_src_constant_patches;
  uint8_t* src_constants;
  iree_host_size_t src_constant_count;
  iree_hal_amdxdna_native_c_cu_index_t src_cu_idx;
  bool src_use_native_partial_elf;
  bool built;
  // False when metadata was refreshed for a device-visible hit but the native
  // child command still holds older bound-buffer pointers. Safe until the next
  // packet rewrite, which must rebind before BO-table generation dereferences.
  bool native_bindings_current;
  // False once an in-place rewrite realized control code from a control program
  // other than the one the `src_*` template above was cloned from. The realized
  // ctrl_words are still correct, but the descriptor no longer describes them,
  // so descriptor-keyed reuse (which resubmits the entry unchanged) must not
  // match this command. Content-keyed reuse compares ctrl_words and stays safe.
  bool descriptor_describes_contents;
} iree_hal_amdxdna_chain_cmd_t;

// A contiguous run of dispatches that share one native queue. Flushed as one
// ERT_CMD_CHAIN (split into multiple chains only if the slot count exceeds the
// exec buffer). A chain runs on a single native context, so a queue change
// between dispatches starts a new group.
typedef struct iree_hal_amdxdna_chain_group_t {
  // Retains the native context owning `queue` until the chain flushes.
  iree_hal_amdxdna_native_context_ref_t* context;
  iree_hal_amdxdna_native_queue_t* queue;
  iree_hal_amdxdna_chain_cmd_t* cmds;
  iree_host_size_t cmd_count;
  iree_host_size_t cmd_capacity;
  // Control-packet sequence BOs (reconfig arg buffers): referenced by address
  // from the slots, kept alive + bound for residency until the chain completes.
  iree_hal_amdxdna_native_buffer_t** reconf_buffers;
  iree_host_size_t reconf_buffer_count;
  iree_host_size_t reconf_buffer_capacity;
  // I/O binding refs: their BOs are bound for residency and their exact ranges
  // are synced device->host after the chain completes.
  iree_hal_buffer_ref_t* binding_refs;
  iree_host_size_t binding_ref_count;
  iree_host_size_t binding_ref_capacity;
  // True when the native context already loaded the PDI and dispatches should
  // use module-style START_NPU/PARTIAL_ELF child packets.
  bool native_partial_elf;
} iree_hal_amdxdna_chain_group_t;

// Accumulates sub-commands across dispatches so a whole command buffer flushes
// per native queue. Native backends that support command chaining flush this as
// ERT_CMD_CHAIN.
typedef struct iree_hal_amdxdna_chain_accum_t {
  iree_hal_amdxdna_chain_group_t* groups;
  iree_host_size_t group_count;
  iree_host_size_t group_capacity;
} iree_hal_amdxdna_chain_accum_t;

typedef struct iree_hal_amdxdna_chain_command_cache_entry_t {
  iree_hal_amdxdna_chain_group_t group;
  uint32_t max_slots;
  iree_hal_amdxdna_native_command_t** chains;
  iree_host_size_t chain_count;
  iree_host_size_t chain_capacity;
  uint64_t last_use;
  // Protected by the parent cache mutex. Non-zero means the entry's mutable
  // native child/parent command BOs have been issued and cannot be reused,
  // rewritten, or evicted until native completion.
  uint32_t in_flight_count;
  // Set when this entry's native queue/context is evicted while the chain is
  // still in flight. The completion release discards the entry once
  // in_flight_count reaches zero.
  bool invalidated;
} iree_hal_amdxdna_chain_command_cache_entry_t;

typedef struct iree_hal_amdxdna_device_chain_command_cache_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_chain_command_cache_entry_t
      entries[kAmdxdnaChainCommandCacheCapacity];
  iree_host_size_t max_child_commands;
  iree_host_size_t entry_count;
  uint64_t use_clock;
} iree_hal_amdxdna_device_chain_command_cache_t;

void iree_hal_amdxdna_chain_cmd_initialize(iree_hal_amdxdna_chain_cmd_t* cmd);
void iree_hal_amdxdna_chain_cmd_deinitialize(iree_allocator_t host_allocator,
                                             iree_hal_amdxdna_chain_cmd_t* cmd);
void iree_hal_amdxdna_chain_cmd_move(iree_hal_amdxdna_chain_cmd_t* dst,
                                     iree_hal_amdxdna_chain_cmd_t* src);
iree_status_t iree_hal_amdxdna_chain_cmd_set_signature(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd,
    const uint32_t* ctrl_words, iree_host_size_t ctrl_word_count,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count);
iree_status_t iree_hal_amdxdna_chain_cmd_set_deferred_descriptor(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches,
    const iree_hal_amdxdna_write32_constant_patch_list_t* constant_patches,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    iree_hal_amdxdna_native_buffer_t* const* binding_buffers,
    const uint64_t* binding_device_addrs,
    const iree_device_size_t* binding_offsets,
    const iree_device_size_t* binding_lengths, iree_host_size_t binding_count);
iree_status_t iree_hal_amdxdna_chain_cmd_make_deferred_lists_owned(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd);
iree_status_t iree_hal_amdxdna_chain_cmd_replace_deferred_lists_owned(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_cmd_t* cmd,
    const iree_hal_amdxdna_chain_cmd_t* fresh);
void iree_hal_amdxdna_chain_group_initialize(
    iree_hal_amdxdna_chain_group_t* group);
void iree_hal_amdxdna_chain_group_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group);
void iree_hal_amdxdna_chain_group_move(iree_hal_amdxdna_chain_group_t* dst,
                                       iree_hal_amdxdna_chain_group_t* src);
iree_status_t iree_hal_amdxdna_chain_group_append_cmd_move(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group,
    iree_hal_amdxdna_chain_cmd_t* cmd);
iree_status_t iree_hal_amdxdna_chain_group_append_reconf_buffer(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group,
    iree_hal_amdxdna_native_buffer_t* buffer);
iree_status_t iree_hal_amdxdna_chain_group_take_reconf_buffers(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* dst,
    iree_hal_amdxdna_chain_group_t* src);
iree_status_t iree_hal_amdxdna_chain_group_append_binding_ref_unique(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* group,
    iree_hal_buffer_ref_t binding_ref);
iree_status_t iree_hal_amdxdna_chain_group_take_cmds(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* dst,
    iree_hal_amdxdna_chain_group_t* src);
iree_status_t iree_hal_amdxdna_chain_group_set_binding_refs(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_group_t* dst,
    const iree_hal_amdxdna_chain_group_t* src);
bool iree_hal_amdxdna_chain_group_binding_refs_match(
    const iree_hal_amdxdna_chain_group_t* lhs,
    const iree_hal_amdxdna_chain_group_t* rhs);
bool iree_hal_amdxdna_chain_group_reconf_buffers_match(
    const iree_hal_amdxdna_chain_group_t* lhs,
    const iree_hal_amdxdna_chain_group_t* rhs);

void iree_hal_amdxdna_chain_accum_initialize(
    iree_hal_amdxdna_chain_accum_t* accum);
void iree_hal_amdxdna_chain_accum_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_accum_t* accum);
void iree_hal_amdxdna_chain_accum_clear(iree_allocator_t host_allocator,
                                        iree_hal_amdxdna_chain_accum_t* accum);
iree_status_t iree_hal_amdxdna_chain_accum_append_group(
    iree_allocator_t host_allocator, iree_hal_amdxdna_chain_accum_t* accum,
    iree_hal_amdxdna_chain_group_t** out_group);

iree_status_t iree_hal_amdxdna_chain_command_cache_create(
    iree_allocator_t host_allocator, iree_host_size_t max_child_commands,
    iree_hal_amdxdna_device_chain_command_cache_t** out_cache);

iree_hal_amdxdna_device_chain_command_cache_t*
iree_hal_amdxdna_get_chain_command_cache(iree_hal_amdxdna_device* device);

void iree_hal_amdxdna_chain_command_cache_invalidate_queue(
    iree_hal_amdxdna_device_chain_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue);

bool iree_hal_amdxdna_chain_cmd_descriptor_matches(
    const iree_hal_amdxdna_chain_cmd_t* lhs,
    const iree_hal_amdxdna_chain_cmd_t* rhs);

bool iree_hal_amdxdna_chain_cmd_descriptor_template_matches(
    const iree_hal_amdxdna_chain_cmd_t* lhs,
    const iree_hal_amdxdna_chain_cmd_t* rhs);

bool iree_hal_amdxdna_chain_command_cache_device_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots);

bool iree_hal_amdxdna_chain_command_cache_shape_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots);

bool iree_hal_amdxdna_chain_command_cache_descriptor_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots);

bool iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
    const iree_hal_amdxdna_chain_command_cache_entry_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots);

bool iree_hal_amdxdna_chain_command_cache_trim_for_group(
    iree_hal_amdxdna_device_chain_command_cache_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots);

iree_status_t iree_hal_amdxdna_update_cached_chain_cmd(
    iree_hal_amdxdna_chain_cmd_t* cached,
    const iree_hal_amdxdna_chain_cmd_t* fresh, bool* out_packet_changed,
    bool* out_code_changed, bool* out_device_bindings_changed,
    bool* out_rebound);

iree_status_t iree_hal_amdxdna_chain_command_cache_entry_append_chain(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry,
    iree_hal_amdxdna_native_command_t* chain);
void iree_hal_amdxdna_chain_command_cache_entry_clear_chains(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry);
iree_hal_amdxdna_chain_command_cache_entry_t*
iree_hal_amdxdna_chain_command_cache_allocate_entry(
    iree_hal_amdxdna_device_chain_command_cache_t* cache,
    const iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots);

void iree_hal_amdxdna_chain_command_cache_entry_acquire_in_flight(
    iree_hal_amdxdna_chain_command_cache_entry_t* entry);

void iree_hal_amdxdna_chain_command_cache_entry_release_in_flight(
    iree_hal_amdxdna_device_chain_command_cache_t* cache,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry);

// Discards a non-in-flight entry after a failed in-place rewrite. The caller
// must hold cache->mutex.
void iree_hal_amdxdna_chain_command_cache_entry_discard(
    iree_hal_amdxdna_device_chain_command_cache_t* cache,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry);

// Always re-map a cached control-code BO before rewriting it.
//
// Windows instruction BOs start as deferred host storage. The first submit
// materializes them into a KMT mapping and frees that temporary storage.
// A `ctrl_code_mapped_ptr` captured before materialize then dangles, so later
// in-place rewrites land in freed memory while the device keeps executing the
// first-token image. The RTP context-length program is the representative
// steady-state case: same 28-word shape, only a WRITE32 immediate changes, and
// a stale mapping freezes the attended length at the first decode step.
static inline iree_status_t iree_hal_amdxdna_remap_ctrl_code(
    iree_hal_amdxdna_native_buffer_t* buffer, void** io_mapped_ptr) {
  void* mapped_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_buffer_c_map(buffer, &mapped_ptr));
  *io_mapped_ptr = mapped_ptr;
  return iree_ok_status();
}

// Instruction BOs are Lock2-mapped. A process write barrier does not evict
// dirty CPU cache lines from that mapping, so Path-B publication (which copies
// from `buffer.cpu_ptr` into the command aperture) can restage the
// pre-rewrite image unless the rewritten range is published first.
static inline iree_status_t iree_hal_amdxdna_publish_ctrl_code(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_c_sync_all(
      buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_CHAIN_CACHE_H_
