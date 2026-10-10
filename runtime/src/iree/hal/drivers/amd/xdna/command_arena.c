// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/command_arena.h"

#include "iree/base/threading/mutex.h"

// Default persistent extent acquired by a growable arena. Larger allocation
// requests acquire a directly sized slab instead.
#define IREE_HAL_AMD_XDNA_COMMAND_SLAB_BYTE_LENGTH (8 * 1024 * 1024)

struct iree_hal_amd_xdna_command_slab_t {
  // Next persistent slab in arena ownership order.
  iree_hal_amd_xdna_command_slab_t* next;
  // One native allocation containing every range in this slab.
  iree_hal_amd_xdna_memory_t memory;
  // First managed byte after aligning the native firmware address.
  uint64_t memory_byte_offset;
  // Offset allocator over the aligned usable instruction range.
  iree_hal_memory_tlsf_t tlsf;
  // True after TLSF initialization succeeds.
  bool tlsf_initialized;
};

struct iree_hal_amd_xdna_command_arena_t {
  // Borrowed native context dominating the arena and all allocations.
  iree_hal_amd_xdna_context_t* context;
  // Allocator owning this structure and TLSF metadata.
  iree_allocator_t host_allocator;
  // First persistent native slab, or NULL before growable-arena use.
  iree_hal_amd_xdna_command_slab_t* slabs;
  // Preferred native allocation length for growable slabs.
  uint64_t preferred_slab_byte_length;
  // True when the profile permits exactly one complete native aperture.
  bool indivisible;
  // Serializes slab growth and TLSF mutation during executable lifecycle.
  iree_slim_mutex_t mutex;
};

static void iree_hal_amd_xdna_command_slab_destroy(
    iree_hal_amd_xdna_command_arena_t* arena,
    iree_hal_amd_xdna_command_slab_t* slab) {
  if (!slab) {
    return;
  }
  if (slab->tlsf_initialized) {
    iree_hal_memory_tlsf_deinitialize(&slab->tlsf);
  }
  iree_hal_amd_xdna_memory_deinitialize(arena->context, &slab->memory);
  iree_allocator_free(arena->host_allocator, slab);
}

// Acquires one native slab with at least |minimum_managed_byte_length| usable
// aligned bytes. Called under the arena mutex during cold growth.
static iree_status_t iree_hal_amd_xdna_command_slab_create(
    iree_hal_amd_xdna_command_arena_t* arena,
    uint64_t minimum_managed_byte_length,
    iree_hal_amd_xdna_command_slab_t** out_slab) {
  *out_slab = NULL;
  const iree_hal_amd_xdna_memory_source_t* source =
      &arena->context->command_source;
  const uint64_t alignment = arena->context->target.instruction_alignment;
  const uint64_t maximum_byte_length =
      source->profile.allocation.maximum_byte_length;
  uint64_t requested_byte_length = maximum_byte_length;
  if (!arena->indivisible) {
    uint64_t padded_byte_length = 0;
    if (!iree_checked_add_u64(minimum_managed_byte_length, alignment - 1,
                              &padded_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA command slab extent overflows");
    }
    requested_byte_length =
        iree_max(arena->preferred_slab_byte_length, padded_byte_length);
    requested_byte_length =
        iree_min(requested_byte_length, maximum_byte_length);
  }

  iree_hal_amd_xdna_command_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(arena->host_allocator,
                                             sizeof(*slab), (void**)&slab));
  iree_status_t status = iree_hal_amd_xdna_memory_allocate(
      arena->context, source, requested_byte_length,
      source->profile.allocation.minimum_alignment, &slab->memory);
  if (iree_status_is_ok(status)) {
    slab->memory_byte_offset =
        (alignment - slab->memory.device_address % alignment) % alignment;
    if (slab->memory_byte_offset >= slab->memory.contents.data_length ||
        slab->memory.contents.data_length - 1 >
            UINT64_MAX - slab->memory.device_address) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA instruction slab has no usable firmware address range");
    }
  }
  uint64_t managed_byte_length = 0;
  if (iree_status_is_ok(status)) {
    managed_byte_length =
        (slab->memory.contents.data_length - slab->memory_byte_offset) &
        ~(alignment - 1);
    if (managed_byte_length < minimum_managed_byte_length) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "XDNA instruction slab provides only %" PRIu64
                           " aligned bytes; %" PRIu64 " required",
                           managed_byte_length, minimum_managed_byte_length);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_memory_tlsf_options_t options = {
        .range_length = managed_byte_length,
        .alignment = alignment,
        .frontier_capacity = 1,
    };
    status = iree_hal_memory_tlsf_initialize(options, arena->host_allocator,
                                             &slab->tlsf);
    slab->tlsf_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    *out_slab = slab;
  } else {
    iree_hal_amd_xdna_command_slab_destroy(arena, slab);
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_command_arena_create(
    iree_hal_amd_xdna_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_xdna_command_arena_t** out_arena) {
  *out_arena = NULL;
  iree_hal_amd_xdna_command_arena_t* arena = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*arena), (void**)&arena));
  arena->context = context;
  arena->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&arena->mutex);

  const iree_hal_amd_xdna_memory_source_t* source = &context->command_source;
  const uint64_t maximum_byte_length =
      source->profile.allocation.maximum_byte_length;
  arena->preferred_slab_byte_length =
      iree_min(maximum_byte_length,
               (uint64_t)IREE_HAL_AMD_XDNA_COMMAND_SLAB_BYTE_LENGTH);
  // Equal maximum and granularity means that the only legal creation request
  // consumes the complete native aperture. Windows uses this contract and
  // permits no replacement allocation during the context lifetime.
  arena->indivisible =
      source->profile.allocation.byte_length_granularity == maximum_byte_length;
  iree_status_t status = iree_ok_status();
  if (arena->indivisible) {
    status = iree_hal_amd_xdna_command_slab_create(
        arena, context->target.instruction_alignment, &arena->slabs);
  }
  if (iree_status_is_ok(status)) {
    *out_arena = arena;
  } else {
    iree_hal_amd_xdna_command_arena_destroy(arena);
  }
  return status;
}

void iree_hal_amd_xdna_command_arena_destroy(
    iree_hal_amd_xdna_command_arena_t* arena) {
  if (!arena) {
    return;
  }
  iree_hal_amd_xdna_command_slab_t* slab = arena->slabs;
  while (slab) {
    iree_hal_amd_xdna_command_slab_t* next = slab->next;
    iree_hal_amd_xdna_command_slab_destroy(arena, slab);
    slab = next;
  }
  iree_slim_mutex_deinitialize(&arena->mutex);
  iree_allocator_free(arena->host_allocator, arena);
}

iree_status_t iree_hal_amd_xdna_command_arena_allocate(
    iree_hal_amd_xdna_command_arena_t* arena, uint64_t byte_length,
    uint64_t alignment,
    iree_hal_amd_xdna_command_allocation_t* out_allocation) {
  *out_allocation = (iree_hal_amd_xdna_command_allocation_t){0};
  const uint64_t arena_alignment = arena->context->target.instruction_alignment;
  if (alignment < arena_alignment ||
      !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid XDNA command alignment %" PRIu64,
                            alignment);
  }
  uint64_t reserved_length = 0;
  if (!iree_checked_add_u64(byte_length, alignment - arena_alignment,
                            &reserved_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA command allocation extent overflows");
  }
  iree_hal_amd_xdna_command_slab_t* selected_slab = NULL;
  iree_hal_memory_tlsf_allocation_t block = {0};
  iree_slim_mutex_lock(&arena->mutex);
  iree_status_t status = iree_ok_status();
  for (iree_hal_amd_xdna_command_slab_t* slab = arena->slabs;
       slab && iree_status_is_ok(status); slab = slab->next) {
    iree_hal_memory_tlsf_allocate_result_t result =
        IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED;
    status = iree_hal_memory_tlsf_try_allocate(&slab->tlsf, reserved_length,
                                               &block, &result);
    if (result == IREE_HAL_MEMORY_TLSF_ALLOCATE_OK) {
      selected_slab = slab;
      break;
    }
  }
  if (iree_status_is_ok(status) && !selected_slab && !arena->indivisible) {
    status = iree_hal_amd_xdna_command_slab_create(arena, reserved_length,
                                                   &selected_slab);
    if (iree_status_is_ok(status)) {
      selected_slab->next = arena->slabs;
      arena->slabs = selected_slab;
      status = iree_hal_memory_tlsf_allocate(&selected_slab->tlsf,
                                             reserved_length, &block);
    }
  }
  if (iree_status_is_ok(status) && !selected_slab) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "XDNA instruction aperture is full");
  }
  iree_slim_mutex_unlock(&arena->mutex);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  const uint64_t block_byte_offset =
      selected_slab->memory_byte_offset + block.offset;
  const uint64_t block_device_address =
      selected_slab->memory.device_address + block_byte_offset;
  const uint64_t alignment_padding =
      (alignment - block_device_address % alignment) % alignment;
  const uint64_t memory_byte_offset = block_byte_offset + alignment_padding;
  *out_allocation = (iree_hal_amd_xdna_command_allocation_t){
      .memory = selected_slab->memory.handle,
      .mapping = selected_slab->memory.mapping,
      .contents = iree_make_byte_span(
          selected_slab->memory.contents.data + memory_byte_offset,
          byte_length),
      .memory_byte_offset = memory_byte_offset,
      .device_address =
          selected_slab->memory.device_address + memory_byte_offset,
      .slab = selected_slab,
      .block_index = block.block_index,
  };
  return iree_ok_status();
}

void iree_hal_amd_xdna_command_arena_release(
    iree_hal_amd_xdna_command_arena_t* arena,
    iree_hal_amd_xdna_command_allocation_t* allocation) {
  if (!allocation->memory) {
    return;
  }
  iree_slim_mutex_lock(&arena->mutex);
  iree_hal_memory_tlsf_free(&allocation->slab->tlsf, allocation->block_index,
                            NULL);
  iree_slim_mutex_unlock(&arena->mutex);
  *allocation = (iree_hal_amd_xdna_command_allocation_t){0};
}
