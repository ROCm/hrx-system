// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#include "iree/hal/drivers/amd/status.h"

static void iree_hal_amd_gpu_pm4_program_instance_free(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  if (!instance) {
    return;
  }
  iree_hal_amd_fabric_memory_deinitialize(
      program->context->api, program->context->event_sink, &instance->memory);
  iree_allocator_free(program->host_allocator, instance);
}

static iree_status_t iree_hal_amd_gpu_pm4_program_instance_allocate(
    iree_hal_amd_gpu_pm4_program_t* program, uint32_t ordinal,
    uint64_t byte_length, bool upload,
    iree_hal_amd_gpu_pm4_program_instance_t** out_instance) {
  *out_instance = NULL;
  iree_hal_amd_gpu_pm4_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      program->host_allocator, sizeof(*instance), (void**)&instance));
  instance->ordinal = ordinal;
  iree_hal_amd_fabric_memory_options_t options = program->memory;
  if (!upload) {
    options.host_access = 0;
  }
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      program->context->api, &options, byte_length, program->alignment,
      program->context->event_sink, &instance->memory);
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        program->context->api->memory_query_address(instance->memory.handle, 0,
                                                    AMDF_MEMORY_ADDRESS_GPU,
                                                    &instance->address),
        "memory_query_address(resident program)");
  }
  if (iree_status_is_ok(status)) {
    *out_instance = instance;
  } else {
    iree_hal_amd_gpu_pm4_program_instance_free(program, instance);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_pm4_program_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* memory, uint32_t queue_count,
    uint32_t alignment, iree_const_byte_span_t contents, uint32_t fixup_count,
    const iree_hal_amd_gpu_pm4_fixup_t* fixups, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_t* out_program) {
  *out_program = (iree_hal_amd_gpu_pm4_program_t){
      .context = context,
      .host_allocator = host_allocator,
      .memory = *memory,
      .capacity = queue_count,
      .byte_length = (uint32_t)contents.data_length,
      .alignment = alignment,
      .fixup_count = fixup_count,
  };
  iree_slim_mutex_initialize(&out_program->mutex);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, 2 * queue_count * sizeof(*out_program->instances),
      (void**)&out_program->instances));
  out_program->preferred = out_program->instances + queue_count;
  const uint64_t fixup_offset = iree_align_uint64(contents.data_length, 16);
  const uint64_t byte_length =
      fixup_offset + (uint64_t)fixup_count * sizeof(*fixups);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_instance_allocate(
      out_program, 0, byte_length, true, &out_program->instances[0]));
  iree_hal_amd_gpu_pm4_program_instance_t* initial = out_program->instances[0];
  memcpy(initial->memory.contents.data, contents.data, contents.data_length);
  if (fixup_count) {
    memcpy(initial->memory.contents.data + fixup_offset, fixups,
           fixup_count * sizeof(*fixups));
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(initial->memory.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, byte_length),
      "host_mapping_cache_control(resident program)"));
  initial->initialized = true;
  for (uint32_t i = 0; i < queue_count; ++i) {
    out_program->preferred[i] = initial;
  }
  out_program->fixup_address = initial->address + fixup_offset;
  return iree_ok_status();
}

static bool iree_hal_amd_gpu_pm4_program_instance_available(
    const iree_hal_amd_gpu_pm4_program_instance_t* instance,
    const iree_async_frontier_t* frontier) {
  if (instance->reserved) {
    return false;
  }
  if (!instance->use_count) {
    return true;
  }
  // Frontiers are sorted; only the last writer is needed because every reuse
  // is ordered after that writer. Clone reads of the initial instance do not
  // constrain its mutable words and cannot release its permanent ownership.
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (frontier->entries[i].axis == instance->last_use.axis) {
      return frontier->entries[i].epoch >= instance->last_use.epoch;
    }
  }
  return false;
}

iree_hal_amd_gpu_pm4_program_instance_t*
iree_hal_amd_gpu_pm4_program_try_acquire(
    iree_hal_amd_gpu_pm4_program_t* program, uint32_t queue_ordinal,
    const iree_async_frontier_t* frontier) {
  iree_slim_mutex_lock(&program->mutex);
  iree_hal_amd_gpu_pm4_program_instance_t* instance =
      program->preferred[queue_ordinal];
  if (!instance ||
      !iree_hal_amd_gpu_pm4_program_instance_available(instance, frontier)) {
    instance = NULL;
    for (uint32_t i = 0; i < program->capacity && !instance; ++i) {
      iree_hal_amd_gpu_pm4_program_instance_t* candidate =
          program->instances[i];
      if (candidate && iree_hal_amd_gpu_pm4_program_instance_available(
                           candidate, frontier)) {
        instance = candidate;
      }
    }
  }
  if (instance) {
    ++instance->use_count;
    instance->reserved = true;
    program->preferred[queue_ordinal] = instance;
  }
  iree_slim_mutex_unlock(&program->mutex);
  return instance;
}

iree_status_t iree_hal_amd_gpu_pm4_program_grow(
    iree_hal_amd_gpu_pm4_program_t* program) {
  uint32_t ordinal = program->capacity;
  bool idle = false;
  iree_slim_mutex_lock(&program->mutex);
  for (uint32_t i = 0; i < program->capacity; ++i) {
    if (program->instances[i]) {
      idle |= !program->instances[i]->use_count;
    } else if (ordinal == program->capacity) {
      ordinal = i;
    }
  }
  iree_slim_mutex_unlock(&program->mutex);
  if (idle) {
    return iree_ok_status();
  }
  // One unpublished use per queue, with every published FIFO prefix in its
  // frontier, bounds independent instances by the provisioned queue count.
  IREE_ASSERT_LT(ordinal, program->capacity);
  iree_hal_amd_gpu_pm4_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_instance_allocate(
      program, ordinal, program->byte_length, false, &instance));
  iree_slim_mutex_lock(&program->mutex);
  program->instances[ordinal] = instance;
  iree_slim_mutex_unlock(&program->mutex);
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_program_publish(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance,
    iree_async_frontier_entry_t next_use) {
  iree_slim_mutex_lock(&program->mutex);
  instance->initialized = true;
  instance->last_use = next_use;
  instance->reserved = false;
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_cancel(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  instance->reserved = false;
  --instance->use_count;
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_release(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  --instance->use_count;
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_trim(
    iree_hal_amd_gpu_pm4_program_t* program) {
  for (uint32_t i = 1; i < program->capacity; ++i) {
    iree_slim_mutex_lock(&program->mutex);
    iree_hal_amd_gpu_pm4_program_instance_t* instance = program->instances[i];
    if (instance && !instance->use_count) {
      program->instances[i] = NULL;
      for (uint32_t q = 0; q < program->capacity; ++q) {
        if (program->preferred[q] == instance) {
          program->preferred[q] = program->instances[0];
        }
      }
    } else {
      instance = NULL;
    }
    iree_slim_mutex_unlock(&program->mutex);
    iree_hal_amd_gpu_pm4_program_instance_free(program, instance);
  }
}

void iree_hal_amd_gpu_pm4_program_deinitialize(
    iree_hal_amd_gpu_pm4_program_t* program) {
  if (!program->context) {
    return;
  }
  if (program->instances) {
    for (uint32_t i = 0; i < program->capacity; ++i) {
      iree_hal_amd_gpu_pm4_program_instance_free(program,
                                                 program->instances[i]);
    }
    iree_allocator_free(program->host_allocator, program->instances);
  }
  iree_slim_mutex_deinitialize(&program->mutex);
}
