// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/recording.h"
#include "iree/hal/utils/resource_set.h"

enum {
  IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE = 0,
  IREE_HAL_AMD_GPU_PM4_VALUE_DISPATCH_ID = 1,
  IREE_HAL_AMD_GPU_PM4_VALUE_CONTINUATION = 2,
  IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT = 3,
  IREE_HAL_AMD_GPU_PM4_INPUT_DIRECT = 0,
  IREE_HAL_AMD_GPU_PM4_INPUT_BINDING = 1,
  IREE_HAL_AMD_GPU_PM4_INPUT_IMAGE = 2,
  IREE_HAL_AMD_GPU_PM4_INPUT_LENGTH = 3,
  IREE_HAL_AMD_GPU_PM4_SITE_QUEUE = 0,
  IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM = 1,
  IREE_HAL_AMD_GPU_PM4_BINDING_LENGTH = 1u << 2,
};

// A deduplicated invocation input. Recording aggregates alignment and overflow
// requirements; publication reads each source exactly once. Byte offsets belong
// to immutable GPU fixups, not to a per-command host relocation list.
typedef struct iree_hal_amd_gpu_pm4_command_input_t {
  // Buffer/image pointer, or binding ordinal, selected by kind.
  uintptr_t key;
  // Largest displacement used by any fixup consuming this address.
  uint64_t maximum_offset;
  // Smallest whole-range displacement, or UINT64_MAX for a required address.
  uint64_t minimum_offset;
  // Direct, binding, image-base, or binding-length input.
  uint16_t kind;
  // Queue/program native site, ignored for image and length inputs.
  uint16_t site;
  // Prepared direct-buffer native slot, unused for other kinds.
  iree_hal_buffer_native_binding_slot_t slot;
  // Strongest power-of-two alignment required of this base plus a displacement.
  uint32_t alignment;
  // Required low bits of the base address under alignment - 1.
  uint32_t residue;
} iree_hal_amd_gpu_pm4_command_input_t;

typedef struct iree_hal_amd_gpu_pm4_binding_requirement_t {
  // Minimum captured extent across all uses, including private native ABIs.
  uint64_t minimum_length;
  // Length minus the root binding's length, modulo 2^64.
  uint64_t difference;
  // Union root for whole-range copy equalities. Slot zero is constant zero;
  // public bindings occupy indices one through binding_capacity.
  uint32_t parent;
  // Required native sites and whether the logical length is consumed.
  uint32_t sites;
} iree_hal_amd_gpu_pm4_binding_requirement_t;

typedef struct iree_hal_amd_gpu_pm4_command_buffer_t {
  // Public resource identity and optional external recording validation.
  iree_hal_command_buffer_t base;
  // Allocator for this object, inputs, and temporary recording bytes.
  iree_allocator_t host_allocator;
  // Borrowed native owner dominated by the retained family device.
  iree_hal_amd_gpu_pm4_program_pool_t* pool;
  // Single-shot service cleanup entry embedded in this resource.
  iree_hal_memory_maintenance_entry_t cleanup;
  // Immutable resident execution facts after end.
  iree_hal_amd_gpu_pm4_command_program_t program;
  // Temporary native bytes and fixups, released after sealing.
  iree_hal_amd_gpu_pm4_recording_t recording;
  // Dense unique inputs, retained after sealing for direct publication.
  iree_hal_amd_gpu_pm4_command_input_t* inputs;
  // Number of initialized input descriptors.
  uint32_t input_count;
  // Allocated descriptor capacity.
  uint32_t input_capacity;
  // Recording-only open-addressed index, with input ordinal plus one in slots.
  uint32_t* input_index;
  // Power-of-two slot count, zero before the first input.
  uint32_t input_index_capacity;
  // Inline native binding and length-equality requirements.
  iree_hal_amd_gpu_pm4_binding_requirement_t* requirements;
  // Device-local LDS capacity for public dispatch admission.
  uint32_t maximum_group_segment_byte_length;
  // Captured physical cache behavior; no topology is retained.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  // First failed native recording mutation; prevents sealing partial packets.
  iree_status_t recording_status;
  // Adjacent public barriers are combined before the next native command.
  iree_hal_amd_gpu_pm4_barrier_t pending_barrier;
  // Blocks backing direct-resource ownership.
  iree_arena_block_pool_t resource_blocks;
  // Retained buffers/executables, or NULL for explicitly unretained recordings.
  iree_hal_resource_set_t* resources;
} iree_hal_amd_gpu_pm4_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_pm4_command_buffer_vtable;

static void iree_hal_amd_gpu_pm4_command_buffer_cleanup(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer =
      (void*)((uint8_t*)entry -
              offsetof(iree_hal_amd_gpu_pm4_command_buffer_t, cleanup));
  iree_hal_device_t* device = buffer->base.queue_family->device;
  iree_hal_amd_gpu_pm4_program_deinitialize(&buffer->program.resident);
  iree_allocator_free(buffer->host_allocator, buffer);
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  // Last HAL ownership proves all native uses joined. Drop caller resource
  // references now so an immediately following pool release sees no live
  // children; only the driver's own IB storage waits for cold service cleanup.
  iree_status_free(buffer->recording_status);
  iree_hal_amd_gpu_pm4_recording_deinitialize(&buffer->recording);
  iree_hal_resource_set_free(buffer->resources);
  iree_arena_block_pool_deinitialize(&buffer->resource_blocks);
  iree_allocator_free(buffer->host_allocator, buffer->inputs);
  iree_allocator_free(buffer->host_allocator, buffer->input_index);
  buffer->cleanup.fn = iree_hal_amd_gpu_pm4_command_buffer_cleanup;
  iree_hal_memory_maintenance_enqueue(buffer->pool->service, &buffer->cleanup);
}

iree_status_t iree_hal_amd_gpu_pm4_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_pm4_program_pool_t* pool,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity,
    uint32_t maximum_group_segment_byte_length, uint32_t tsan_shadow_slot_count,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_allocator_t host_allocator, iree_hal_command_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (binding_capacity >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident binding table exceeds its index space");
  }
  iree_host_size_t size = 0, requirements_offset = 0, validation_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_pm4_command_buffer_t), &size,
      IREE_STRUCT_FIELD(binding_capacity + 1,
                        iree_hal_amd_gpu_pm4_binding_requirement_t,
                        &requirements_offset),
      IREE_STRUCT_FIELD(
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
          uint8_t, &validation_offset)));
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&buffer));
  iree_hal_command_buffer_initialize(
      family, mode, categories, binding_capacity,
      (uint8_t*)buffer + validation_offset,
      &iree_hal_amd_gpu_pm4_command_buffer_vtable, &buffer->base);
  iree_hal_device_retain(family->device);
  buffer->host_allocator = host_allocator;
  buffer->pool = pool;
  buffer->maximum_group_segment_byte_length = maximum_group_segment_byte_length;
  buffer->profile = *profile;
  buffer->program.tsan_shadow_slot_count = tsan_shadow_slot_count;
  buffer->requirements = (void*)((uint8_t*)buffer + requirements_offset);
  for (uint32_t i = 0; i <= binding_capacity; ++i) {
    buffer->requirements[i].parent = i;
  }
  iree_hal_amd_gpu_pm4_recording_initialize(
      host_allocator, IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, &buffer->recording);
  iree_arena_block_pool_initialize(4096, host_allocator,
                                   &buffer->resource_blocks);
  iree_status_t status = iree_ok_status();
  if (!iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(&buffer->resource_blocks,
                                            &buffer->resources);
  }
  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    iree_hal_command_buffer_release(&buffer->base);
  }
  return status;
}

iree_host_size_t iree_hal_amd_gpu_pm4_command_buffer_operation_count(
    iree_hal_command_buffer_t* base) {
  return ((iree_hal_amd_gpu_pm4_command_buffer_t*)base)
      ->program.operation_count;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_qualify_binding(
    const iree_hal_queue_family_t* family, iree_hal_buffer_t* buffer,
    uint32_t site, iree_hal_buffer_native_binding_slot_t* out_slot) {
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  const uint64_t scope_id = (uint64_t)family->memory.queue_scope_id + site;
  if (!contract || !family->memory.domain ||
      contract->domain != family->memory.domain ||
      scope_id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "recorded buffer has no native site contract");
  }
  const uint16_t index =
      contract->scopes[scope_id]
          .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS];
  if (index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "recorded buffer has no prepared GPU address slot");
  }
  *out_slot = (iree_hal_buffer_native_binding_slot_t){
      .index = index, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  return iree_ok_status();
}

static uint32_t iree_hal_amd_gpu_pm4_command_input_hash(uintptr_t key,
                                                        uint16_t kind,
                                                        uint16_t site) {
  uint64_t value =
      (uint64_t)key ^ ((uint64_t)kind << 48) ^ ((uint64_t)site << 56);
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return (uint32_t)(value ^ (value >> 31));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_input(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_amd_gpu_pm4_command_input_t input, uint32_t* out_value) {
  if ((buffer->input_count + 1) * 2 >= buffer->input_index_capacity) {
    const uint32_t capacity = iree_max(buffer->input_index_capacity * 2, 32u);
    uint32_t* index = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        buffer->host_allocator, capacity * sizeof(*index), (void**)&index));
    for (uint32_t i = 0; i < buffer->input_count; ++i) {
      const iree_hal_amd_gpu_pm4_command_input_t* value = &buffer->inputs[i];
      uint32_t position = iree_hal_amd_gpu_pm4_command_input_hash(
                              value->key, value->kind, value->site) &
                          (capacity - 1);
      while (index[position]) {
        position = (position + 1) & (capacity - 1);
      }
      index[position] = i + 1;
    }
    iree_allocator_free(buffer->host_allocator, buffer->input_index);
    buffer->input_index = index;
    buffer->input_index_capacity = capacity;
  }
  uint32_t position = iree_hal_amd_gpu_pm4_command_input_hash(
                          input.key, input.kind, input.site) &
                      (buffer->input_index_capacity - 1);
  uint32_t ordinal = buffer->input_count;
  while (buffer->input_index[position]) {
    const uint32_t candidate = buffer->input_index[position] - 1;
    const iree_hal_amd_gpu_pm4_command_input_t* previous =
        &buffer->inputs[candidate];
    if (previous->key == input.key && previous->kind == input.kind &&
        previous->site == input.site) {
      ordinal = candidate;
      break;
    }
    position = (position + 1) & (buffer->input_index_capacity - 1);
  }
  if (ordinal == buffer->input_count) {
    if (ordinal >= 0x1000000u - IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "resident program has too many distinct inputs");
    }
    if (ordinal == buffer->input_capacity) {
      const uint32_t capacity = iree_max(buffer->input_capacity * 2, 16u);
      IREE_RETURN_IF_ERROR(iree_allocator_realloc(
          buffer->host_allocator, capacity * sizeof(*buffer->inputs),
          (void**)&buffer->inputs));
      buffer->input_capacity = capacity;
    }
    buffer->inputs[ordinal] = input;
    buffer->input_index[position] = ordinal + 1;
    ++buffer->input_count;
  } else {
    iree_hal_amd_gpu_pm4_command_input_t* previous = &buffer->inputs[ordinal];
    if (((previous->residue ^ input.residue) &
         (iree_min(previous->alignment, input.alignment) - 1)) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "recorded input has incompatible address alignment");
    }
    if (input.alignment > previous->alignment) {
      previous->alignment = input.alignment;
      previous->residue = input.residue;
    }
    previous->maximum_offset =
        iree_max(previous->maximum_offset, input.maximum_offset);
    previous->minimum_offset =
        previous->minimum_offset == UINT64_MAX ||
                input.minimum_offset == UINT64_MAX
            ? UINT64_MAX
            : iree_min(previous->minimum_offset, input.minimum_offset);
  }
  *out_value = IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT + ordinal;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_binding(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_buffer_ref_t reference, uint32_t site, uint32_t minimum_length,
    uint32_t alignment, bool whole_transfer, uint32_t* out_value) {
  if (minimum_length && reference.length != IREE_HAL_WHOLE_BUFFER &&
      reference.length < minimum_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded arguments do not cover the native ABI");
  }
  iree_hal_amd_gpu_pm4_command_input_t input = {
      .key = reference.buffer ? (uintptr_t)reference.buffer
                              : reference.buffer_slot,
      .maximum_offset = reference.offset,
      .minimum_offset = whole_transfer ? reference.offset : UINT64_MAX,
      .kind = reference.buffer ? IREE_HAL_AMD_GPU_PM4_INPUT_DIRECT
                               : IREE_HAL_AMD_GPU_PM4_INPUT_BINDING,
      .site = (uint16_t)site,
      .alignment = alignment,
      .residue = (uint32_t)(0 - reference.offset) & (alignment - 1),
  };
  if (reference.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_qualify_binding(
        buffer->base.queue_family, reference.buffer, site, &input.slot));
    if (minimum_length &&
        iree_hal_buffer_byte_length(reference.buffer) - reference.offset <
            minimum_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "recorded argument buffer is too short");
    }
    if (buffer->resources) {
      IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(buffer->resources, 1,
                                                        &reference.buffer));
    }
  } else {
    iree_hal_amd_gpu_pm4_binding_requirement_t* requirement =
        &buffer->requirements[reference.buffer_slot + 1];
    requirement->sites |= 1u << site;
    uint64_t length = 0;
    if (!iree_checked_add_u64(reference.offset, minimum_length, &length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "recorded argument extent overflows");
    }
    requirement->minimum_length = iree_max(requirement->minimum_length, length);
    buffer->base.binding_count =
        iree_max(buffer->base.binding_count, reference.buffer_slot + 1);
  }
  return iree_hal_amd_gpu_pm4_command_input(buffer, input, out_value);
}

// A weighted union forest collapses arbitrarily many whole-copy constraints to
// one root/difference per binding. Sealing flattens it; invocation validation
// is linear in the table size, independent of the number of recorded copies.
static uint32_t iree_hal_amd_gpu_pm4_command_length_root(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, uint32_t index,
    uint64_t* out_difference) {
  uint64_t difference = 0;
  uint32_t root = index;
  while (buffer->requirements[root].parent != root) {
    difference += buffer->requirements[root].difference;
    root = buffer->requirements[root].parent;
  }
  *out_difference = difference;
  return root;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_equal_lengths(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target) {
  uint64_t source_difference = 0, target_difference = 0;
  const uint32_t source_root = iree_hal_amd_gpu_pm4_command_length_root(
      buffer, source.buffer ? 0 : source.buffer_slot + 1, &source_difference);
  const uint32_t target_root = iree_hal_amd_gpu_pm4_command_length_root(
      buffer, target.buffer ? 0 : target.buffer_slot + 1, &target_difference);
  source_difference +=
      source.buffer ? iree_hal_buffer_byte_length(source.buffer) : 0;
  target_difference +=
      target.buffer ? iree_hal_buffer_byte_length(target.buffer) : 0;
  const uint64_t difference =
      source.offset - target.offset - source_difference + target_difference;
  if (source_root == target_root) {
    if (difference) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "recorded whole-copy extents cannot agree");
    }
  } else if (!source_root) {
    buffer->requirements[target_root].parent = source_root;
    buffer->requirements[target_root].difference = 0 - difference;
  } else {
    buffer->requirements[source_root].parent = target_root;
    buffer->requirements[source_root].difference = difference;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_flush_barrier(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer) {
  const uint32_t count =
      iree_hal_amd_gpu_pm4_barrier_word_count(buffer->pending_barrier);
  if (!count) {
    return iree_ok_status();
  }
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
      &buffer->recording, count, &offset, &words));
  iree_hal_amd_gpu_pm4_barrier_emit(words, buffer->pending_barrier);
  if (buffer->pending_barrier.execution &
      IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA) {
    buffer->program.pending_dma = false;
  }
  buffer->pending_barrier = (iree_hal_amd_gpu_pm4_barrier_t){0};
  ++buffer->program.operation_count;
  return iree_ok_status();
}

// One service call owns native allocation and sealing. Temporary bytes are
// copied directly into the prepared upload, with no giant host flattening copy.
typedef struct iree_hal_amd_gpu_pm4_command_seal_t {
  // Recorder borrowed until the synchronous cold call joins.
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer;
  // Checked total instance extent, excluding the immutable fixup table.
  uint32_t length;
  // Native allocation/publication result returned to command_buffer_end.
  iree_status_t status;
} iree_hal_amd_gpu_pm4_command_seal_t;

static void iree_hal_amd_gpu_pm4_command_seal(void* user_data) {
  iree_hal_amd_gpu_pm4_command_seal_t* seal = user_data;
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = seal->buffer;
  iree_hal_amd_gpu_pm4_program_t* program = &buffer->program.resident;
  seal->status = iree_hal_amd_gpu_pm4_program_initialize(
      buffer->pool, buffer->recording.alignment, seal->length,
      iree_hal_amd_gpu_pm4_recording_fixup_count(&buffer->recording), program);
  if (iree_status_is_ok(seal->status)) {
    iree_hal_amd_gpu_pm4_recording_write(
        &buffer->recording, program->contents.data,
        (void*)(program->contents.data + program->fixup_offset));
    seal->status = iree_hal_amd_gpu_pm4_program_seal(program);
  }
}

// Failed append/seal operations invalidate the native recording. Preserve the
// original error instead of allowing an end/submission to consume a partially
// initialized packet or a partially acquired native allocation.
static iree_status_t iree_hal_amd_gpu_pm4_command_record_status(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, iree_status_t status) {
  if (!iree_status_is_ok(status) &&
      iree_status_is_ok(buffer->recording_status)) {
    buffer->recording_status = iree_status_clone(status);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_begin(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_end(
      &buffer->recording, IREE_HAL_AMD_GPU_PM4_VALUE_CONTINUATION,
      buffer->pool->chain));
  buffer->program.entry = (iree_hal_amd_gpu_pm4_program_block_t){
      .offset = 0,
      .word_count = buffer->recording.entry_word_count,
  };
  const uint64_t length = iree_align_uint64(buffer->recording.commands.length,
                                            buffer->recording.alignment) +
                          buffer->recording.data.length;
  if (length > UINT32_MAX - 15) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident command/data placement exceeds 4 GiB");
  }
  buffer->program.value_count =
      buffer->input_count + IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT;
  buffer->program.argument_length =
      sizeof(iree_hal_amd_gpu_pm4_program_arguments_t) +
      buffer->program.value_count * 8;
  for (uint32_t i = 1; i <= base->binding_count; ++i) {
    uint64_t difference = 0;
    const uint32_t root =
        iree_hal_amd_gpu_pm4_command_length_root(buffer, i, &difference);
    buffer->requirements[i].parent = root;
    buffer->requirements[i].difference = difference;
  }
  iree_hal_amd_gpu_pm4_command_seal_t seal = {
      .buffer = buffer,
      .length = (uint32_t)length,
  };
  iree_hal_memory_maintenance_call(buffer->pool->service,
                                   iree_hal_amd_gpu_pm4_command_seal, &seal);
  if (iree_status_is_ok(seal.status)) {
    if (buffer->resources) {
      iree_hal_resource_set_freeze(buffer->resources);
    }
    iree_hal_amd_gpu_pm4_recording_deinitialize(&buffer->recording);
    iree_allocator_free(buffer->host_allocator, buffer->input_index);
    buffer->input_index = NULL;
  }
  return seal.status;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_end(base));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base, iree_string_view_t label,
    iree_hal_label_color_t color, const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_barrier(
    iree_hal_command_buffer_t* base, const iree_hal_barrier_t* barrier) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  const iree_hal_barrier_list_t list = {.count = 1, .values = barrier};
  const iree_hal_amd_gpu_pm4_barrier_t prepared =
      iree_hal_amd_gpu_pm4_barrier_prepare(&buffer->profile, &list);
  buffer->pending_barrier.cache_control |= prepared.cache_control;
  buffer->pending_barrier.execution |= prepared.execution;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_wait_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded atomic waits are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_store_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded atomic stores are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_rmw_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded atomic RMWs are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t reference,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

// Appends one address replacement without retaining a host record for the
// operation. Both halves read the same dense input and overwrite complete
// words.
static iree_status_t iree_hal_amd_gpu_pm4_command_pointer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, bool data, uint32_t offset,
    uint32_t value, uint64_t displacement, bool private_data) {
  return iree_hal_amd_gpu_pm4_recording_fixup(&buffer->recording, data, offset,
                                              value, displacement, private_data,
                                              0, true);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_transfer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  const bool copy = type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  const bool update = type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  const bool whole = target.length == IREE_HAL_WHOLE_BUFFER;
  if (copy && whole) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_pm4_command_equal_lengths(buffer, source, target));
  }
  uint64_t length = target.length;
  if (whole && target.buffer) {
    length = iree_hal_buffer_byte_length(target.buffer) - target.offset;
  }
  if (!length) {
    return iree_ok_status();
  }
  // Byte-aligned copies use CP DMA. A direct fill can use its immediate form
  // only when prepared backing proves alignment without reading an address.
  const bool aligned_fill =
      !copy && !update && target.buffer && target.buffer->memory.backing &&
      target.buffer->memory.backing->allocation_alignment >= 4 &&
      !((target.buffer->memory.offset + target.offset) & 3) && !(length & 3);
  const bool native_dma =
      length != IREE_HAL_WHOLE_BUFFER && (copy || update || aligned_fill);
  const uint32_t site = native_dma ? IREE_HAL_AMD_GPU_PM4_SITE_QUEUE
                                   : IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM;
  uint32_t source_value = IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, target_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_binding(
      buffer, target, site, 0, 1, whole, &target_value));
  if (copy) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_binding(
        buffer, source, site, 0, 1, whole, &source_value));
  }
  uint32_t update_offset = 0;
  if (update) {
    void* bytes = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_data(
        &buffer->recording, (uint32_t)length, 1, &update_offset, &bytes));
    memcpy(bytes, source_bytes, (iree_host_size_t)length);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  if (native_dma) {
    iree_status_t status = iree_ok_status();
    uint64_t position = 0;
    while (position < length && iree_status_is_ok(status)) {
      const uint32_t part =
          (uint32_t)iree_min(length - position, UINT64_C(0x03fffffc));
      uint32_t command_offset = 0;
      uint32_t* words = NULL;
      status = iree_hal_amd_gpu_pm4_recording_commands(&buffer->recording, 7,
                                                       &command_offset, &words);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (aligned_fill) {
        iree_hal_amd_gpu_pm4_emit_dma_fill(words, 0, part, pattern);
      } else {
        iree_hal_amd_gpu_pm4_emit_dma_copy(words, 0, 0, part);
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, false, command_offset + 8, source_value,
            (update ? update_offset : source.offset) + position, update);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, false, command_offset + 16, target_value,
            target.offset + position, false);
      }
      position += part;
    }
    IREE_RETURN_IF_ERROR(status);
    buffer->program.pending_dma = true;
  } else {
    uint32_t argument_offset = 0;
    iree_hal_amd_gpu_pm4_blit_arguments_t* arguments = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_data(
        &buffer->recording, sizeof(*arguments), 8, &argument_offset,
        (void**)&arguments));
    *arguments = (iree_hal_amd_gpu_pm4_blit_arguments_t){
        .length = length == IREE_HAL_WHOLE_BUFFER ? 0 : length,
        .pattern = pattern,
        .fill = !copy,
    };
    if (copy) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, argument_offset, source_value, source.offset, false));
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, true, argument_offset + 8, target_value, target.offset, false));
    if (length == IREE_HAL_WHOLE_BUFFER) {
      buffer->requirements[target.buffer_slot + 1].sites |=
          IREE_HAL_AMD_GPU_PM4_BINDING_LENGTH;
      const iree_hal_amd_gpu_pm4_command_input_t input = {
          .key = target.buffer_slot,
          .kind = IREE_HAL_AMD_GPU_PM4_INPUT_LENGTH,
          .alignment = 1,
          .minimum_offset = UINT64_MAX,
      };
      uint32_t length_value = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_gpu_pm4_command_input(buffer, input, &length_value));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, argument_offset + 16, length_value, 0 - target.offset,
          false));
    }
    uint32_t command_offset = 0;
    uint32_t* words = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
        &buffer->recording, 31, &command_offset, &words));
    iree_hal_amd_gpu_pm4_blit_emit(words, &buffer->pool->blit, 0);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, command_offset + 24 * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, argument_offset, true));
  }
  ++buffer->program.operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_transfer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_transfer(
                  buffer, type, source, target, source_bytes, pattern));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t target,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  uint32_t value = 0;
  memcpy(&value, pattern, pattern_length);
  if (pattern_length == 1) {
    value *= 0x01010101u;
  }
  if (pattern_length == 2) {
    value |= value << 16;
  }
  return iree_hal_amd_gpu_pm4_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      (iree_hal_buffer_ref_t){0}, target, NULL, value);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base, const void* source,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target,
    iree_hal_update_flags_t flags) {
  return iree_hal_amd_gpu_pm4_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      (iree_hal_buffer_ref_t){0}, target,
      (const uint8_t*)source + source_offset, 0);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, iree_hal_copy_flags_t flags) {
  return iree_hal_amd_gpu_pm4_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY, source, target, NULL,
      0);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_image(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, const uint64_t* addresses,
    bool data, uint32_t offset, uint64_t address, uint32_t shift) {
  const iree_hal_amd_gpu_pm4_command_input_t input = {
      .key = (uintptr_t)addresses,
      .kind = IREE_HAL_AMD_GPU_PM4_INPUT_IMAGE,
      .alignment = 1,
      .minimum_offset = UINT64_MAX,
  };
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_pm4_command_input(buffer, input, &value));
  return iree_hal_amd_gpu_pm4_recording_fixup(&buffer->recording, data, offset,
                                              value, address - addresses[0],
                                              false, shift, true);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  const iree_hal_amd_gpu_pm4_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_executable_lookup_pm4(executable, token, &function));
  iree_hal_amd_gpu_pm4_dispatch_t dispatch;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_dispatch_prepare(
      function, buffer->maximum_group_segment_byte_length, config, constants,
      bindings.count, flags, &dispatch));
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  const bool indirect = iree_hal_dispatch_uses_indirect_arguments(flags);
  if (!indirect_parameters &&
      (!dispatch.workgroup_count[0] || !dispatch.workgroup_count[1] ||
       !dispatch.workgroup_count[2])) {
    return iree_ok_status();
  }
  if (buffer->resources) {
    IREE_RETURN_IF_ERROR(
        iree_hal_resource_set_insert(buffer->resources, 1, &executable));
  }
  const bool prepare_packet = function->packet && indirect_parameters;
  if (prepare_packet) {
    buffer->pending_barrier.cache_control |=
        buffer->profile.command_processor_bypasses_l2 ? buffer->profile.acquire
                                                      : buffer->profile.shader;
  }
  if (buffer->program.tsan_shadow_slot_count &&
      !(buffer->program.dispatch_count &
        (buffer->program.tsan_shadow_slot_count - 1))) {
    // Each invocation starts at a fresh shadow-window boundary. These fixed
    // native joins protect reuse without host inspection of recorded
    // dispatches.
    buffer->pending_barrier.execution |=
        IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
    buffer->pending_barrier.cache_control |= buffer->profile.shader;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
      iree_hal_amd_gpu_pm4_dispatch_storage(function, flags);
  uint32_t argument_offset = 0;
  void* arguments = NULL;
  if (storage.length) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_data(
        &buffer->recording, storage.length, storage.alignment, &argument_offset,
        &arguments));
    iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
        function, &dispatch, constants, NULL, flags, arguments);
  }
  const uint32_t prepare_words =
      prepare_packet ? IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS : 0;
  const uint32_t launch_words = function->launch.user_data_count
                                    ? 24 + function->launch.user_data_count
                                    : 22;
  const uint16_t copy_count = indirect_parameters && !indirect
                                  ? function->kernarg->indirect_copy_count
                                  : 0;
  const uint32_t dispatch_words =
      indirect_parameters
          ? iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(copy_count)
          : 5;
  const uint32_t word_count = prepare_words + launch_words + dispatch_words;
  uint32_t command_offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
      &buffer->recording, word_count, &command_offset, &words));
  uint32_t* cursor = words;
  if (prepare_packet) {
    cursor = iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
        cursor, function->packet->prepare, 0, buffer->profile.shader);
  }
  const uint32_t launch_offset = command_offset + prepare_words * 4;
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {0};
  cursor = iree_hal_amd_gpu_pm4_launch_emit(
      cursor, &function->launch, dispatch.workgroup_size, dispatch.partial_size,
      dispatch.group_segment_byte_length, &inputs);
  const uint32_t initiator = function->launch.dispatch_initiator |
                             (dispatch.partial_size[0] ? UINT32_C(1) << 6 : 0);
  if (indirect_parameters) {
    iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
        cursor, function->kernarg, copy_count, 0, 0,
        buffer->profile.command_processor_bypasses_l2 ? buffer->profile.acquire
                                                      : buffer->profile.shader,
        initiator);
  } else {
    iree_hal_amd_gpu_pm4_emit_dispatch_direct(cursor, dispatch.workgroup_count,
                                              initiator);
  }
  const uint16_t kernarg_word =
      iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
          function->launch.user_data.kernarg_pointer);
  const uint32_t packet_offset = argument_offset + storage.packet_offset;
  uint32_t kernarg_value = IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE;
  uint64_t kernarg_displacement = argument_offset;
  if (indirect && (kernarg_word != UINT16_MAX || function->packet)) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_binding(
        buffer, bindings.values[0], IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM,
        function->kernarg->byte_length, function->kernarg->alignment, false,
        &kernarg_value));
    kernarg_displacement = bindings.values[0].offset;
  } else if (!indirect) {
    const uint16_t* offsets =
        iree_hal_amd_gpu_kernarg_layout_binding_offsets(function->kernarg);
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < bindings.count && iree_status_is_ok(status); ++i) {
      uint32_t value = 0;
      status = iree_hal_amd_gpu_pm4_command_binding(
          buffer, bindings.values[i], IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM, 0, 1,
          false, &value);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, true, argument_offset + offsets[i], value,
            bindings.values[i].offset, false);
      }
    }
    IREE_RETURN_IF_ERROR(status);
  }
  if (kernarg_word != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, launch_offset + kernarg_word * 4, kernarg_value,
        kernarg_displacement, !indirect));
  }
  const uint16_t dispatch_id_word =
      iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
          function->launch.user_data.dispatch_id);
  if (dispatch_id_word != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, launch_offset + dispatch_id_word * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_DISPATCH_ID, buffer->program.dispatch_count,
        false));
  }
  if (function->packet) {
    const uint16_t packet_word =
        iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
            function->launch.user_data.dispatch_pointer);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, launch_offset + packet_word * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, packet_offset, true));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, true,
        packet_offset +
            offsetof(iree_hal_amd_gpu_pm4_dispatch_packet_t, kernarg_address),
        kernarg_value, kernarg_displacement, !indirect));
  }
  if (indirect_parameters) {
    uint32_t parameters = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_binding(
        buffer, config.workgroup_count_ref, IREE_HAL_AMD_GPU_PM4_SITE_QUEUE, 12,
        4, false, &parameters));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, command_offset + (word_count - 3) * 4, parameters,
        config.workgroup_count_ref.offset, false));
    iree_status_t status = iree_ok_status();
    for (uint16_t i = 0; i < copy_count && iree_status_is_ok(status); ++i) {
      uint32_t* copy = cursor + i * 7;
      const uint32_t offset =
          command_offset + (prepare_words + launch_words + i * 7) * 4;
      status = iree_hal_amd_gpu_pm4_command_pointer(
          buffer, false, offset + 8, parameters,
          config.workgroup_count_ref.offset +
              iree_unaligned_load_le_u64(copy + 2),
          false);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, false, offset + 16, IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE,
            argument_offset + iree_unaligned_load_le_u64(copy + 4), true);
      }
    }
    IREE_RETURN_IF_ERROR(status);
    if (copy_count) {
      buffer->program.pending_dma = false;
    }
    if (prepare_packet) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, packet_offset + 64, parameters,
          config.workgroup_count_ref.offset, false));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, packet_offset + 72, IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE,
          packet_offset, true));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, false, command_offset + 24 * 4,
          IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, packet_offset + 64, true));
    }
  }
  const uint64_t* image_addresses =
      iree_hal_amd_gpu_executable_queue_image_addresses(executable);
  if (image_addresses) {
    const uint64_t address = ((uint64_t)function->launch.program[0] << 8) |
                             ((uint64_t)function->launch.program[1] << 40);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_image(
        buffer, image_addresses, false, launch_offset + 8, address, 8));
    if (function->packet) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_image(
          buffer, image_addresses, true,
          packet_offset +
              offsetof(iree_hal_amd_gpu_pm4_dispatch_packet_t, kernel_object),
          function->packet->kernel_object, 0));
      if (prepare_packet) {
        const iree_hal_amd_gpu_pm4_launch_t* prepare =
            function->packet->prepare;
        const uint64_t prepare_address = ((uint64_t)prepare->program[0] << 8) |
                                         ((uint64_t)prepare->program[1] << 40);
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_image(
            buffer, image_addresses, false, command_offset + 8, prepare_address,
            8));
      }
    }
  }
  ++buffer->program.dispatch_count;
  ++buffer->program.operation_count;
  return iree_ok_status();
}
static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_dispatch(
                  base, executable, token, config, constants, bindings, flags));
}

iree_status_t iree_hal_amd_gpu_pm4_command_invocation_initialize(
    iree_hal_command_buffer_t* base, iree_hal_buffer_binding_table_t bindings,
    void* storage, iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_command_invocation_t* out_invocation) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  *out_invocation = (iree_hal_amd_gpu_pm4_command_invocation_t){
      .command_buffer = base,
      .program = &buffer->program,
      .bindings = storage,
      .borrow_bindings = iree_any_bit_set(
          flags, IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME),
  };
  iree_hal_command_buffer_retain(base);
  if (base->binding_count) {
    memset(storage, 0, base->binding_count * sizeof(*out_invocation->bindings));
  }
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < base->binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_pm4_binding_requirement_t* requirement =
        &buffer->requirements[i + 1];
    if (!requirement->sites) {
      continue;
    }
    const iree_hal_buffer_binding_t binding = bindings.bindings[i];
    const uint64_t length =
        binding.length == IREE_HAL_WHOLE_BUFFER
            ? iree_hal_buffer_byte_length(binding.buffer) - binding.offset
            : binding.length;
    if (length < requirement->minimum_length || length > INT64_MAX - 262144) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "binding %u cannot cover native argument/transfer extents", i);
    } else {
      iree_hal_amd_gpu_pm4_command_binding_t* captured =
          &out_invocation->bindings[i];
      for (uint32_t site = 0; site < 2 && iree_status_is_ok(status); ++site) {
        if (requirement->sites & (1u << site)) {
          status = iree_hal_amd_gpu_pm4_command_qualify_binding(
              base->queue_family, binding.buffer, site, &captured->slots[site]);
        }
      }
      if (iree_status_is_ok(status)) {
        captured->buffer = binding.buffer;
        captured->offset = binding.offset;
        captured->length = length;
        if (!out_invocation->borrow_bindings) {
          iree_hal_buffer_retain(binding.buffer);
        }
      }
    }
  }
  for (uint32_t i = 0; i < base->binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_pm4_binding_requirement_t* requirement =
        &buffer->requirements[i + 1];
    if (!requirement->sites) {
      continue;
    }
    const uint64_t root_length =
        requirement->parent
            ? out_invocation->bindings[requirement->parent - 1].length
            : 0;
    if (out_invocation->bindings[i].length - root_length !=
        requirement->difference) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "recorded whole-copy remaining ranges differ at binding %u", i);
    }
  }
  return status;
}

void iree_hal_amd_gpu_pm4_command_invocation_deinitialize(
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation) {
  if (invocation->instance) {
    if (invocation->published) {
      iree_hal_amd_gpu_pm4_program_release(&invocation->program->resident,
                                           invocation->instance);
    } else {
      iree_hal_amd_gpu_pm4_program_cancel(&invocation->program->resident,
                                          invocation->instance);
    }
  }
  if (!invocation->borrow_bindings) {
    for (uint32_t i = 0; i < invocation->command_buffer->binding_count; ++i) {
      iree_hal_buffer_release(invocation->bindings[i].buffer);
    }
  }
  iree_hal_command_buffer_release(invocation->command_buffer);
}

iree_status_t iree_hal_amd_gpu_pm4_command_invocation_write(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t dispatch_id, uint64_t continuation,
    void* upload) {
  const iree_hal_amd_gpu_pm4_command_buffer_t* buffer =
      (const void*)invocation->command_buffer;
  uint64_t* values =
      (void*)((iree_hal_amd_gpu_pm4_program_arguments_t*)upload + 1);
  values[IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE] = invocation->instance->address;
  values[IREE_HAL_AMD_GPU_PM4_VALUE_DISPATCH_ID] = dispatch_id;
  values[IREE_HAL_AMD_GPU_PM4_VALUE_CONTINUATION] = continuation;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < buffer->input_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_pm4_command_input_t* input = &buffer->inputs[i];
    uint64_t value = 0;
    if (input->kind == IREE_HAL_AMD_GPU_PM4_INPUT_IMAGE) {
      value = ((const uint64_t*)input->key)[queue_ordinal];
    } else if (input->kind == IREE_HAL_AMD_GPU_PM4_INPUT_LENGTH) {
      value = invocation->bindings[input->key].length;
    } else {
      iree_hal_buffer_t* source = (void*)input->key;
      iree_hal_buffer_native_binding_slot_t slot = input->slot;
      uint64_t offset = 0, length = 0;
      if (input->kind == IREE_HAL_AMD_GPU_PM4_INPUT_BINDING) {
        const iree_hal_amd_gpu_pm4_command_binding_t* binding =
            &invocation->bindings[input->key];
        source = binding->buffer;
        slot = binding->slots[input->site];
        offset = binding->offset;
        length = binding->length;
      } else {
        length = iree_hal_buffer_byte_length(source);
      }
      value = iree_hal_buffer_native_binding(source, slot).device_address;
      const bool empty = input->minimum_offset != UINT64_MAX &&
                         input->minimum_offset == input->maximum_offset &&
                         input->maximum_offset == length;
      if (!value && !empty) {
        status =
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "recorded input has no committed GPU address");
      } else if (!iree_checked_add_u64(value, offset, &value) ||
                 input->maximum_offset > UINT64_MAX - value ||
                 ((value & (input->alignment - 1)) != input->residue)) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "recorded GPU input overflows or is misaligned");
      }
    }
    values[IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT + i] = value;
  }
  return status;
}
static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_pm4_command_buffer_vtable = {
        .destroy = iree_hal_amd_gpu_pm4_command_buffer_destroy,
        .begin = iree_hal_amd_gpu_pm4_command_buffer_begin,
        .end = iree_hal_amd_gpu_pm4_command_buffer_end,
        .begin_debug_group =
            iree_hal_amd_gpu_pm4_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amd_gpu_pm4_command_buffer_end_debug_group,
        .barrier = iree_hal_amd_gpu_pm4_command_buffer_barrier,
        .atomic_wait = iree_hal_amd_gpu_pm4_command_buffer_atomic_wait,
        .atomic_store = iree_hal_amd_gpu_pm4_command_buffer_atomic_store,
        .atomic_rmw = iree_hal_amd_gpu_pm4_command_buffer_atomic_rmw,
        .advise_buffer = iree_hal_amd_gpu_pm4_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amd_gpu_pm4_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amd_gpu_pm4_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amd_gpu_pm4_command_buffer_copy_buffer,
        .dispatch = iree_hal_amd_gpu_pm4_command_buffer_dispatch,
};
