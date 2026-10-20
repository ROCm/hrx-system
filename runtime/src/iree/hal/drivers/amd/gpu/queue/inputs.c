// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/inputs.h"

enum {
  IREE_HAL_AMD_GPU_INPUT_DIRECT = 0,
  IREE_HAL_AMD_GPU_INPUT_BINDING = 1,
  IREE_HAL_AMD_GPU_INPUT_IMAGE = 2,
  IREE_HAL_AMD_GPU_INPUT_LENGTH = 3,
  IREE_HAL_AMD_GPU_BINDING_LENGTH = 1u << 2,
};

// A deduplicated invocation input. Recording aggregates alignment and overflow
// requirements; publication reads each source exactly once. Byte offsets belong
// to immutable GPU fixups, not to a per-command host relocation list.
typedef struct iree_hal_amd_gpu_command_input_t {
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
} iree_hal_amd_gpu_command_input_t;

void iree_hal_amd_gpu_command_inputs_initialize(
    iree_hal_command_buffer_t* command_buffer, uint32_t value_base,
    iree_hal_resource_set_t* resources,
    iree_hal_amd_gpu_binding_requirement_t* requirements,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_command_inputs_t* out_inputs) {
  *out_inputs = (iree_hal_amd_gpu_command_inputs_t){
      .command_buffer = command_buffer,
      .host_allocator = host_allocator,
      .resources = resources,
      .requirements = requirements,
      .value_base = value_base,
  };
  for (uint32_t i = 0; i <= command_buffer->binding_capacity; ++i) {
    requirements[i].parent = i;
  }
}

void iree_hal_amd_gpu_command_inputs_deinitialize(
    iree_hal_amd_gpu_command_inputs_t* inputs) {
  iree_allocator_free(inputs->host_allocator, inputs->entries);
  iree_allocator_free(inputs->host_allocator, inputs->index);
}

static iree_status_t iree_hal_amd_gpu_command_qualify_binding(
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

static uint32_t iree_hal_amd_gpu_command_input_hash(uintptr_t key,
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

static iree_status_t iree_hal_amd_gpu_command_input(
    iree_hal_amd_gpu_command_inputs_t* buffer,
    iree_hal_amd_gpu_command_input_t input, uint32_t* out_value) {
  if ((buffer->count + 1) * 2 >= buffer->index_capacity) {
    const uint32_t capacity = iree_max(buffer->index_capacity * 2, 32u);
    uint32_t* index = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        buffer->host_allocator, capacity * sizeof(*index), (void**)&index));
    for (uint32_t i = 0; i < buffer->count; ++i) {
      const iree_hal_amd_gpu_command_input_t* value = &buffer->entries[i];
      uint32_t position = iree_hal_amd_gpu_command_input_hash(
                              value->key, value->kind, value->site) &
                          (capacity - 1);
      while (index[position]) {
        position = (position + 1) & (capacity - 1);
      }
      index[position] = i + 1;
    }
    iree_allocator_free(buffer->host_allocator, buffer->index);
    buffer->index = index;
    buffer->index_capacity = capacity;
  }
  uint32_t position =
      iree_hal_amd_gpu_command_input_hash(input.key, input.kind, input.site) &
      (buffer->index_capacity - 1);
  uint32_t ordinal = buffer->count;
  while (buffer->index[position]) {
    const uint32_t candidate = buffer->index[position] - 1;
    const iree_hal_amd_gpu_command_input_t* previous =
        &buffer->entries[candidate];
    if (previous->key == input.key && previous->kind == input.kind &&
        previous->site == input.site) {
      ordinal = candidate;
      break;
    }
    position = (position + 1) & (buffer->index_capacity - 1);
  }
  if (ordinal == buffer->count) {
    if (ordinal >= 0x1000000u - buffer->value_base) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "resident program has too many distinct inputs");
    }
    if (ordinal == buffer->capacity) {
      const uint32_t capacity = iree_max(buffer->capacity * 2, 16u);
      IREE_RETURN_IF_ERROR(iree_allocator_realloc(
          buffer->host_allocator, capacity * sizeof(*buffer->entries),
          (void**)&buffer->entries));
      buffer->capacity = capacity;
    }
    buffer->entries[ordinal] = input;
    buffer->index[position] = ordinal + 1;
    ++buffer->count;
  } else {
    iree_hal_amd_gpu_command_input_t* previous = &buffer->entries[ordinal];
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
  *out_value = buffer->value_base + ordinal;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_command_inputs_add_binding(
    iree_hal_amd_gpu_command_inputs_t* buffer, iree_hal_buffer_ref_t reference,
    uint32_t site, uint32_t minimum_length, uint32_t alignment,
    bool whole_transfer, uint32_t* out_value) {
  if (minimum_length && reference.length != IREE_HAL_WHOLE_BUFFER &&
      reference.length < minimum_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded arguments do not cover the native ABI");
  }
  iree_hal_amd_gpu_command_input_t input = {
      .key = reference.buffer ? (uintptr_t)reference.buffer
                              : reference.buffer_slot,
      .maximum_offset = reference.offset,
      .minimum_offset = whole_transfer ? reference.offset : UINT64_MAX,
      .kind = reference.buffer ? IREE_HAL_AMD_GPU_INPUT_DIRECT
                               : IREE_HAL_AMD_GPU_INPUT_BINDING,
      .site = (uint16_t)site,
      .alignment = alignment,
      .residue = (uint32_t)(0 - reference.offset) & (alignment - 1),
  };
  if (reference.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_qualify_binding(
        buffer->command_buffer->queue_family, reference.buffer, site,
        &input.slot));
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
    iree_hal_amd_gpu_binding_requirement_t* requirement =
        &buffer->requirements[reference.buffer_slot + 1];
    requirement->sites |= 1u << site;
    uint64_t length = 0;
    if (!iree_checked_add_u64(reference.offset, minimum_length, &length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "recorded argument extent overflows");
    }
    requirement->minimum_length = iree_max(requirement->minimum_length, length);
    buffer->command_buffer->binding_count = iree_max(
        buffer->command_buffer->binding_count, reference.buffer_slot + 1);
  }
  return iree_hal_amd_gpu_command_input(buffer, input, out_value);
}

// A weighted union forest collapses arbitrarily many whole-copy constraints to
// one root/difference per binding. Sealing flattens it; invocation validation
// is linear in the table size, independent of the number of recorded copies.
static uint32_t iree_hal_amd_gpu_command_length_root(
    iree_hal_amd_gpu_command_inputs_t* buffer, uint32_t index,
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

iree_status_t iree_hal_amd_gpu_command_inputs_equal_lengths(
    iree_hal_amd_gpu_command_inputs_t* buffer, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target) {
  uint64_t source_difference = 0, target_difference = 0;
  const uint32_t source_root = iree_hal_amd_gpu_command_length_root(
      buffer, source.buffer ? 0 : source.buffer_slot + 1, &source_difference);
  const uint32_t target_root = iree_hal_amd_gpu_command_length_root(
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

iree_status_t iree_hal_amd_gpu_command_inputs_add_length(
    iree_hal_amd_gpu_command_inputs_t* inputs, uint32_t binding_slot,
    uint32_t* out_value) {
  inputs->requirements[binding_slot + 1].sites |=
      IREE_HAL_AMD_GPU_BINDING_LENGTH;
  const iree_hal_amd_gpu_command_input_t input = {
      .key = binding_slot,
      .kind = IREE_HAL_AMD_GPU_INPUT_LENGTH,
      .alignment = 1,
      .minimum_offset = UINT64_MAX};
  return iree_hal_amd_gpu_command_input(inputs, input, out_value);
}

iree_status_t iree_hal_amd_gpu_command_inputs_add_image(
    iree_hal_amd_gpu_command_inputs_t* inputs, const uint64_t* addresses,
    uint32_t* out_value) {
  const iree_hal_amd_gpu_command_input_t input = {
      .key = (uintptr_t)addresses,
      .kind = IREE_HAL_AMD_GPU_INPUT_IMAGE,
      .alignment = 1,
      .minimum_offset = UINT64_MAX};
  return iree_hal_amd_gpu_command_input(inputs, input, out_value);
}

void iree_hal_amd_gpu_command_inputs_seal(
    iree_hal_amd_gpu_command_inputs_t* buffer) {
  for (uint32_t i = 1; i <= buffer->command_buffer->binding_count; ++i) {
    uint64_t difference = 0;
    const uint32_t root =
        iree_hal_amd_gpu_command_length_root(buffer, i, &difference);
    buffer->requirements[i].parent = root;
    buffer->requirements[i].difference = difference;
  }
  iree_allocator_free(buffer->host_allocator, buffer->index);
  buffer->index = NULL;
}

iree_status_t iree_hal_amd_gpu_command_inputs_capture(
    const iree_hal_amd_gpu_command_inputs_t* buffer,
    iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_command_binding_t* captured_bindings) {
  const iree_hal_command_buffer_t* base = buffer->command_buffer;
  const bool borrow_bindings =
      (flags & IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME) != 0;
  if (base->binding_count) {
    memset(captured_bindings, 0,
           base->binding_count * sizeof(*captured_bindings));
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < base->binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_binding_requirement_t* requirement =
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
      iree_hal_amd_gpu_command_binding_t* captured = &captured_bindings[i];
      for (uint32_t site = 0; site < 2 && iree_status_is_ok(status); ++site) {
        if (requirement->sites & (1u << site)) {
          status = iree_hal_amd_gpu_command_qualify_binding(
              base->queue_family, binding.buffer, site, &captured->slots[site]);
        }
      }
      if (iree_status_is_ok(status)) {
        captured->buffer = binding.buffer;
        captured->offset = binding.offset;
        captured->length = length;
        if (!borrow_bindings) {
          iree_hal_buffer_retain(binding.buffer);
        }
      }
    }
  }
  for (uint32_t i = 0; i < base->binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_binding_requirement_t* requirement =
        &buffer->requirements[i + 1];
    if (!requirement->sites) {
      continue;
    }
    const uint64_t root_length =
        requirement->parent ? captured_bindings[requirement->parent - 1].length
                            : 0;
    if (captured_bindings[i].length - root_length != requirement->difference) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "recorded whole-copy remaining ranges differ at binding %u", i);
    }
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_command_inputs_write(
    const iree_hal_amd_gpu_command_inputs_t* buffer,
    const iree_hal_amd_gpu_command_binding_t* bindings, uint32_t queue_ordinal,
    uint64_t* values) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < buffer->count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_gpu_command_input_t* input = &buffer->entries[i];
    uint64_t value = 0;
    if (input->kind == IREE_HAL_AMD_GPU_INPUT_IMAGE) {
      value = ((const uint64_t*)input->key)[queue_ordinal];
    } else if (input->kind == IREE_HAL_AMD_GPU_INPUT_LENGTH) {
      value = bindings[input->key].length;
    } else {
      iree_hal_buffer_t* source = (void*)input->key;
      iree_hal_buffer_native_binding_slot_t slot = input->slot;
      uint64_t offset = 0, length = 0;
      if (input->kind == IREE_HAL_AMD_GPU_INPUT_BINDING) {
        const iree_hal_amd_gpu_command_binding_t* binding =
            &bindings[input->key];
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
    values[i] = value;
  }
  return status;
}
