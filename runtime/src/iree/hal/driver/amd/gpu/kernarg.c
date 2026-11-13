// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/kernarg.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/memory_copy.h"

// Selectors index the existing compact tables in native ABI order. The cold
// metadata walk preserves this order so streaming writers never merge tables
// or revisit reflection on the dispatch path.
enum iree_hal_amd_gpu_kernarg_source_e {
  IREE_HAL_AMD_GPU_KERNARG_SOURCE_CONSTANT = 0,
  IREE_HAL_AMD_GPU_KERNARG_SOURCE_RUNTIME = 1,
  IREE_HAL_AMD_GPU_KERNARG_SOURCE_BINDING = 2,
};

typedef struct iree_hal_amd_gpu_kernarg_runtime_field_t {
  // Byte offset in the typed host runtime input record.
  uint16_t offset;
  // Native ABI field width, or zero for a reserved slot.
  uint16_t size;
  // Required native service, or zero for ordinary dispatch geometry.
  iree_hal_amd_gpu_kernarg_requirements_t requirements;
} iree_hal_amd_gpu_kernarg_runtime_field_t;

// Widths and semantics follow the AMDHSA V3/V5 kernel argument metadata ABI.
// Destinations come from each admitted image, never an assumed suffix layout.
// https://llvm.org/docs/AMDGPUUsage.html#amdhsa-code-object-v5-metadata
#define FIELD(kind, member, requirements)                     \
  [IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_##kind] = { \
      offsetof(iree_hal_amd_gpu_kernarg_inputs_t, member),    \
      sizeof(((iree_hal_amd_gpu_kernarg_inputs_t*)0)->member), requirements}
static const iree_hal_amd_gpu_kernarg_runtime_field_t
    iree_hal_amd_gpu_kernarg_runtime_fields[] = {
        [IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE] = {0},
        FIELD(GLOBAL_OFFSET_X, global_offset[0], 0),
        FIELD(GLOBAL_OFFSET_Y, global_offset[1], 0),
        FIELD(GLOBAL_OFFSET_Z, global_offset[2], 0),
        FIELD(PRINTF_BUFFER, services.printf_buffer,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_PRINTF),
        FIELD(HOSTCALL_BUFFER, services.hostcall_buffer,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HOSTCALL),
        FIELD(DEFAULT_QUEUE, services.default_queue,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_DEVICE_ENQUEUE),
        FIELD(COMPLETION_ACTION, services.completion_action,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_COMPLETION_ACTION),
        FIELD(MULTIGRID_SYNC_ARG, services.grid_sync,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_GRID_SYNC),
        FIELD(BLOCK_COUNT_X, block_count[0], 0),
        FIELD(BLOCK_COUNT_Y, block_count[1], 0),
        FIELD(BLOCK_COUNT_Z, block_count[2], 0),
        FIELD(GROUP_SIZE_X, group_size[0], 0),
        FIELD(GROUP_SIZE_Y, group_size[1], 0),
        FIELD(GROUP_SIZE_Z, group_size[2], 0),
        FIELD(REMAINDER_X, remainder[0], 0),
        FIELD(REMAINDER_Y, remainder[1], 0),
        FIELD(REMAINDER_Z, remainder[2], 0),
        FIELD(GRID_DIMS, grid_dimensions, 0),
        FIELD(HEAP_V1, services.heap,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HEAP),
        FIELD(DYNAMIC_LDS_SIZE, dynamic_lds_size, 0),
        FIELD(PRIVATE_BASE, apertures.private_base,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_PRIVATE_APERTURE),
        FIELD(SHARED_BASE, apertures.shared_base,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_SHARED_APERTURE),
        FIELD(QUEUE_PTR, services.queue_pointer,
              IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_QUEUE_POINTER),
};
#undef FIELD
static_assert(IREE_ARRAYSIZE(iree_hal_amd_gpu_kernarg_runtime_fields) ==
                  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_QUEUE_PTR + 1,
              "runtime input table must cover every admitted hidden kind");

static bool iree_hal_amd_gpu_kernarg_is_binding(
    iree_hal_amd_gpu_metadata_argument_kind_t kind) {
  return kind == IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER ||
         (kind >= IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_IMAGE &&
          kind <= IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_QUEUE);
}

static iree_status_t iree_hal_amd_gpu_kernarg_analyze(
    const iree_hal_amd_gpu_metadata_kernel_t* kernel,
    iree_hal_amd_gpu_kernarg_layout_t* layout) {
  if (!kernel->has_argument_metadata && kernel->kernarg_segment_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "kernel argument metadata is required to prepare "
                            "a nonempty native argument segment");
  }
  if (kernel->kernarg_segment_size > UINT16_MAX ||
      kernel->argument_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel arguments exceed HAL parameter reflection "
                            "offset or count capacity");
  }
  layout->byte_length = kernel->kernarg_segment_size;
  layout->alignment = kernel->kernarg_segment_alignment;
  uint32_t constant_end = UINT32_MAX;
  uint32_t runtime_source_end = UINT32_MAX;
  uint32_t runtime_target_end = UINT32_MAX;
  uint32_t indirect_source_end = UINT32_MAX;
  uint32_t indirect_target_end = UINT32_MAX;
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    const iree_hal_amd_gpu_metadata_argument_t* argument =
        &kernel->arguments[i];
    const iree_hal_amd_gpu_metadata_argument_kind_t kind = argument->kind;
    if (kind == IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_UNKNOWN) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "kernel argument kind `%.*s` is unsupported",
                              (int)argument->value_kind.size,
                              argument->value_kind.data);
    }
    if (kind == IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE) {
      continue;
    }
    uint32_t expected_size = argument->size;
    if (iree_hal_amd_gpu_metadata_argument_is_hidden(kind)) {
      const iree_hal_amd_gpu_kernarg_runtime_field_t field =
          iree_hal_amd_gpu_kernarg_runtime_fields[kind];
      expected_size = field.size;
      layout->requirements |= field.requirements;
      if (kind ==
          IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_MULTIGRID_SYNC_ARG) {
        layout->grid_sync_offset = (uint16_t)argument->offset;
      }
      if (field.offset != runtime_source_end ||
          argument->offset != runtime_target_end) {
        ++layout->runtime_copy_count;
      }
      runtime_source_end = field.offset + field.size;
      runtime_target_end = argument->offset + argument->size;
      if (kind >=
              IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X &&
          kind <=
              IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Z) {
        const uint32_t source_offset =
            (kind -
             IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X) *
            4;
        if (source_offset != indirect_source_end ||
            argument->offset != indirect_target_end) {
          ++layout->indirect_copy_count;
        }
        indirect_source_end = source_offset + 4;
        indirect_target_end = argument->offset + 4;
      }
    } else {
      ++layout->parameter_count;
      layout->native_byte_length = argument->offset + argument->size;
      if (iree_hal_amd_gpu_kernarg_is_binding(kind)) {
        expected_size = sizeof(uint64_t);
        ++layout->binding_count;
      } else {
        if (kind ==
            IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_DYNAMIC_SHARED_POINTER) {
          expected_size = sizeof(uint32_t);
        }
        if (argument->size) {
          if (constant_end != argument->offset) {
            ++layout->constant_copy_count;
          }
          constant_end = argument->offset + argument->size;
        }
        layout->constant_byte_length += argument->size;
      }
    }
    if (argument->size != expected_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel argument kind `%.*s` requires %u bytes, received %u",
          (int)argument->value_kind.size, argument->value_kind.data,
          expected_size, argument->size);
    }
  }
  layout->span_count = layout->binding_count + layout->constant_copy_count +
                       layout->runtime_copy_count;
  return iree_ok_status();
}

static iree_hal_amd_gpu_kernarg_copy_t* iree_hal_amd_gpu_kernarg_copies(
    const iree_hal_amd_gpu_kernarg_layout_t* layout) {
  return (
      iree_hal_amd_gpu_kernarg_copy_t*)(iree_hal_amd_gpu_kernarg_layout_binding_offsets(
                                            layout) +
                                        layout->binding_count);
}

static uint32_t* iree_hal_amd_gpu_kernarg_order(
    const iree_hal_amd_gpu_kernarg_layout_t* layout) {
  const void* end = iree_hal_amd_gpu_kernarg_layout_indirect_copies(layout) +
                    layout->indirect_copy_count;
  return (uint32_t*)iree_host_align((iree_host_size_t)(uintptr_t)end,
                                    iree_alignof(uint32_t));
}

iree_status_t iree_hal_amd_gpu_kernarg_layout_allocate(
    const iree_hal_amd_gpu_metadata_kernel_t* kernel,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_kernarg_layout_t** out_layout) {
  *out_layout = NULL;
  iree_hal_amd_gpu_kernarg_layout_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_kernarg_analyze(kernel, &plan));
  iree_host_size_t byte_length = 0, parameters_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(plan), &byte_length,
      IREE_STRUCT_FIELD_ALIGNED(plan.binding_count, uint16_t,
                                iree_alignof(uint16_t), NULL),
      IREE_STRUCT_FIELD_ALIGNED(
          (iree_host_size_t)plan.constant_copy_count + plan.runtime_copy_count +
              plan.indirect_copy_count,
          iree_hal_amd_gpu_kernarg_copy_t,
          iree_alignof(iree_hal_amd_gpu_kernarg_copy_t), NULL),
      IREE_STRUCT_FIELD_ALIGNED(plan.span_count, uint32_t,
                                iree_alignof(uint32_t), NULL),
      IREE_STRUCT_FIELD_ALIGNED(
          plan.parameter_count, iree_hal_executable_function_parameter_t,
          iree_alignof(iree_hal_executable_function_parameter_t),
          &parameters_offset)));
  iree_hal_amd_gpu_kernarg_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, byte_length, (void**)&layout));
  *layout = plan;
  layout->parameters_offset = (uint32_t)parameters_offset;
  layout->host_allocator = host_allocator;
  uint16_t* bindings =
      (uint16_t*)iree_hal_amd_gpu_kernarg_layout_binding_offsets(layout);
  iree_hal_amd_gpu_kernarg_copy_t* copies =
      iree_hal_amd_gpu_kernarg_copies(layout);
  iree_hal_amd_gpu_kernarg_copy_t* runtime_copies =
      copies + layout->constant_copy_count;
  iree_hal_amd_gpu_kernarg_copy_t* indirect_copies =
      runtime_copies + layout->runtime_copy_count;
  uint32_t* order = iree_hal_amd_gpu_kernarg_order(layout);
  iree_hal_executable_function_parameter_t* parameters =
      (iree_hal_executable_function_parameter_t*)((uint8_t*)layout +
                                                  parameters_offset);
  uint16_t binding_index = 0, copy_index = 0, runtime_index = 0;
  uint16_t parameter_index = 0, constant_offset = 0;
  uint16_t indirect_index = 0, span_index = 0;
  for (iree_host_size_t i = 0; i < kernel->argument_count; ++i) {
    const iree_hal_amd_gpu_metadata_argument_t* argument =
        &kernel->arguments[i];
    const iree_hal_amd_gpu_metadata_argument_kind_t kind = argument->kind;
    if (kind == IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE) {
      continue;
    }
    if (iree_hal_amd_gpu_metadata_argument_is_hidden(kind)) {
      const iree_hal_amd_gpu_kernarg_runtime_field_t field =
          iree_hal_amd_gpu_kernarg_runtime_fields[kind];
      iree_hal_amd_gpu_kernarg_copy_t* previous =
          runtime_index ? &runtime_copies[runtime_index - 1] : NULL;
      if (previous &&
          previous->source_offset + previous->length == field.offset &&
          previous->target_offset + previous->length == argument->offset) {
        previous->length += field.size;
      } else {
        order[span_index++] =
            (IREE_HAL_AMD_GPU_KERNARG_SOURCE_RUNTIME << 16) | runtime_index;
        runtime_copies[runtime_index++] = (iree_hal_amd_gpu_kernarg_copy_t){
            (uint16_t)argument->offset, field.offset, field.size};
      }
      if (kind >=
              IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X &&
          kind <=
              IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Z) {
        const uint16_t source_offset =
            (uint16_t)((kind -
                        IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X) *
                       4);
        iree_hal_amd_gpu_kernarg_copy_t* previous =
            indirect_index ? &indirect_copies[indirect_index - 1] : NULL;
        if (previous &&
            previous->source_offset + previous->length == source_offset &&
            previous->target_offset + previous->length == argument->offset) {
          previous->length += 4;
        } else {
          indirect_copies[indirect_index++] = (iree_hal_amd_gpu_kernarg_copy_t){
              (uint16_t)argument->offset, source_offset, 4};
        }
      }
      continue;
    }
    const bool is_binding = iree_hal_amd_gpu_kernarg_is_binding(kind);
    parameters[parameter_index++] = (iree_hal_executable_function_parameter_t){
        .type = is_binding
                    ? IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING
                    : IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
        .flags = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET,
        .size = (uint16_t)argument->size,
        .offset = is_binding ? binding_index : constant_offset,
        .native_abi_offset = (uint16_t)argument->offset,
        .name = argument->name,
    };
    if (is_binding) {
      order[span_index++] =
          (IREE_HAL_AMD_GPU_KERNARG_SOURCE_BINDING << 16) | binding_index;
      bindings[binding_index++] = (uint16_t)argument->offset;
    } else {
      if (argument->size) {
        if (copy_index && (uint32_t)copies[copy_index - 1].target_offset +
                                  copies[copy_index - 1].length ==
                              argument->offset) {
          copies[copy_index - 1].length += (uint16_t)argument->size;
        } else {
          order[span_index++] =
              (IREE_HAL_AMD_GPU_KERNARG_SOURCE_CONSTANT << 16) | copy_index;
          copies[copy_index++] = (iree_hal_amd_gpu_kernarg_copy_t){
              (uint16_t)argument->offset, constant_offset,
              (uint16_t)argument->size};
        }
      }
      constant_offset += (uint16_t)argument->size;
    }
  }
  *out_layout = layout;
  return iree_ok_status();
}

void iree_hal_amd_gpu_kernarg_layout_free(
    iree_hal_amd_gpu_kernarg_layout_t* layout) {
  if (layout) {
    iree_allocator_free(layout->host_allocator, layout);
  }
}

static void iree_hal_amd_gpu_kernarg_copy(
    uint16_t count, const iree_hal_amd_gpu_kernarg_copy_t* copies,
    const void* source, void* target) {
  for (uint16_t i = 0; i < count; ++i) {
    memcpy((uint8_t*)target + copies[i].target_offset,
           (const uint8_t*)source + copies[i].source_offset, copies[i].length);
  }
}

void iree_hal_amd_gpu_kernarg_layout_emplace(
    const iree_hal_amd_gpu_kernarg_layout_t* layout, const void* constants,
    const uint64_t* binding_addresses,
    const iree_hal_amd_gpu_kernarg_inputs_t* inputs, void* target) {
  if (layout->byte_length) {
    memset(target, 0, layout->byte_length);
  }
  const uint16_t* offsets =
      iree_hal_amd_gpu_kernarg_layout_binding_offsets(layout);
  for (uint16_t i = 0; binding_addresses && i < layout->binding_count; ++i) {
    iree_unaligned_store_le_u64((uint8_t*)target + offsets[i],
                                binding_addresses[i]);
  }
  const iree_hal_amd_gpu_kernarg_copy_t* copies =
      iree_hal_amd_gpu_kernarg_copies(layout);
  iree_hal_amd_gpu_kernarg_copy(layout->constant_copy_count, copies, constants,
                                target);
  iree_hal_amd_gpu_kernarg_copy(layout->runtime_copy_count,
                                copies + layout->constant_copy_count, inputs,
                                target);
}

void iree_hal_amd_gpu_kernarg_layout_emplace_native(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_const_byte_span_t constants,
    const iree_hal_amd_gpu_kernarg_inputs_t* inputs, void* target) {
  const iree_host_size_t copy_length =
      iree_min(constants.data_length, layout->byte_length);
  if (copy_length) {
    memcpy(target, constants.data, copy_length);
  }
  if (copy_length < layout->byte_length) {
    memset((uint8_t*)target + copy_length, 0,
           layout->byte_length - copy_length);
  }
  iree_hal_amd_gpu_kernarg_copy(
      layout->runtime_copy_count,
      iree_hal_amd_gpu_kernarg_copies(layout) + layout->constant_copy_count,
      inputs, target);
}

// Only the not-yet-written DWORD lives in CPU scratch. A source span can end at
// any ABI byte; adjacent fields complete this word before its sole GPU write.
typedef struct iree_hal_amd_gpu_kernarg_writer_t {
  // Next aligned destination word, exclusively owned by this invocation.
  uint8_t* target;
  // Low bytes awaiting their one non-temporal destination write.
  uint32_t pending;
  // Number of initialized low bytes, between zero and three.
  uint32_t pending_length : 2;
} iree_hal_amd_gpu_kernarg_writer_t;

static void iree_hal_amd_gpu_kernarg_writer_flush(
    iree_hal_amd_gpu_kernarg_writer_t* writer) {
  uint8_t bytes[4];
  iree_unaligned_store_le_u32(bytes, writer->pending);
  iree_memcpy_stream_dst(writer->target, bytes, sizeof(bytes));
  writer->target += 4;
  writer->pending = 0;
  writer->pending_length = 0;
}

static void iree_hal_amd_gpu_kernarg_writer_append(
    iree_hal_amd_gpu_kernarg_writer_t* writer, const void* source,
    uint32_t length) {
  const uint8_t* bytes = (const uint8_t*)source;
  if (writer->pending_length) {
    const uint32_t prefix = iree_min(length, 4 - writer->pending_length);
    for (uint32_t i = 0; i < prefix; ++i) {
      writer->pending |= (uint32_t)bytes[i]
                         << ((writer->pending_length + i) * 8);
    }
    const uint32_t filled = writer->pending_length + prefix;
    if (filled == 4) {
      iree_hal_amd_gpu_kernarg_writer_flush(writer);
    } else {
      writer->pending_length = filled;
    }
    bytes += prefix;
    length -= prefix;
  }
  const uint32_t aligned_length = length & ~UINT32_C(3);
  if (aligned_length) {
    iree_memcpy_stream_dst(writer->target, bytes, aligned_length);
    writer->target += aligned_length;
    bytes += aligned_length;
    length -= aligned_length;
  }
  for (uint32_t i = 0; i < length; ++i) {
    writer->pending |= (uint32_t)bytes[i] << (writer->pending_length++ * 8);
  }
}

static void iree_hal_amd_gpu_kernarg_writer_zero(
    iree_hal_amd_gpu_kernarg_writer_t* writer, uint32_t length) {
  if (writer->pending_length) {
    const uint32_t prefix = iree_min(length, 4 - writer->pending_length);
    const uint32_t filled = writer->pending_length + prefix;
    length -= prefix;
    if (filled == 4) {
      iree_hal_amd_gpu_kernarg_writer_flush(writer);
    } else {
      writer->pending_length = filled;
    }
  }
  const uint32_t aligned_length = length & ~UINT32_C(3);
  if (aligned_length) {
    iree_memset_stream_dst(writer->target, 0, aligned_length);
    writer->target += aligned_length;
    length -= aligned_length;
  }
  writer->pending_length += length;
}

void iree_hal_amd_gpu_kernarg_layout_emplace_stream(
    const iree_hal_amd_gpu_kernarg_layout_t* layout, const void* constants,
    const uint64_t* binding_addresses,
    const iree_hal_amd_gpu_kernarg_inputs_t* inputs, void* target) {
  iree_hal_amd_gpu_kernarg_writer_t writer = {.target = target};
  const uint32_t* order = iree_hal_amd_gpu_kernarg_order(layout);
  const uint16_t* bindings =
      iree_hal_amd_gpu_kernarg_layout_binding_offsets(layout);
  const iree_hal_amd_gpu_kernarg_copy_t* copies =
      iree_hal_amd_gpu_kernarg_copies(layout);
  uint32_t position = 0;
  for (uint16_t i = 0; i < layout->span_count; ++i) {
    const uint32_t kind = order[i] >> 16;
    const uint16_t index = (uint16_t)order[i];
    uint32_t offset, length;
    const void* source;
    uint8_t binding_bytes[8];
    if (kind == IREE_HAL_AMD_GPU_KERNARG_SOURCE_BINDING) {
      offset = bindings[index];
      length = sizeof(binding_bytes);
      iree_unaligned_store_le_u64(
          binding_bytes, binding_addresses ? binding_addresses[index] : 0);
      source = binding_bytes;
    } else {
      const bool runtime = kind == IREE_HAL_AMD_GPU_KERNARG_SOURCE_RUNTIME;
      const iree_hal_amd_gpu_kernarg_copy_t copy =
          copies[index + (runtime ? layout->constant_copy_count : 0)];
      offset = copy.target_offset;
      length = copy.length;
      source = (const uint8_t*)(runtime ? (const void*)inputs : constants) +
               copy.source_offset;
    }
    iree_hal_amd_gpu_kernarg_writer_zero(&writer, offset - position);
    iree_hal_amd_gpu_kernarg_writer_append(&writer, source, length);
    position = offset + length;
  }
  iree_hal_amd_gpu_kernarg_writer_zero(&writer, layout->byte_length - position);
  if (writer.pending_length) {
    iree_hal_amd_gpu_kernarg_writer_flush(&writer);
  }
}

static void iree_hal_amd_gpu_kernarg_writer_native_range(
    iree_hal_amd_gpu_kernarg_writer_t* writer, iree_const_byte_span_t constants,
    uint32_t begin, uint32_t end) {
  if (begin < constants.data_length) {
    const uint32_t copy_end = iree_min(end, constants.data_length);
    iree_hal_amd_gpu_kernarg_writer_append(writer, constants.data + begin,
                                           copy_end - begin);
    begin = copy_end;
  }
  iree_hal_amd_gpu_kernarg_writer_zero(writer, end - begin);
}

void iree_hal_amd_gpu_kernarg_layout_emplace_native_stream(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_const_byte_span_t constants,
    const iree_hal_amd_gpu_kernarg_inputs_t* inputs, void* target) {
  // A complete native ABI without hidden inputs is already its final image.
  // Whole DWORDs need neither partial-word assembly nor a zero-filled suffix.
  if (!layout->runtime_copy_count &&
      constants.data_length >= layout->byte_length &&
      !(layout->byte_length & 3)) {
    iree_memcpy_stream_dst(target, constants.data, layout->byte_length);
    return;
  }
  iree_hal_amd_gpu_kernarg_writer_t writer = {.target = target};
  const iree_hal_amd_gpu_kernarg_copy_t* copies =
      iree_hal_amd_gpu_kernarg_copies(layout) + layout->constant_copy_count;
  uint32_t position = 0;
  for (uint16_t i = 0; i < layout->runtime_copy_count; ++i) {
    const iree_hal_amd_gpu_kernarg_copy_t copy = copies[i];
    iree_hal_amd_gpu_kernarg_writer_native_range(&writer, constants, position,
                                                 copy.target_offset);
    iree_hal_amd_gpu_kernarg_writer_append(
        &writer, (const uint8_t*)inputs + copy.source_offset, copy.length);
    position = copy.target_offset + copy.length;
  }
  iree_hal_amd_gpu_kernarg_writer_native_range(&writer, constants, position,
                                               layout->byte_length);
  if (writer.pending_length) {
    iree_hal_amd_gpu_kernarg_writer_flush(&writer);
  }
}
