// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/image/image.h"

#include <string.h>

#include "iree/base/alignment.h"

void iree_hal_amd_gpu_image_deinitialize(iree_hal_amd_gpu_image_t* image) {
  iree_hal_amd_gpu_metadata_deinitialize(&image->metadata);
  iree_allocator_free(image->host_allocator, image->asan_data_objects);
  iree_allocator_free(image->host_allocator, image->reflection_storage);
  iree_allocator_free(image->host_allocator, image->storage);
  iree_byte_sequence_release(image->source);
  memset(image, 0, sizeof(*image));
}

const iree_hal_amd_gpu_image_symbol_t* iree_hal_amd_gpu_image_find_symbol(
    const iree_hal_amd_gpu_image_t* image, iree_string_view_t name) {
  iree_host_size_t low = 0;
  iree_host_size_t high = image->symbol_count;
  while (low < high) {
    iree_host_size_t middle = low + (high - low) / 2;
    int order = iree_string_view_compare(name, image->symbols[middle].name);
    if (order == 0) {
      return &image->symbols[middle];
    }
    if (order < 0) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return NULL;
}

uint64_t iree_hal_amd_gpu_image_resolve_address(
    iree_hal_amd_gpu_image_address_t reference, uint64_t base_address,
    const iree_hal_amd_gpu_image_import_binding_t* bindings) {
  switch (reference.kind) {
    case IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMAGE:
      return base_address + reference.offset;
    case IREE_HAL_AMD_GPU_IMAGE_ADDRESS_ABSOLUTE:
      return reference.offset;
    case IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMPORT:
      return (bindings[reference.import_ordinal].resolved
                  ? bindings[reference.import_ordinal].address
                  : 0) +
             reference.offset;
    default:
      IREE_CHECK_UNREACHABLE("invalid admitted image address kind");
  }
}

typedef struct iree_hal_amd_gpu_image_copy_state_t {
  // Admitted source ranges and target offsets.
  const iree_hal_amd_gpu_image_t* image;
  // Writable mappings borrowed for this load.
  const iree_byte_span_t* mappings;
  // Logical offset of the next enumerated source span.
  uint64_t source_offset;
} iree_hal_amd_gpu_image_copy_state_t;

static iree_status_t iree_hal_amd_gpu_image_copy_segment(
    void* user_data, iree_const_byte_span_t source) {
  iree_hal_amd_gpu_image_copy_state_t* state = user_data;
  uint64_t source_end = state->source_offset + source.data_length;
  for (iree_host_size_t i = 0; i < state->image->segment_count; ++i) {
    const iree_hal_amd_gpu_image_segment_t* segment =
        &state->image->segments[i];
    uint64_t begin = iree_max(state->source_offset, segment->source_offset);
    uint64_t end =
        iree_min(source_end, segment->source_offset + segment->source_length);
    if (begin < end) {
      memcpy(state->mappings[i].data + (begin - segment->source_offset),
             source.data + (begin - state->source_offset), end - begin);
    }
  }
  state->source_offset = source_end;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_image_load(
    const iree_hal_amd_gpu_image_t* image, uint64_t base_address,
    iree_host_size_t binding_count,
    const iree_hal_amd_gpu_image_import_binding_t* bindings,
    iree_host_size_t mapping_count, const iree_byte_span_t* mappings) {
  if ((base_address & (image->alignment - 1)) != 0 ||
      image->byte_length > UINT64_MAX - base_address) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU image placement is misaligned or overflows");
  }
  if (mapping_count != image->segment_count || (mapping_count && !mappings) ||
      binding_count != image->import_count || (binding_count && !bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU image requires one mapping per segment and "
                            "one binding per import");
  }
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    if (!mappings[i].data ||
        mappings[i].data_length < image->segments[i].length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "GPU image segment %" PRIhsz " has insufficient writable storage", i);
    }
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (!bindings[i].resolved && !image->imports[i].weak) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND, "GPU image import `%.*s` is unresolved",
          (int)image->imports[i].name.size, image->imports[i].name.data);
    }
  }
  // Narrow absolute relocations can only be checked after placement/imports.
  for (iree_host_size_t i = 0; i < image->relocation_count; ++i) {
    const iree_hal_amd_gpu_image_relocation_t* relocation =
        &image->relocations[i];
    if (relocation->kind == IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_32 &&
        iree_hal_amd_gpu_image_resolve_address(relocation->value, base_address,
                                               bindings) > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "GPU image relocation does not fit 32 bits");
    }
  }

  iree_hal_amd_gpu_image_copy_state_t state = {
      .image = image,
      .mappings = mappings,
  };
  IREE_RETURN_IF_ERROR(iree_byte_sequence_enumerate(
      image->source,
      (iree_byte_sequence_segment_callback_t){
          .fn = iree_hal_amd_gpu_image_copy_segment, .user_data = &state}));
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    const iree_hal_amd_gpu_image_segment_t* segment = &image->segments[i];
    memset(mappings[i].data + segment->source_length, 0,
           segment->length - segment->source_length);
  }
  for (iree_host_size_t i = 0; i < image->relocation_count; ++i) {
    const iree_hal_amd_gpu_image_relocation_t* relocation =
        &image->relocations[i];
    uint8_t* target =
        mappings[relocation->segment_ordinal].data + relocation->offset;
    uint64_t value = iree_hal_amd_gpu_image_resolve_address(
        relocation->value, base_address, bindings);
    switch (relocation->kind) {
      case IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_64:
        iree_unaligned_store_le_u64(target, value);
        break;
      case IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_32:
      case IREE_HAL_AMD_GPU_IMAGE_RELOCATION_LOW_32:
        iree_unaligned_store_le_u32(target, (uint32_t)value);
        break;
      case IREE_HAL_AMD_GPU_IMAGE_RELOCATION_HIGH_32:
        iree_unaligned_store_le_u32(target, (uint32_t)(value >> 32));
        break;
      default:
        IREE_CHECK_UNREACHABLE("invalid admitted image relocation kind");
    }
  }
  return iree_ok_status();
}
