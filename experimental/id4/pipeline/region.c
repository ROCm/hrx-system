// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/region.h"

#include <string.h>

#define ID4_PIPELINE_REGION_INVALID_TENSOR_ORDINAL UINT32_MAX

typedef struct id4_pipeline_region_free_range_t {
  // Byte offset into the local slab.
  iree_device_size_t offset;
  // Byte length of the reusable range.
  iree_device_size_t length;
} id4_pipeline_region_free_range_t;

typedef struct id4_pipeline_region_tensor_record_t {
  // Value handle returned to stage code.
  id4_pipeline_tensor_t tensor;
  // Tensor name copied into the builder arena.
  iree_string_view_t name;
  // Operation ordinal where the tensor was acquired or imported.
  iree_host_size_t acquire_operation_ordinal;
  // Epoch where the tensor was acquired or imported.
  uint32_t acquire_epoch;
  // Operation ordinal where the tensor was released.
  iree_host_size_t release_operation_ordinal;
  // Epoch where the tensor was released.
  uint32_t release_epoch;
  // Access flags observed in access_epoch.
  id4_pipeline_tensor_access_flags_t epoch_access;
  // Epoch that epoch_access describes.
  uint32_t access_epoch;
  // True when the tensor contents may be read.
  bool initialized;
  // True after a local tensor has been released.
  bool released;
  // True after a released local range has been returned to the free list.
  bool range_published;
} id4_pipeline_region_tensor_record_t;

struct id4_pipeline_region_builder_t {
  // Host allocator used for builder storage.
  iree_allocator_t host_allocator;
  // Arena for transient region metadata.
  iree_arena_allocator_t arena;
  // Region name copied into the builder arena.
  iree_string_view_t region_name;
  // Builder execution mode.
  id4_pipeline_region_builder_mode_t mode;
  // Builder behavior flags.
  id4_pipeline_region_builder_flags_t flags;
  // HAL command buffer borrowed in RECORD mode.
  iree_hal_command_buffer_t* command_buffer;
  // Exact issue-time binding-table capacity for the region.
  iree_host_size_t binding_capacity;
  // Binding-table slot reserved for the local transient slab.
  uint32_t local_binding_slot;
  // Accumulated region statistics.
  id4_pipeline_region_statistics_t statistics;
  // Tensor record array allocated from the arena.
  id4_pipeline_region_tensor_record_t* tensor_records;
  // Number of live tensor records.
  iree_host_size_t tensor_record_count;
  // Allocated tensor record capacity.
  iree_host_size_t tensor_record_capacity;
  // Reusable local slab ranges allocated from the arena.
  id4_pipeline_region_free_range_t* free_ranges;
  // Number of reusable local slab ranges.
  iree_host_size_t free_range_count;
  // Allocated reusable local slab range capacity.
  iree_host_size_t free_range_capacity;
  // Next bump-pointer offset in the local slab.
  iree_device_size_t local_next_offset;
};

static iree_status_t id4_pipeline_region_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_region_copy_string(
    id4_pipeline_region_builder_t* builder, iree_string_view_t source,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return iree_ok_status();
  }
  char* target = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&builder->arena, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_string = iree_make_string_view(target, source.size);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_validate_shape(
    id4_pipeline_tensor_shape_t shape, iree_string_view_t tensor_name) {
  if (shape.rank > ID4_PIPELINE_TENSOR_MAX_RANK) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor %.*s rank %u exceeds max rank %u",
                            (int)tensor_name.size, tensor_name.data, shape.rank,
                            ID4_PIPELINE_TENSOR_MAX_RANK);
  }
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (shape.dims[i] == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tensor %.*s dimension %u is zero",
                              (int)tensor_name.size, tensor_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_validate_layout(
    const id4_pipeline_tensor_layout_t* layout) {
  if (!layout) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout is required");
  }
  if (iree_string_view_is_empty(layout->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout name is required");
  }
  if (layout->byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor %.*s byte length is zero",
                            (int)layout->name.size, layout->name.data);
  }
  if (layout->alignment != 0 &&
      !iree_device_size_is_power_of_two(layout->alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor %.*s alignment must be a power of two",
                            (int)layout->name.size, layout->name.data);
  }
  return id4_pipeline_region_validate_shape(layout->shape, layout->name);
}

static iree_status_t id4_pipeline_region_validate_binding_slot(
    const id4_pipeline_region_builder_t* builder, uint32_t binding_slot,
    iree_string_view_t tensor_name) {
  if (binding_slot < builder->binding_capacity) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_OUT_OF_RANGE,
      "tensor %.*s binding slot %u exceeds capacity %" PRIhsz,
      (int)tensor_name.size, tensor_name.data, binding_slot,
      builder->binding_capacity);
}

static iree_status_t id4_pipeline_region_validate_access(
    id4_pipeline_tensor_access_flags_t access, iree_string_view_t kernel_name,
    iree_host_size_t binding_index) {
  if (access == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz " access flags are required",
        (int)kernel_name.size, kernel_name.data, binding_index);
  }
  const id4_pipeline_tensor_access_flags_t allowed =
      ID4_PIPELINE_TENSOR_ACCESS_READ | ID4_PIPELINE_TENSOR_ACCESS_WRITE;
  if (iree_any_bit_set(access, ~allowed)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz " has unsupported access flags 0x%x",
        (int)kernel_name.size, kernel_name.data, binding_index, access);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_reserve_tensor_records(
    id4_pipeline_region_builder_t* builder, iree_host_size_t capacity) {
  if (capacity <= builder->tensor_record_capacity) return iree_ok_status();
  return iree_arena_grow_array(&builder->arena, builder->tensor_record_count,
                               capacity, sizeof(builder->tensor_records[0]),
                               &builder->tensor_record_capacity,
                               (void**)&builder->tensor_records);
}

static iree_status_t id4_pipeline_region_reserve_free_ranges(
    id4_pipeline_region_builder_t* builder, iree_host_size_t capacity) {
  if (capacity <= builder->free_range_capacity) return iree_ok_status();
  return iree_arena_grow_array(&builder->arena, builder->free_range_count,
                               capacity, sizeof(builder->free_ranges[0]),
                               &builder->free_range_capacity,
                               (void**)&builder->free_ranges);
}

static void id4_pipeline_region_remove_free_range(
    id4_pipeline_region_builder_t* builder, iree_host_size_t range_index) {
  IREE_ASSERT(range_index < builder->free_range_count);
  builder->free_ranges[range_index] =
      builder->free_ranges[builder->free_range_count - 1];
  --builder->free_range_count;
}

static iree_status_t id4_pipeline_region_append_free_range(
    id4_pipeline_region_builder_t* builder, iree_device_size_t offset,
    iree_device_size_t length) {
  if (length == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_free_ranges(
      builder, builder->free_range_count + 1));
  builder->free_ranges[builder->free_range_count++] =
      (id4_pipeline_region_free_range_t){
          // Byte offset into the local slab.
          .offset = offset,
          // Byte length of the reusable range.
          .length = length,
      };
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_try_acquire_free_range(
    id4_pipeline_region_builder_t* builder, iree_device_size_t byte_length,
    iree_device_size_t alignment, iree_device_size_t* out_offset,
    bool* out_found) {
  *out_offset = 0;
  *out_found = false;
  if (iree_all_bits_set(builder->flags,
                        ID4_PIPELINE_REGION_BUILDER_FLAG_DISABLE_LOCAL_REUSE)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < builder->free_range_count; ++i) {
    id4_pipeline_region_free_range_t range = builder->free_ranges[i];
    iree_device_size_t aligned_offset = 0;
    if (!iree_device_size_checked_align(range.offset, alignment,
                                        &aligned_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "free range alignment overflow");
    }
    iree_device_size_t range_end = 0;
    if (!iree_device_size_checked_add(range.offset, range.length, &range_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "free range length overflow");
    }
    iree_device_size_t allocation_end = 0;
    if (!iree_device_size_checked_add(aligned_offset, byte_length,
                                      &allocation_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "free range allocation overflow");
    }
    if (allocation_end > range_end) continue;

    const iree_device_size_t prefix_length = aligned_offset - range.offset;
    const iree_device_size_t suffix_length = range_end - allocation_end;
    if (prefix_length && suffix_length) {
      builder->free_ranges[i].length = prefix_length;
      IREE_RETURN_IF_ERROR(id4_pipeline_region_append_free_range(
          builder, allocation_end, suffix_length));
    } else if (prefix_length) {
      builder->free_ranges[i].length = prefix_length;
    } else if (suffix_length) {
      builder->free_ranges[i].offset = allocation_end;
      builder->free_ranges[i].length = suffix_length;
    } else {
      id4_pipeline_region_remove_free_range(builder, i);
    }
    *out_offset = aligned_offset;
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_bump_local_range(
    id4_pipeline_region_builder_t* builder, iree_device_size_t byte_length,
    iree_device_size_t alignment, iree_device_size_t* out_offset) {
  iree_device_size_t aligned_offset = 0;
  if (!iree_device_size_checked_align(builder->local_next_offset, alignment,
                                      &aligned_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local slab alignment overflow");
  }
  iree_device_size_t allocation_end = 0;
  if (!iree_device_size_checked_add(aligned_offset, byte_length,
                                    &allocation_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local slab allocation overflow");
  }
  builder->local_next_offset = allocation_end;
  builder->statistics.local_slab_byte_length =
      iree_max(builder->statistics.local_slab_byte_length, allocation_end);
  builder->statistics.local_slab_high_water_mark =
      builder->statistics.local_slab_byte_length;
  *out_offset = aligned_offset;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_make_tensor_record(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_tensor_storage_class_t storage_class,
    const id4_pipeline_tensor_layout_t* layout, uint32_t binding_slot,
    iree_device_size_t offset, bool initialized,
    id4_pipeline_region_tensor_record_t** out_record) {
  *out_record = NULL;
  if (builder->tensor_record_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many region tensors");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_tensor_records(
      builder, builder->tensor_record_count + 1));
  id4_pipeline_region_tensor_record_t* record =
      &builder->tensor_records[builder->tensor_record_count];
  memset(record, 0, sizeof(*record));
  record->tensor = (id4_pipeline_tensor_t){
      // Tensor storage class.
      .storage_class = storage_class,
      // Builder-local tensor ordinal.
      .ordinal = (uint32_t)builder->tensor_record_count,
      // Issue-time binding-table slot containing this tensor.
      .binding_slot = binding_slot,
      // Byte offset into the binding-table slot.
      .offset = offset,
      // Byte length visible to dispatches.
      .length = layout->byte_length,
      // Tensor shape.
      .shape = layout->shape,
  };
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_copy_string(builder, layout->name, &record->name));
  record->acquire_operation_ordinal = builder->statistics.operation_count;
  record->acquire_epoch = builder->statistics.current_epoch;
  record->release_operation_ordinal = IREE_HOST_SIZE_MAX;
  record->release_epoch = UINT32_MAX;
  record->access_epoch = UINT32_MAX;
  record->initialized = initialized;
  ++builder->tensor_record_count;
  *out_record = record;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_lookup_tensor_record(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t tensor,
    id4_pipeline_region_tensor_record_t** out_record) {
  *out_record = NULL;
  if (tensor.storage_class == ID4_PIPELINE_TENSOR_STORAGE_CLASS_INVALID ||
      tensor.ordinal == ID4_PIPELINE_REGION_INVALID_TENSOR_ORDINAL ||
      tensor.ordinal >= builder->tensor_record_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor handle is not known to this region");
  }
  id4_pipeline_region_tensor_record_t* record =
      &builder->tensor_records[tensor.ordinal];
  if (record->tensor.storage_class != tensor.storage_class ||
      record->tensor.binding_slot != tensor.binding_slot ||
      record->tensor.offset != tensor.offset ||
      record->tensor.length != tensor.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor handle does not match region record");
  }
  *out_record = record;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_publish_released_ranges(
    id4_pipeline_region_builder_t* builder) {
  if (iree_all_bits_set(builder->flags,
                        ID4_PIPELINE_REGION_BUILDER_FLAG_DISABLE_LOCAL_REUSE)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < builder->tensor_record_count; ++i) {
    id4_pipeline_region_tensor_record_t* record = &builder->tensor_records[i];
    if (record->tensor.storage_class !=
            ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL ||
        !record->released || record->range_published ||
        record->release_epoch >= builder->statistics.current_epoch) {
      continue;
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_region_append_free_range(
        builder, record->tensor.offset, record->tensor.length));
    record->range_published = true;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_validate_builder_create_options(
    const id4_pipeline_region_builder_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region builder create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("region builder")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "region builder extension structures are not "
                            "supported");
  }
  if (iree_string_view_is_empty(options->region_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region name is required");
  }
  if (!options->block_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region builder block pool is required");
  }
  if (options->binding_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region binding capacity must be nonzero");
  }
  if (options->local_binding_slot >= options->binding_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local binding slot %u exceeds capacity %" PRIhsz,
                            options->local_binding_slot,
                            options->binding_capacity);
  }
  switch (options->mode) {
    case ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN:
      if (options->command_buffer) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dry-run region builders must not provide a HAL command buffer");
      }
      return iree_ok_status();
    case ID4_PIPELINE_REGION_BUILDER_MODE_RECORD:
      if (!options->command_buffer) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "record region builders require a HAL command buffer");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported region builder mode %d",
                              (int)options->mode);
  }
}

iree_status_t id4_pipeline_region_builder_create(
    const id4_pipeline_region_builder_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_region_builder_t** out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  *out_builder = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_validate_builder_create_options(options));

  id4_pipeline_region_builder_t* builder = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*builder),
                                             (void**)&builder));
  memset(builder, 0, sizeof(*builder));
  builder->host_allocator = host_allocator;
  iree_arena_initialize(options->block_pool, &builder->arena);
  builder->mode = options->mode;
  builder->flags = options->flags;
  builder->command_buffer = options->command_buffer;
  builder->binding_capacity = options->binding_capacity;
  builder->local_binding_slot = options->local_binding_slot;

  iree_status_t status = id4_pipeline_region_copy_string(
      builder, options->region_name, &builder->region_name);
  if (iree_status_is_ok(status)) {
    *out_builder = builder;
  } else {
    id4_pipeline_region_builder_destroy(builder);
  }
  return status;
}

void id4_pipeline_region_builder_destroy(
    id4_pipeline_region_builder_t* builder) {
  if (!builder) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  iree_arena_deinitialize(&builder->arena);
  iree_allocator_free(host_allocator, builder);
}

id4_pipeline_region_builder_mode_t id4_pipeline_region_builder_mode(
    const id4_pipeline_region_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  return builder->mode;
}

uint32_t id4_pipeline_region_builder_current_epoch(
    const id4_pipeline_region_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  return builder->statistics.current_epoch;
}

void id4_pipeline_region_builder_statistics(
    const id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_statistics);
  *out_statistics = builder->statistics;
}

iree_status_t id4_pipeline_region_acquire_tensor(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_tensor_layout_t* layout,
    id4_pipeline_tensor_t* out_tensor) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_tensor);
  *out_tensor = (id4_pipeline_tensor_t){
      // Tensor storage class.
      .storage_class = ID4_PIPELINE_TENSOR_STORAGE_CLASS_INVALID,
      // Invalid tensor ordinal.
      .ordinal = ID4_PIPELINE_REGION_INVALID_TENSOR_ORDINAL,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_layout(layout));

  const iree_device_size_t alignment =
      layout->alignment == 0 ? 1 : layout->alignment;
  iree_device_size_t offset = 0;
  bool found_reusable_range = false;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_try_acquire_free_range(
      builder, layout->byte_length, alignment, &offset, &found_reusable_range));
  if (!found_reusable_range) {
    IREE_RETURN_IF_ERROR(id4_pipeline_region_bump_local_range(
        builder, layout->byte_length, alignment, &offset));
  } else {
    ++builder->statistics.local_reuse_count;
  }

  id4_pipeline_region_tensor_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_make_tensor_record(
      builder, ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL, layout,
      builder->local_binding_slot, offset, /*initialized=*/false, &record));
  ++builder->statistics.local_acquire_count;
  *out_tensor = record->tensor;
  return iree_ok_status();
}

iree_status_t id4_pipeline_region_import_tensor(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_tensor_import_t* import,
    id4_pipeline_tensor_t* out_tensor) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_tensor);
  *out_tensor = (id4_pipeline_tensor_t){
      // Tensor storage class.
      .storage_class = ID4_PIPELINE_TENSOR_STORAGE_CLASS_INVALID,
      // Invalid tensor ordinal.
      .ordinal = ID4_PIPELINE_REGION_INVALID_TENSOR_ORDINAL,
  };
  if (!import) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor import is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_layout(&import->layout));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_binding_slot(
      builder, import->binding_slot, import->layout.name));
  if (import->binding_slot == builder->local_binding_slot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor %.*s import uses the local binding slot",
                            (int)import->layout.name.size,
                            import->layout.name.data);
  }
  const iree_device_size_t alignment =
      import->layout.alignment == 0 ? 1 : import->layout.alignment;
  if ((import->offset % alignment) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "tensor %.*s offset is not aligned",
        (int)import->layout.name.size, import->layout.name.data);
  }

  const bool initialized = iree_any_bit_set(
      import->flags, ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED);
  id4_pipeline_region_tensor_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_make_tensor_record(
      builder, ID4_PIPELINE_TENSOR_STORAGE_CLASS_BOUND, &import->layout,
      import->binding_slot, import->offset, initialized, &record));
  ++builder->statistics.bound_import_count;
  *out_tensor = record->tensor;
  return iree_ok_status();
}

iree_status_t id4_pipeline_region_release_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t tensor) {
  IREE_ASSERT_ARGUMENT(builder);
  id4_pipeline_region_tensor_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_lookup_tensor_record(builder, tensor, &record));
  if (record->tensor.storage_class != ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only local tensors can be released");
  }
  if (record->released) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "tensor %.*s has already been released",
                            (int)record->name.size, record->name.data);
  }
  record->released = true;
  record->release_operation_ordinal = builder->statistics.operation_count;
  record->release_epoch = builder->statistics.current_epoch;
  ++builder->statistics.local_release_count;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_validate_dispatch_binding(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_kernel_t* kernel,
    const id4_pipeline_region_dispatch_binding_t* binding,
    iree_host_size_t binding_index,
    id4_pipeline_region_tensor_record_t** out_record) {
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_access(
      binding->access, kernel->name, binding_index));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_lookup_tensor_record(
      builder, binding->tensor, out_record));
  id4_pipeline_region_tensor_record_t* record = *out_record;
  if (record->released) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel %.*s binding %" PRIhsz " uses released tensor %.*s",
        (int)kernel->name.size, kernel->name.data, binding_index,
        (int)record->name.size, record->name.data);
  }
  if (iree_all_bits_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_READ) &&
      !record->initialized) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel %.*s binding %" PRIhsz " reads uninitialized tensor %.*s",
        (int)kernel->name.size, kernel->name.data, binding_index,
        (int)record->name.size, record->name.data);
  }
  const bool same_epoch =
      record->access_epoch == builder->statistics.current_epoch;
  if (!same_epoch) return iree_ok_status();
  if (iree_any_bit_set(record->epoch_access,
                       ID4_PIPELINE_TENSOR_ACCESS_WRITE) &&
      iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_READ)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "kernel %.*s binding %" PRIhsz
                            " reads tensor %.*s written in the same epoch",
                            (int)kernel->name.size, kernel->name.data,
                            binding_index, (int)record->name.size,
                            record->name.data);
  }
  if (iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE) &&
      record->epoch_access != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "kernel %.*s binding %" PRIhsz
                            " writes tensor %.*s already used in the same "
                            "epoch",
                            (int)kernel->name.size, kernel->name.data,
                            binding_index, (int)record->name.size,
                            record->name.data);
  }
  return iree_ok_status();
}

static void id4_pipeline_region_apply_dispatch_binding_access(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_tensor_record_t* record,
    id4_pipeline_tensor_access_flags_t access) {
  if (record->access_epoch != builder->statistics.current_epoch) {
    record->access_epoch = builder->statistics.current_epoch;
    record->epoch_access = 0;
  }
  record->epoch_access |= access;
  if (iree_any_bit_set(access, ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
    record->initialized = true;
  }
}

static iree_status_t id4_pipeline_region_validate_kernel(
    const id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_kernel_t* kernel,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings) {
  if (!kernel) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region dispatch kernel is required");
  }
  if (iree_string_view_is_empty(kernel->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region dispatch kernel name is required");
  }
  if (binding_count != kernel->binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel %.*s expected %" PRIhsz
                            " bindings but got %" PRIhsz,
                            (int)kernel->name.size, kernel->name.data,
                            kernel->binding_count, binding_count);
  }
  if (binding_count != 0 && !bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel %.*s dispatch bindings are required",
                            (int)kernel->name.size, kernel->name.data);
  }
  if (constants.data_length != kernel->constant_byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s expected %" PRIhsz " constant bytes but got %" PRIhsz,
        (int)kernel->name.size, kernel->name.data, kernel->constant_byte_length,
        constants.data_length);
  }
  if (builder->mode == ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    if (!kernel->executable) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel %.*s HAL executable is required in "
                              "record mode",
                              (int)kernel->name.size, kernel->name.data);
    }
    if (!iree_hal_executable_function_is_valid(kernel->function)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel %.*s HAL function is required in record "
                              "mode",
                              (int)kernel->name.size, kernel->name.data);
    }
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_region_dispatch(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_kernel_t* kernel,
    iree_hal_dispatch_config_t dispatch_config,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_kernel(
      builder, kernel, constants, binding_count, bindings));

  id4_pipeline_region_tensor_record_t** records = NULL;
  if (binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &builder->arena, binding_count, sizeof(records[0]), (void**)&records));
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_dispatch_binding(
        builder, kernel, &bindings[i], i, &records[i]));
  }

  if (builder->mode == ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    iree_hal_buffer_ref_t* hal_bindings = NULL;
    if (binding_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &builder->arena, binding_count, sizeof(hal_bindings[0]),
          (void**)&hal_bindings));
    }
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      hal_bindings[i] = iree_hal_make_indirect_buffer_ref(
          bindings[i].tensor.binding_slot, bindings[i].tensor.offset,
          bindings[i].tensor.length);
    }
    iree_hal_buffer_ref_list_t hal_binding_list = {
        // Number of HAL buffer references.
        .count = binding_count,
        // HAL buffer references allocated from the builder arena.
        .values = hal_bindings,
    };
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
        builder->command_buffer, kernel->executable, kernel->function,
        dispatch_config, constants, hal_binding_list, flags));
  }

  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    id4_pipeline_region_apply_dispatch_binding_access(builder, records[i],
                                                      bindings[i].access);
  }
  ++builder->statistics.operation_count;
  ++builder->statistics.dispatch_count;
  return iree_ok_status();
}

iree_status_t id4_pipeline_region_barrier(
    id4_pipeline_region_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_ASSERT_ARGUMENT(builder);
  if (memory_barrier_count != 0 && !memory_barriers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory barriers are required");
  }
  if (buffer_barrier_count != 0 && !buffer_barriers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer barriers are required");
  }
  if (builder->statistics.current_epoch == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "region epoch overflow");
  }
  if (builder->mode == ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
        builder->command_buffer, source_stage_mask, target_stage_mask, flags,
        memory_barrier_count, memory_barriers, buffer_barrier_count,
        buffer_barriers));
  }
  ++builder->statistics.operation_count;
  ++builder->statistics.barrier_count;
  ++builder->statistics.current_epoch;
  return id4_pipeline_region_publish_released_ranges(builder);
}
