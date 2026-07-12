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
  // Scalar element type.
  id4_pipeline_tensor_dtype_t dtype;
  // Tensor ordinal owning the backing storage allocation.
  uint32_t storage_root_ordinal;
  // Number of unreleased logical views retaining this root allocation.
  uint32_t active_view_count;
  // Required base alignment in bytes.
  iree_device_size_t alignment;
  // Operation ordinal where the tensor was acquired or imported.
  iree_host_size_t acquire_operation_ordinal;
  // Epoch where the tensor was acquired or imported.
  uint32_t acquire_epoch;
  // Operation ordinal where the tensor was released.
  iree_host_size_t release_operation_ordinal;
  // Epoch where the tensor was released.
  uint32_t release_epoch;
  // Operation ordinal where the root storage became dead.
  iree_host_size_t storage_release_operation_ordinal;
  // Epoch where the root storage became dead.
  uint32_t storage_release_epoch;
  // Access flags observed in access_epoch.
  id4_pipeline_tensor_access_flags_t epoch_access;
  // Epoch that epoch_access describes.
  uint32_t access_epoch;
  // Write coverage ranges observed in access_epoch.
  id4_pipeline_region_tensor_byte_range_t* epoch_write_ranges;
  // Number of write coverage ranges observed in access_epoch.
  iree_host_size_t epoch_write_range_count;
  // Allocated capacity of epoch_write_ranges.
  iree_host_size_t epoch_write_range_capacity;
  // Read coverage ranges observed in access_epoch.
  id4_pipeline_region_tensor_byte_range_t* epoch_read_ranges;
  // Number of read coverage ranges observed in access_epoch.
  iree_host_size_t epoch_read_range_count;
  // Allocated capacity of epoch_read_ranges.
  iree_host_size_t epoch_read_range_capacity;
  // Write coverage ranges accumulated across epochs before full initialization.
  id4_pipeline_region_tensor_byte_range_t* initialized_write_ranges;
  // Number of accumulated initialization write coverage ranges.
  iree_host_size_t initialized_write_range_count;
  // Allocated capacity of initialized_write_ranges.
  iree_host_size_t initialized_write_range_capacity;
  // True when the tensor contents may be read.
  bool initialized;
  // True after a local tensor has been released.
  bool released;
  // True after a destructive alias write invalidates this logical tensor.
  bool consumed;
  // True after a released local range has been returned to the free list.
  bool range_published;
  // Local lifetime flags used for memory diagnostics.
  id4_pipeline_region_local_lifetime_flags_t lifetime_flags;
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
  // Authored Loom kernel specialization records allocated from the arena.
  id4_pipeline_region_kernel_plan_t* kernel_plans;
  // Number of authored Loom kernel specialization records.
  iree_host_size_t kernel_plan_count;
  // Allocated Loom kernel specialization record capacity.
  iree_host_size_t kernel_plan_capacity;
  // Reusable local slab ranges allocated from the arena.
  id4_pipeline_region_free_range_t* free_ranges;
  // Number of reusable local slab ranges.
  iree_host_size_t free_range_count;
  // Allocated reusable local slab range capacity.
  iree_host_size_t free_range_capacity;
  // Next bump-pointer offset in the local slab.
  iree_device_size_t local_next_offset;
  // Current sum of live local tensor byte lengths.
  iree_device_size_t local_live_byte_length;
};

struct id4_pipeline_prepared_region_t {
  // Reference count for shared prepared-region ownership.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for prepared-region storage.
  iree_allocator_t host_allocator;
  // Retained device group that keeps topology state valid.
  iree_hal_device_group_t* device_group;
  // Borrowed device selected from the retained device group.
  iree_hal_device_t* device;
  // Retained command buffer recorded by the source builder.
  iree_hal_command_buffer_t* command_buffer;
  // Retained allocation pool for the local transient slab.
  iree_hal_pool_t* local_slab_pool;
  // Queue affinity used by all region queue operations.
  iree_hal_queue_affinity_t queue_affinity;
  // HAL buffer parameters for the local transient slab.
  iree_hal_buffer_params_t local_slab_params;
  // HAL queue-alloca flags for the local transient slab.
  iree_hal_alloca_flags_t local_slab_alloca_flags;
  // HAL queue-dealloca flags for the local transient slab.
  iree_hal_dealloca_flags_t local_slab_dealloca_flags;
  // Exact issue-time binding-table capacity.
  iree_host_size_t binding_capacity;
  // Binding-table slot reserved for the local transient slab.
  uint32_t local_binding_slot;
  // Realized region statistics copied from the source builder.
  id4_pipeline_region_statistics_t statistics;
};

iree_device_size_t id4_pipeline_tensor_dtype_byte_length(
    id4_pipeline_tensor_dtype_t dtype) {
  switch (dtype) {
    case ID4_PIPELINE_TENSOR_DTYPE_F32:
    case ID4_PIPELINE_TENSOR_DTYPE_I32:
    case ID4_PIPELINE_TENSOR_DTYPE_U32:
      return 4;
    case ID4_PIPELINE_TENSOR_DTYPE_F16:
    case ID4_PIPELINE_TENSOR_DTYPE_BF16:
      return 2;
    case ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3:
      return 1;
    default:
      return 0;
  }
}

iree_string_view_t id4_pipeline_tensor_dtype_format(
    id4_pipeline_tensor_dtype_t dtype) {
  switch (dtype) {
    case ID4_PIPELINE_TENSOR_DTYPE_F32:
      return IREE_SV("f32");
    case ID4_PIPELINE_TENSOR_DTYPE_F16:
      return IREE_SV("f16");
    case ID4_PIPELINE_TENSOR_DTYPE_BF16:
      return IREE_SV("bf16");
    case ID4_PIPELINE_TENSOR_DTYPE_I32:
      return IREE_SV("i32");
    case ID4_PIPELINE_TENSOR_DTYPE_U32:
      return IREE_SV("u32");
    case ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3:
      return IREE_SV("f8e4m3");
    default:
      return IREE_SV("invalid");
  }
}

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

static iree_status_t id4_pipeline_region_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
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
  const iree_device_size_t element_byte_length =
      id4_pipeline_tensor_dtype_byte_length(layout->dtype);
  if (element_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor %.*s dtype is invalid",
                            (int)layout->name.size, layout->name.data);
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
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_validate_shape(layout->shape, layout->name));
  uint64_t element_count = 1;
  for (uint32_t i = 0; i < layout->shape.rank; ++i) {
    if (element_count > UINT64_MAX / layout->shape.dims[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor %.*s element count overflow",
                              (int)layout->name.size, layout->name.data);
    }
    element_count *= layout->shape.dims[i];
  }
  if (element_count > UINT64_MAX / element_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor %.*s byte length overflow",
                            (int)layout->name.size, layout->name.data);
  }
  const iree_device_size_t expected_byte_length =
      (iree_device_size_t)(element_count * element_byte_length);
  if (layout->byte_length != expected_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor %.*s byte length %" PRIu64
                            " does not match dtype/shape byte length %" PRIu64,
                            (int)layout->name.size, layout->name.data,
                            (uint64_t)layout->byte_length,
                            (uint64_t)expected_byte_length);
  }
  return iree_ok_status();
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

static iree_status_t id4_pipeline_region_validate_dispatch_binding_flags(
    id4_pipeline_region_dispatch_binding_flags_t flags,
    iree_string_view_t kernel_name, iree_host_size_t binding_index) {
  const id4_pipeline_region_dispatch_binding_flags_t allowed_flags =
      ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_WRITE_RANGE |
      ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_READ_RANGE |
      ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_DESTRUCTIVE_ALIAS_WRITE;
  if (!iree_any_bit_set(flags, ~allowed_flags)) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "kernel %.*s binding %" PRIhsz " has unsupported flags 0x%x",
      (int)kernel_name.size, kernel_name.data, binding_index, flags);
}

static iree_status_t id4_pipeline_region_validate_subview_flags(
    id4_pipeline_region_subview_flags_t flags, iree_string_view_t tensor_name) {
  const id4_pipeline_region_subview_flags_t allowed_flags =
      ID4_PIPELINE_REGION_SUBVIEW_FLAG_DISCARD_CONTENTS;
  if (!iree_any_bit_set(flags, ~allowed_flags)) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "subview tensor %.*s has unsupported flags 0x%x",
                          (int)tensor_name.size, tensor_name.data, flags);
}

static id4_pipeline_region_tensor_byte_range_t
id4_pipeline_region_binding_write_range(
    const id4_pipeline_region_dispatch_binding_t* binding) {
  if (iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_WRITE_RANGE)) {
    return binding->write_range;
  }
  return (id4_pipeline_region_tensor_byte_range_t){
      // Byte offset from the start of the logical tensor.
      .offset = 0,
      // Byte length covering the full logical tensor.
      .length = binding->tensor.length,
  };
}

static id4_pipeline_region_tensor_byte_range_t
id4_pipeline_region_binding_read_range(
    const id4_pipeline_region_dispatch_binding_t* binding) {
  if (iree_all_bits_set(binding->flags,
                        ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_READ_RANGE)) {
    return binding->read_range;
  }
  return (id4_pipeline_region_tensor_byte_range_t){
      // Byte offset from the start of the logical tensor.
      .offset = 0,
      // Byte length covering the full logical tensor.
      .length = binding->tensor.length,
  };
}

static iree_status_t id4_pipeline_region_binding_storage_write_range(
    const id4_pipeline_region_dispatch_binding_t* binding,
    id4_pipeline_region_tensor_byte_range_t* out_range) {
  *out_range = id4_pipeline_region_binding_write_range(binding);
  if (!iree_device_size_checked_add(binding->tensor.offset, out_range->offset,
                                    &out_range->offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch write storage offset overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_binding_storage_read_range(
    const id4_pipeline_region_dispatch_binding_t* binding,
    id4_pipeline_region_tensor_byte_range_t* out_range) {
  *out_range = id4_pipeline_region_binding_read_range(binding);
  if (!iree_device_size_checked_add(binding->tensor.offset, out_range->offset,
                                    &out_range->offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch read storage offset overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_tensor_byte_range_end(
    id4_pipeline_region_tensor_byte_range_t range,
    iree_device_size_t* out_end) {
  if (!iree_device_size_checked_add(range.offset, range.length, out_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor byte range length overflow");
  }
  return iree_ok_status();
}

static bool id4_pipeline_region_tensor_byte_ranges_overlap(
    id4_pipeline_region_tensor_byte_range_t lhs,
    id4_pipeline_region_tensor_byte_range_t rhs) {
  const iree_device_size_t lhs_end = lhs.offset + lhs.length;
  const iree_device_size_t rhs_end = rhs.offset + rhs.length;
  return lhs.offset < rhs_end && rhs.offset < lhs_end;
}

static iree_status_t id4_pipeline_region_validate_binding_write_range(
    const id4_pipeline_region_dispatch_binding_t* binding,
    iree_string_view_t kernel_name, iree_host_size_t binding_index) {
  if (!iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_WRITE_RANGE)) {
    return iree_ok_status();
  }
  if (!iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel %.*s binding %" PRIhsz
                            " has write coverage without write access",
                            (int)kernel_name.size, kernel_name.data,
                            binding_index);
  }
  if (binding->write_range.length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz " write coverage is empty",
        (int)kernel_name.size, kernel_name.data, binding_index);
  }
  iree_device_size_t end = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_tensor_byte_range_end(binding->write_range, &end));
  if (end <= binding->tensor.length) return iree_ok_status();
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "kernel %.*s binding %" PRIhsz
                          " write coverage [%" PRIu64 ", %" PRIu64
                          ") exceeds tensor length %" PRIu64,
                          (int)kernel_name.size, kernel_name.data,
                          binding_index, (uint64_t)binding->write_range.offset,
                          (uint64_t)end, (uint64_t)binding->tensor.length);
}

static iree_status_t id4_pipeline_region_validate_binding_read_range(
    const id4_pipeline_region_dispatch_binding_t* binding,
    iree_string_view_t kernel_name, iree_host_size_t binding_index) {
  if (!iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_READ_RANGE)) {
    return iree_ok_status();
  }
  if (!iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_READ)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz " has read coverage without read access",
        (int)kernel_name.size, kernel_name.data, binding_index);
  }
  if (binding->read_range.length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz " read coverage is empty",
        (int)kernel_name.size, kernel_name.data, binding_index);
  }
  iree_device_size_t end = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_tensor_byte_range_end(binding->read_range, &end));
  if (end <= binding->tensor.length) return iree_ok_status();
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "kernel %.*s binding %" PRIhsz
                          " read coverage [%" PRIu64 ", %" PRIu64
                          ") exceeds tensor length %" PRIu64,
                          (int)kernel_name.size, kernel_name.data,
                          binding_index, (uint64_t)binding->read_range.offset,
                          (uint64_t)end, (uint64_t)binding->tensor.length);
}

static iree_status_t id4_pipeline_region_validate_initialized_import_ranges(
    const id4_pipeline_tensor_import_t* import) {
  if (iree_all_bits_set(import->flags,
                        ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED) &&
      import->initialized_range_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "tensor %.*s import specifies both full and range initialization",
        (int)import->layout.name.size, import->layout.name.data);
  }
  if (import->initialized_range_count != 0 && !import->initialized_ranges) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "tensor %.*s import initialized ranges are required",
        (int)import->layout.name.size, import->layout.name.data);
  }
  for (iree_host_size_t i = 0; i < import->initialized_range_count; ++i) {
    const id4_pipeline_region_tensor_byte_range_t range =
        import->initialized_ranges[i];
    if (range.length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "tensor %.*s import initialized range %" PRIhsz " is empty",
          (int)import->layout.name.size, import->layout.name.data, i);
    }
    iree_device_size_t range_end = 0;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_region_tensor_byte_range_end(range, &range_end));
    if (range_end <= import->layout.byte_length) continue;
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "tensor %.*s import initialized range %" PRIhsz " [%" PRIu64
        ", %" PRIu64 ") exceeds tensor byte length %" PRIu64,
        (int)import->layout.name.size, import->layout.name.data, i,
        (uint64_t)range.offset, (uint64_t)range_end,
        (uint64_t)import->layout.byte_length);
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

static iree_status_t id4_pipeline_region_reserve_kernel_plans(
    id4_pipeline_region_builder_t* builder, iree_host_size_t capacity) {
  if (capacity <= builder->kernel_plan_capacity) return iree_ok_status();
  return iree_arena_grow_array(&builder->arena, builder->kernel_plan_count,
                               capacity, sizeof(builder->kernel_plans[0]),
                               &builder->kernel_plan_capacity,
                               (void**)&builder->kernel_plans);
}

static iree_status_t id4_pipeline_region_reserve_free_ranges(
    id4_pipeline_region_builder_t* builder, iree_host_size_t capacity) {
  if (capacity <= builder->free_range_capacity) return iree_ok_status();
  return iree_arena_grow_array(&builder->arena, builder->free_range_count,
                               capacity, sizeof(builder->free_ranges[0]),
                               &builder->free_range_capacity,
                               (void**)&builder->free_ranges);
}

static iree_status_t id4_pipeline_region_reserve_epoch_write_ranges(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_tensor_record_t* record, iree_host_size_t capacity) {
  if (capacity <= record->epoch_write_range_capacity) return iree_ok_status();
  return iree_arena_grow_array(&builder->arena, record->epoch_write_range_count,
                               capacity, sizeof(record->epoch_write_ranges[0]),
                               &record->epoch_write_range_capacity,
                               (void**)&record->epoch_write_ranges);
}

static iree_status_t id4_pipeline_region_reserve_initialized_write_ranges(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_tensor_record_t* record, iree_host_size_t capacity) {
  if (capacity <= record->initialized_write_range_capacity)
    return iree_ok_status();
  return iree_arena_grow_array(&builder->arena,
                               record->initialized_write_range_count, capacity,
                               sizeof(record->initialized_write_ranges[0]),
                               &record->initialized_write_range_capacity,
                               (void**)&record->initialized_write_ranges);
}

static iree_status_t id4_pipeline_region_reserve_epoch_read_ranges(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_tensor_record_t* record, iree_host_size_t capacity) {
  if (capacity <= record->epoch_read_range_capacity) return iree_ok_status();
  return iree_arena_grow_array(&builder->arena, record->epoch_read_range_count,
                               capacity, sizeof(record->epoch_read_ranges[0]),
                               &record->epoch_read_range_capacity,
                               (void**)&record->epoch_read_ranges);
}

static void id4_pipeline_region_remove_free_range(
    id4_pipeline_region_builder_t* builder, iree_host_size_t range_index) {
  IREE_ASSERT(range_index < builder->free_range_count);
  builder->free_ranges[range_index] =
      builder->free_ranges[builder->free_range_count - 1];
  --builder->free_range_count;
}

static iree_status_t id4_pipeline_region_string_clone(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  if (source.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region string is too large to clone");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void id4_pipeline_region_string_release(
    iree_string_view_t value, iree_allocator_t host_allocator) {
  if (value.data) {
    iree_allocator_free(host_allocator, (void*)value.data);
  }
}

static iree_status_t id4_pipeline_region_free_range_end(
    id4_pipeline_region_free_range_t range, iree_device_size_t* out_end) {
  if (!iree_device_size_checked_add(range.offset, range.length, out_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "free range length overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_append_free_range(
    id4_pipeline_region_builder_t* builder, iree_device_size_t offset,
    iree_device_size_t length) {
  if (length == 0) return iree_ok_status();
  iree_device_size_t range_offset = offset;
  iree_device_size_t range_end = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_free_range_end(
      (id4_pipeline_region_free_range_t){
          // Byte offset into the local slab.
          .offset = range_offset,
          // Byte length of the reusable range.
          .length = length,
      },
      &range_end));

  for (iree_host_size_t i = 0; i < builder->free_range_count;) {
    const id4_pipeline_region_free_range_t existing_range =
        builder->free_ranges[i];
    iree_device_size_t existing_end = 0;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_region_free_range_end(existing_range, &existing_end));
    if (range_end < existing_range.offset || existing_end < range_offset) {
      ++i;
      continue;
    }
    if (range_end != existing_range.offset && existing_end != range_offset) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "free range [%" PRIu64 ", %" PRIu64
          ") overlaps existing range [%" PRIu64 ", %" PRIu64 ")",
          (uint64_t)range_offset, (uint64_t)range_end,
          (uint64_t)existing_range.offset, (uint64_t)existing_end);
    }
    range_offset = iree_min(range_offset, existing_range.offset);
    range_end = iree_max(range_end, existing_end);
    id4_pipeline_region_remove_free_range(builder, i);
  }

  IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_free_ranges(
      builder, builder->free_range_count + 1));
  const iree_device_size_t coalesced_length = range_end - range_offset;
  builder->free_ranges[builder->free_range_count++] =
      (id4_pipeline_region_free_range_t){
          // Byte offset into the local slab.
          .offset = range_offset,
          // Byte length of the reusable range.
          .length = coalesced_length,
      };
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_note_local_live_acquire(
    id4_pipeline_region_builder_t* builder, iree_device_size_t byte_length) {
  iree_device_size_t live_byte_length = 0;
  if (!iree_device_size_checked_add(builder->local_live_byte_length,
                                    byte_length, &live_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local live byte length overflow");
  }
  builder->local_live_byte_length = live_byte_length;
  builder->statistics.local_slab_high_water_mark = iree_max(
      builder->statistics.local_slab_high_water_mark, live_byte_length);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_note_local_live_release(
    id4_pipeline_region_builder_t* builder, iree_device_size_t byte_length,
    iree_string_view_t tensor_name) {
  if (byte_length > builder->local_live_byte_length) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "tensor %.*s release underflows local live byte length",
        (int)tensor_name.size, tensor_name.data);
  }
  builder->local_live_byte_length -= byte_length;
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

  iree_host_size_t selected_range_index = IREE_HOST_SIZE_MAX;
  iree_device_size_t selected_offset = IREE_DEVICE_SIZE_MAX;
  iree_host_size_t tail_range_index = IREE_HOST_SIZE_MAX;
  iree_device_size_t tail_offset = IREE_DEVICE_SIZE_MAX;
  for (iree_host_size_t i = 0; i < builder->free_range_count; ++i) {
    const id4_pipeline_region_free_range_t range = builder->free_ranges[i];
    iree_device_size_t aligned_offset = 0;
    if (!iree_device_size_checked_align(range.offset, alignment,
                                        &aligned_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "free range alignment overflow");
    }
    iree_device_size_t range_end = 0;
    IREE_RETURN_IF_ERROR(id4_pipeline_region_free_range_end(range, &range_end));
    iree_device_size_t allocation_end = 0;
    if (!iree_device_size_checked_add(aligned_offset, byte_length,
                                      &allocation_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "free range allocation overflow");
    }
    if (allocation_end <= range_end && aligned_offset < selected_offset) {
      selected_range_index = i;
      selected_offset = aligned_offset;
    } else if (range_end == builder->local_next_offset &&
               aligned_offset < tail_offset) {
      tail_range_index = i;
      tail_offset = aligned_offset;
    }
  }
  bool extends_tail = false;
  if (selected_range_index == IREE_HOST_SIZE_MAX) {
    selected_range_index = tail_range_index;
    selected_offset = tail_offset;
    extends_tail = selected_range_index != IREE_HOST_SIZE_MAX;
  }
  if (selected_range_index == IREE_HOST_SIZE_MAX) return iree_ok_status();

  const id4_pipeline_region_free_range_t range =
      builder->free_ranges[selected_range_index];
  iree_device_size_t range_end = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_free_range_end(range, &range_end));
  iree_device_size_t allocation_end = 0;
  if (!iree_device_size_checked_add(selected_offset, byte_length,
                                    &allocation_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "free range allocation overflow");
  }
  const iree_device_size_t prefix_length = selected_offset - range.offset;
  const iree_device_size_t suffix_length =
      extends_tail ? 0 : range_end - allocation_end;
  if (prefix_length && suffix_length != 0) {
    builder->free_ranges[selected_range_index].length = prefix_length;
    IREE_RETURN_IF_ERROR(id4_pipeline_region_append_free_range(
        builder, allocation_end, suffix_length));
  } else if (prefix_length) {
    builder->free_ranges[selected_range_index].length = prefix_length;
  } else if (suffix_length) {
    builder->free_ranges[selected_range_index].offset = allocation_end;
    builder->free_ranges[selected_range_index].length = suffix_length;
  } else {
    id4_pipeline_region_remove_free_range(builder, selected_range_index);
  }
  if (extends_tail) {
    builder->local_next_offset = allocation_end;
    builder->statistics.local_slab_byte_length =
        iree_max(builder->statistics.local_slab_byte_length, allocation_end);
  }
  *out_offset = selected_offset;
  *out_found = true;
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
  record->dtype = layout->dtype;
  record->storage_root_ordinal = record->tensor.ordinal;
  record->active_view_count =
      storage_class == ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL ? 1 : 0;
  record->alignment = layout->alignment;
  record->acquire_operation_ordinal = builder->statistics.operation_count;
  record->acquire_epoch = builder->statistics.current_epoch;
  record->release_operation_ordinal = IREE_HOST_SIZE_MAX;
  record->release_epoch = UINT32_MAX;
  record->storage_release_operation_ordinal = IREE_HOST_SIZE_MAX;
  record->storage_release_epoch = UINT32_MAX;
  record->access_epoch = UINT32_MAX;
  record->initialized = initialized;
  ++builder->tensor_record_count;
  *out_record = record;
  return iree_ok_status();
}

static void id4_pipeline_region_remove_last_tensor_record(
    id4_pipeline_region_builder_t* builder) {
  if (builder->tensor_record_count == 0) return;
  --builder->tensor_record_count;
  memset(&builder->tensor_records[builder->tensor_record_count], 0,
         sizeof(builder->tensor_records[builder->tensor_record_count]));
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

static bool id4_pipeline_region_write_ranges_cover_tensor(
    id4_pipeline_tensor_t tensor,
    const id4_pipeline_region_tensor_byte_range_t* ranges,
    iree_host_size_t range_count) {
  iree_device_size_t covered_end = 0;
  bool changed = true;
  while (changed && covered_end < tensor.length) {
    changed = false;
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      const id4_pipeline_region_tensor_byte_range_t range = ranges[i];
      if (range.offset > covered_end) continue;
      iree_device_size_t range_end = 0;
      if (!iree_device_size_checked_add(range.offset, range.length,
                                        &range_end)) {
        continue;
      }
      if (range_end <= covered_end) continue;
      covered_end = range_end;
      changed = true;
    }
  }
  return covered_end >= tensor.length;
}

static bool id4_pipeline_region_write_ranges_cover_range(
    id4_pipeline_region_tensor_byte_range_t required_range,
    const id4_pipeline_region_tensor_byte_range_t* ranges,
    iree_host_size_t range_count) {
  iree_device_size_t required_end = 0;
  if (!iree_device_size_checked_add(required_range.offset,
                                    required_range.length, &required_end)) {
    return false;
  }
  iree_device_size_t covered_end = required_range.offset;
  bool changed = true;
  while (changed && covered_end < required_end) {
    changed = false;
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      const id4_pipeline_region_tensor_byte_range_t range = ranges[i];
      if (range.offset > covered_end) continue;
      iree_device_size_t range_end = 0;
      if (!iree_device_size_checked_add(range.offset, range.length,
                                        &range_end)) {
        continue;
      }
      if (range_end <= covered_end) continue;
      covered_end = range_end;
      changed = true;
    }
  }
  return covered_end >= required_end;
}

static iree_status_t id4_pipeline_region_note_initialized_write_range(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_tensor_record_t* record,
    id4_pipeline_region_tensor_byte_range_t write_range) {
  if (record->initialized) return iree_ok_status();
  IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_initialized_write_ranges(
      builder, record, record->initialized_write_range_count + 1));
  record->initialized_write_ranges[record->initialized_write_range_count++] =
      write_range;
  if (id4_pipeline_region_write_ranges_cover_tensor(
          record->tensor, record->initialized_write_ranges,
          record->initialized_write_range_count)) {
    record->initialized = true;
  }
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
        record->storage_root_ordinal != record->tensor.ordinal ||
        record->range_published ||
        record->storage_release_epoch == UINT32_MAX ||
        record->storage_release_epoch >= builder->statistics.current_epoch) {
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

iree_status_t id4_pipeline_region_builder_set_recording_command_buffer(
    id4_pipeline_region_builder_t* builder,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(builder);
  if (builder->mode == ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN) {
    if (command_buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dry-run region builders cannot select a HAL command buffer");
    }
    return iree_ok_status();
  }
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record region builders require a HAL command buffer");
  }
  if (builder->statistics.operation_count != 0 &&
      builder->command_buffer != command_buffer) {
    for (iree_host_size_t i = 0; i < builder->tensor_record_count; ++i) {
      const id4_pipeline_region_tensor_record_t* record =
          &builder->tensor_records[i];
      if (record->access_epoch == builder->statistics.current_epoch &&
          record->epoch_access != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "region command buffer switch requires an authored barrier after "
            "tensor %.*s access",
            (int)record->name.size, record->name.data);
      }
    }
  }
  builder->command_buffer = command_buffer;
  return iree_ok_status();
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

iree_host_size_t id4_pipeline_region_builder_kernel_count(
    const id4_pipeline_region_builder_t* builder) {
  return builder ? builder->kernel_plan_count : 0;
}

const id4_pipeline_region_kernel_plan_t* id4_pipeline_region_builder_kernel_at(
    const id4_pipeline_region_builder_t* builder, iree_host_size_t index) {
  if (!builder || index >= builder->kernel_plan_count) return NULL;
  return &builder->kernel_plans[index];
}

iree_host_size_t id4_pipeline_region_builder_local_lifetime_count(
    const id4_pipeline_region_builder_t* builder) {
  if (!builder) return 0;
  iree_host_size_t lifetime_count = 0;
  for (iree_host_size_t i = 0; i < builder->tensor_record_count; ++i) {
    if (builder->tensor_records[i].tensor.storage_class ==
        ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL) {
      ++lifetime_count;
    }
  }
  return lifetime_count;
}

iree_status_t id4_pipeline_region_builder_clone_local_lifetimes(
    const id4_pipeline_region_builder_t* builder,
    iree_allocator_t host_allocator, iree_host_size_t* out_lifetime_count,
    id4_pipeline_region_local_lifetime_t** out_lifetimes) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_lifetime_count);
  IREE_ASSERT_ARGUMENT(out_lifetimes);
  *out_lifetime_count = 0;
  *out_lifetimes = NULL;

  const iree_host_size_t lifetime_count =
      id4_pipeline_region_builder_local_lifetime_count(builder);
  if (lifetime_count == 0) return iree_ok_status();

  id4_pipeline_region_local_lifetime_t* lifetimes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, lifetime_count,
                                  sizeof(lifetimes[0]), (void**)&lifetimes));
  memset(lifetimes, 0, lifetime_count * sizeof(lifetimes[0]));

  iree_status_t status = iree_ok_status();
  iree_host_size_t lifetime_index = 0;
  for (iree_host_size_t i = 0;
       i < builder->tensor_record_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_region_tensor_record_t* record =
        &builder->tensor_records[i];
    if (record->tensor.storage_class !=
        ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL) {
      continue;
    }
    id4_pipeline_region_local_lifetime_t* lifetime =
        &lifetimes[lifetime_index++];
    lifetime->ordinal = record->tensor.ordinal;
    lifetime->storage_root_ordinal = record->storage_root_ordinal;
    lifetime->flags = record->lifetime_flags;
    lifetime->dtype = record->dtype;
    lifetime->shape = record->tensor.shape;
    lifetime->byte_length = record->tensor.length;
    lifetime->alignment = record->alignment;
    lifetime->offset = record->tensor.offset;
    if (record->storage_root_ordinal >= builder->tensor_record_count) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "local lifetime storage root is invalid");
      continue;
    }
    const id4_pipeline_region_tensor_record_t* root =
        &builder->tensor_records[record->storage_root_ordinal];
    if (record->tensor.offset < root->tensor.offset) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "local lifetime precedes its storage root");
      continue;
    }
    lifetime->storage_byte_offset = record->tensor.offset - root->tensor.offset;
    lifetime->acquire_operation_ordinal = record->acquire_operation_ordinal;
    lifetime->acquire_epoch = record->acquire_epoch;
    lifetime->release_operation_ordinal = record->release_operation_ordinal;
    lifetime->release_epoch = record->release_epoch;
    lifetime->storage_release_operation_ordinal =
        root->storage_release_operation_ordinal;
    lifetime->storage_release_epoch = root->storage_release_epoch;
    status = id4_pipeline_region_string_clone(record->name, host_allocator,
                                              &lifetime->name);
  }
  if (iree_status_is_ok(status)) {
    *out_lifetime_count = lifetime_count;
    *out_lifetimes = lifetimes;
  } else {
    id4_pipeline_region_local_lifetime_list_release(lifetime_count, lifetimes,
                                                    host_allocator);
  }
  return status;
}

void id4_pipeline_region_local_lifetime_list_release(
    iree_host_size_t lifetime_count,
    id4_pipeline_region_local_lifetime_t* lifetimes,
    iree_allocator_t host_allocator) {
  if (!lifetimes) return;
  for (iree_host_size_t i = 0; i < lifetime_count; ++i) {
    id4_pipeline_region_string_release(lifetimes[i].name, host_allocator);
  }
  iree_allocator_free(host_allocator, lifetimes);
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
  if (found_reusable_range) {
    record->lifetime_flags |= ID4_PIPELINE_REGION_LOCAL_LIFETIME_FLAG_REUSED;
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_note_local_live_acquire(
      builder, layout->byte_length));
  ++builder->statistics.local_acquire_count;
  *out_tensor = record->tensor;
  return iree_ok_status();
}

iree_status_t id4_pipeline_region_subview_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t source,
    const id4_pipeline_tensor_layout_t* layout,
    iree_device_size_t source_byte_offset,
    id4_pipeline_region_subview_flags_t flags,
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
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_validate_subview_flags(flags, layout->name));

  id4_pipeline_region_tensor_record_t* source_record = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_lookup_tensor_record(
      builder, source, &source_record));
  if (source_record->tensor.storage_class !=
      ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "subview source %.*s is not a local tensor",
                            (int)source_record->name.size,
                            source_record->name.data);
  }
  if (source_record->released || source_record->consumed) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "subview source %.*s is no longer live",
                            (int)source_record->name.size,
                            source_record->name.data);
  }
  if (source_record->dtype != layout->dtype) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "subview tensor %.*s changes source dtype",
                            (int)layout->name.size, layout->name.data);
  }
  const uint32_t storage_root_ordinal = source_record->storage_root_ordinal;
  if (storage_root_ordinal >= builder->tensor_record_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "subview source storage root is invalid");
  }
  id4_pipeline_region_tensor_record_t* root_record =
      &builder->tensor_records[storage_root_ordinal];
  if (root_record->storage_root_ordinal != storage_root_ordinal ||
      root_record->tensor.storage_class !=
          ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL ||
      root_record->active_view_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "subview source storage is not live");
  }
  if (root_record->active_view_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "subview source view count overflow");
  }
  for (iree_host_size_t i = 0; i < builder->tensor_record_count; ++i) {
    const id4_pipeline_region_tensor_record_t* record =
        &builder->tensor_records[i];
    if (record->storage_root_ordinal == storage_root_ordinal &&
        record->access_epoch == builder->statistics.current_epoch) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "subview tensor %.*s requires a barrier after storage access",
          (int)layout->name.size, layout->name.data);
    }
  }

  iree_device_size_t source_end = 0;
  if (!iree_device_size_checked_add(source_byte_offset, layout->byte_length,
                                    &source_end) ||
      source_end > source_record->tensor.length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "subview tensor %.*s range exceeds source tensor %.*s byte length",
        (int)layout->name.size, layout->name.data,
        (int)source_record->name.size, source_record->name.data);
  }
  iree_device_size_t offset = 0;
  if (!iree_device_size_checked_add(source_record->tensor.offset,
                                    source_byte_offset, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "subview tensor storage offset overflow");
  }
  const iree_device_size_t alignment =
      layout->alignment == 0 ? 1 : layout->alignment;
  if ((offset % alignment) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "subview tensor %.*s offset is not aligned",
                            (int)layout->name.size, layout->name.data);
  }

  const uint32_t source_ordinal = source_record->tensor.ordinal;
  const bool discard_contents = iree_any_bit_set(
      flags, ID4_PIPELINE_REGION_SUBVIEW_FLAG_DISCARD_CONTENTS);
  const bool initialized = source_record->initialized && !discard_contents;
  id4_pipeline_region_tensor_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_make_tensor_record(
      builder, ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL, layout,
      builder->local_binding_slot, offset, initialized, &record));
  record->storage_root_ordinal = storage_root_ordinal;
  record->active_view_count = 0;
  record->lifetime_flags |= ID4_PIPELINE_REGION_LOCAL_LIFETIME_FLAG_SUBVIEW;

  iree_status_t status = iree_ok_status();
  source_record = &builder->tensor_records[source_ordinal];
  if (!discard_contents && !source_record->initialized) {
    const iree_device_size_t subview_end =
        source_byte_offset + layout->byte_length;
    for (iree_host_size_t i = 0;
         i < source_record->initialized_write_range_count &&
         iree_status_is_ok(status);
         ++i) {
      const id4_pipeline_region_tensor_byte_range_t source_range =
          source_record->initialized_write_ranges[i];
      const iree_device_size_t source_range_end =
          source_range.offset + source_range.length;
      const iree_device_size_t intersection_begin =
          iree_max(source_byte_offset, source_range.offset);
      const iree_device_size_t intersection_end =
          iree_min(subview_end, source_range_end);
      if (intersection_begin >= intersection_end) continue;
      const id4_pipeline_region_tensor_byte_range_t target_range = {
          // Byte offset relative to the new logical tensor.
          .offset = intersection_begin - source_byte_offset,
          // Byte length initialized by the source range.
          .length = intersection_end - intersection_begin,
      };
      status = id4_pipeline_region_note_initialized_write_range(builder, record,
                                                                target_range);
    }
  }
  if (!iree_status_is_ok(status)) {
    id4_pipeline_region_remove_last_tensor_record(builder);
    return status;
  }

  root_record = &builder->tensor_records[storage_root_ordinal];
  ++root_record->active_view_count;
  ++builder->statistics.local_subview_count;
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
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_validate_initialized_import_ranges(import));
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
  for (iree_host_size_t i = 0; i < import->initialized_range_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_region_note_initialized_write_range(
        builder, record, import->initialized_ranges[i]));
  }
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
  if (record->storage_root_ordinal >= builder->tensor_record_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "tensor storage root is invalid");
  }
  id4_pipeline_region_tensor_record_t* root =
      &builder->tensor_records[record->storage_root_ordinal];
  if (root->active_view_count == 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "tensor storage reference count underflow");
  }
  record->released = true;
  record->release_operation_ordinal = builder->statistics.operation_count;
  record->release_epoch = builder->statistics.current_epoch;
  --root->active_view_count;
  if (root->active_view_count == 0) {
    root->storage_release_operation_ordinal =
        builder->statistics.operation_count;
    root->storage_release_epoch = builder->statistics.current_epoch;
    IREE_RETURN_IF_ERROR(id4_pipeline_region_note_local_live_release(
        builder, root->tensor.length, root->name));
  }
  ++builder->statistics.local_release_count;
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_region_validate_destructive_alias_write_binding(
    const id4_pipeline_region_dispatch_binding_t* binding,
    const id4_pipeline_region_tensor_record_t* record,
    iree_string_view_t kernel_name, iree_host_size_t binding_index) {
  if (!iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_DESTRUCTIVE_ALIAS_WRITE)) {
    return iree_ok_status();
  }
  if (!iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel %.*s binding %" PRIhsz
                            " destructive alias binding requires write access",
                            (int)kernel_name.size, kernel_name.data,
                            binding_index);
  }
  if (iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_WRITE_RANGE) &&
      (binding->write_range.offset != 0 ||
       binding->write_range.length != record->tensor.length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz
        " destructive alias write must cover the full logical tensor",
        (int)kernel_name.size, kernel_name.data, binding_index);
  }
  if (record->tensor.storage_class != ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL ||
      record->storage_root_ordinal == record->tensor.ordinal) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel %.*s binding %" PRIhsz
        " destructive alias write requires a local transient subview",
        (int)kernel_name.size, kernel_name.data, binding_index);
  }
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
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_dispatch_binding_flags(
      binding->flags, kernel->name, binding_index));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_lookup_tensor_record(
      builder, binding->tensor, out_record));
  id4_pipeline_region_tensor_record_t* record = *out_record;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_binding_read_range(
      binding, kernel->name, binding_index));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_binding_write_range(
      binding, kernel->name, binding_index));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_validate_destructive_alias_write_binding(
          binding, record, kernel->name, binding_index));
  if (record->released) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel %.*s binding %" PRIhsz " uses released tensor %.*s",
        (int)kernel->name.size, kernel->name.data, binding_index,
        (int)record->name.size, record->name.data);
  }
  if (record->consumed) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel %.*s binding %" PRIhsz " uses consumed tensor %.*s",
        (int)kernel->name.size, kernel->name.data, binding_index,
        (int)record->name.size, record->name.data);
  }
  if (iree_all_bits_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_READ)) {
    const id4_pipeline_region_tensor_byte_range_t read_range =
        id4_pipeline_region_binding_read_range(binding);
    if (!record->initialized &&
        !id4_pipeline_region_write_ranges_cover_range(
            read_range, record->initialized_write_ranges,
            record->initialized_write_range_count)) {
      iree_device_size_t read_end = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_region_tensor_byte_range_end(read_range, &read_end));
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel %.*s binding %" PRIhsz
          " reads uninitialized tensor %.*s range [%" PRIu64 ", %" PRIu64 ")",
          (int)kernel->name.size, kernel->name.data, binding_index,
          (int)record->name.size, record->name.data,
          (uint64_t)read_range.offset, (uint64_t)read_end);
    }
  }
  const bool same_epoch =
      record->access_epoch == builder->statistics.current_epoch;
  if (!same_epoch) return iree_ok_status();
  if (iree_any_bit_set(record->epoch_access,
                       ID4_PIPELINE_TENSOR_ACCESS_WRITE) &&
      iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_READ)) {
    const id4_pipeline_region_tensor_byte_range_t read_range =
        id4_pipeline_region_binding_read_range(binding);
    for (iree_host_size_t i = 0; i < record->epoch_write_range_count; ++i) {
      if (!id4_pipeline_region_tensor_byte_ranges_overlap(
              read_range, record->epoch_write_ranges[i])) {
        continue;
      }
      iree_device_size_t read_end = 0;
      iree_device_size_t existing_end = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_region_tensor_byte_range_end(read_range, &read_end));
      IREE_RETURN_IF_ERROR(id4_pipeline_region_tensor_byte_range_end(
          record->epoch_write_ranges[i], &existing_end));
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel %.*s binding %" PRIhsz " reads tensor %.*s range [%" PRIu64
          ", %" PRIu64 ") overlapping same-epoch write range [%" PRIu64
          ", %" PRIu64 ")",
          (int)kernel->name.size, kernel->name.data, binding_index,
          (int)record->name.size, record->name.data,
          (uint64_t)read_range.offset, (uint64_t)read_end,
          (uint64_t)record->epoch_write_ranges[i].offset,
          (uint64_t)existing_end);
    }
  }
  if (iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE) &&
      iree_any_bit_set(record->epoch_access, ID4_PIPELINE_TENSOR_ACCESS_READ)) {
    const id4_pipeline_region_tensor_byte_range_t write_range =
        id4_pipeline_region_binding_write_range(binding);
    for (iree_host_size_t i = 0; i < record->epoch_read_range_count; ++i) {
      if (!id4_pipeline_region_tensor_byte_ranges_overlap(
              write_range, record->epoch_read_ranges[i])) {
        continue;
      }
      iree_device_size_t write_end = 0;
      iree_device_size_t existing_end = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_region_tensor_byte_range_end(write_range, &write_end));
      IREE_RETURN_IF_ERROR(id4_pipeline_region_tensor_byte_range_end(
          record->epoch_read_ranges[i], &existing_end));
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel %.*s binding %" PRIhsz " writes tensor %.*s range [%" PRIu64
          ", %" PRIu64 ") overlapping same-epoch read range [%" PRIu64
          ", %" PRIu64 ")",
          (int)kernel->name.size, kernel->name.data, binding_index,
          (int)record->name.size, record->name.data,
          (uint64_t)write_range.offset, (uint64_t)write_end,
          (uint64_t)record->epoch_read_ranges[i].offset,
          (uint64_t)existing_end);
    }
  }
  if (iree_any_bit_set(binding->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
    const id4_pipeline_region_tensor_byte_range_t write_range =
        id4_pipeline_region_binding_write_range(binding);
    for (iree_host_size_t i = 0; i < record->epoch_write_range_count; ++i) {
      if (!id4_pipeline_region_tensor_byte_ranges_overlap(
              write_range, record->epoch_write_ranges[i])) {
        continue;
      }
      iree_device_size_t write_end = 0;
      iree_device_size_t existing_end = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_region_tensor_byte_range_end(write_range, &write_end));
      IREE_RETURN_IF_ERROR(id4_pipeline_region_tensor_byte_range_end(
          record->epoch_write_ranges[i], &existing_end));
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel %.*s binding %" PRIhsz " writes tensor %.*s range [%" PRIu64
          ", %" PRIu64 ") overlapping same-epoch write range [%" PRIu64
          ", %" PRIu64 ")",
          (int)kernel->name.size, kernel->name.data, binding_index,
          (int)record->name.size, record->name.data,
          (uint64_t)write_range.offset, (uint64_t)write_end,
          (uint64_t)record->epoch_write_ranges[i].offset,
          (uint64_t)existing_end);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_apply_dispatch_binding_access(
    id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_tensor_record_t* record,
    const id4_pipeline_region_dispatch_binding_t* binding) {
  if (record->access_epoch != builder->statistics.current_epoch) {
    record->access_epoch = builder->statistics.current_epoch;
    record->epoch_access = 0;
    record->epoch_write_range_count = 0;
    record->epoch_read_range_count = 0;
  }
  const id4_pipeline_tensor_access_flags_t access = binding->access;
  record->epoch_access |= access;
  if (iree_any_bit_set(access, ID4_PIPELINE_TENSOR_ACCESS_READ)) {
    const id4_pipeline_region_tensor_byte_range_t read_range =
        id4_pipeline_region_binding_read_range(binding);
    IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_epoch_read_ranges(
        builder, record, record->epoch_read_range_count + 1));
    record->epoch_read_ranges[record->epoch_read_range_count++] = read_range;
  }
  if (iree_any_bit_set(access, ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
    const id4_pipeline_region_tensor_byte_range_t write_range =
        id4_pipeline_region_binding_write_range(binding);
    IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_epoch_write_ranges(
        builder, record, record->epoch_write_range_count + 1));
    record->epoch_write_ranges[record->epoch_write_range_count++] = write_range;
    IREE_RETURN_IF_ERROR(id4_pipeline_region_note_initialized_write_range(
        builder, record, write_range));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_validate_dispatch_binding_aliases(
    const id4_pipeline_region_kernel_t* kernel, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    id4_pipeline_region_tensor_record_t* const* records) {
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const id4_pipeline_region_dispatch_binding_t* lhs = &bindings[i];
    for (iree_host_size_t j = i + 1; j < binding_count; ++j) {
      const id4_pipeline_region_dispatch_binding_t* rhs = &bindings[j];
      if (lhs->tensor.binding_slot != rhs->tensor.binding_slot) continue;

      if (iree_any_bit_set(lhs->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE) &&
          iree_any_bit_set(rhs->access, ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
        id4_pipeline_region_tensor_byte_range_t lhs_storage_range;
        id4_pipeline_region_tensor_byte_range_t rhs_storage_range;
        IREE_RETURN_IF_ERROR(id4_pipeline_region_binding_storage_write_range(
            lhs, &lhs_storage_range));
        IREE_RETURN_IF_ERROR(id4_pipeline_region_binding_storage_write_range(
            rhs, &rhs_storage_range));
        if (id4_pipeline_region_tensor_byte_ranges_overlap(lhs_storage_range,
                                                           rhs_storage_range)) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "kernel %.*s bindings %" PRIhsz
                                  " and %" PRIhsz " write overlapping storage",
                                  (int)kernel->name.size, kernel->name.data, i,
                                  j);
        }
      }

      const id4_pipeline_region_dispatch_binding_t* writers[2] = {lhs, rhs};
      const id4_pipeline_region_tensor_record_t* writer_records[2] = {
          records[i], records[j]};
      const iree_host_size_t writer_indices[2] = {i, j};
      const id4_pipeline_region_dispatch_binding_t* readers[2] = {rhs, lhs};
      const id4_pipeline_region_tensor_record_t* reader_records[2] = {
          records[j], records[i]};
      const iree_host_size_t reader_indices[2] = {j, i};
      for (iree_host_size_t direction = 0; direction < 2; ++direction) {
        const id4_pipeline_region_dispatch_binding_t* writer =
            writers[direction];
        const id4_pipeline_region_dispatch_binding_t* reader =
            readers[direction];
        if (!iree_any_bit_set(writer->access,
                              ID4_PIPELINE_TENSOR_ACCESS_WRITE) ||
            !iree_any_bit_set(reader->access,
                              ID4_PIPELINE_TENSOR_ACCESS_READ)) {
          continue;
        }
        id4_pipeline_region_tensor_byte_range_t write_storage_range;
        id4_pipeline_region_tensor_byte_range_t read_storage_range;
        IREE_RETURN_IF_ERROR(id4_pipeline_region_binding_storage_write_range(
            writer, &write_storage_range));
        IREE_RETURN_IF_ERROR(id4_pipeline_region_binding_storage_read_range(
            reader, &read_storage_range));
        if (!id4_pipeline_region_tensor_byte_ranges_overlap(
                write_storage_range, read_storage_range)) {
          continue;
        }
        if (!iree_all_bits_set(
                writer->flags,
                ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_DESTRUCTIVE_ALIAS_WRITE) ||
            writer_records[direction]->storage_root_ordinal !=
                reader_records[direction]->storage_root_ordinal) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "kernel %.*s write binding %" PRIhsz
                                  " overlaps read binding %" PRIhsz
                                  " without destructive alias permission",
                                  (int)kernel->name.size, kernel->name.data,
                                  writer_indices[direction],
                                  reader_indices[direction]);
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_collect_dispatch_records(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_kernel_t* kernel, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    id4_pipeline_region_tensor_record_t*** out_records) {
  *out_records = NULL;
  if (binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &builder->arena, binding_count, sizeof((*out_records)[0]),
        (void**)out_records));
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_dispatch_binding(
        builder, kernel, &bindings[i], i, &(*out_records)[i]));
  }
  return id4_pipeline_region_validate_dispatch_binding_aliases(
      kernel, binding_count, bindings, *out_records);
}

static iree_status_t id4_pipeline_region_record_dispatch(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_kernel_t* kernel,
    iree_hal_dispatch_config_t dispatch_config,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    iree_hal_dispatch_flags_t flags) {
  if (builder->mode != ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    return iree_ok_status();
  }

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
  return iree_hal_command_buffer_dispatch(
      builder->command_buffer, kernel->executable, kernel->function,
      dispatch_config, constants, hal_binding_list, flags);
}

static iree_status_t id4_pipeline_region_commit_dispatch(
    id4_pipeline_region_builder_t* builder, iree_host_size_t binding_count,
    id4_pipeline_region_tensor_record_t** records,
    const id4_pipeline_region_dispatch_binding_t* bindings) {
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_region_apply_dispatch_binding_access(
        builder, records[i], &bindings[i]));
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (!iree_all_bits_set(
            bindings[i].flags,
            ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_DESTRUCTIVE_ALIAS_WRITE)) {
      continue;
    }
    id4_pipeline_region_tensor_byte_range_t write_storage_range;
    IREE_RETURN_IF_ERROR(id4_pipeline_region_binding_storage_write_range(
        &bindings[i], &write_storage_range));
    const uint32_t storage_root_ordinal = records[i]->storage_root_ordinal;
    for (iree_host_size_t j = 0; j < builder->tensor_record_count; ++j) {
      id4_pipeline_region_tensor_record_t* record = &builder->tensor_records[j];
      if (j == records[i]->tensor.ordinal || record->released ||
          record->storage_root_ordinal != storage_root_ordinal) {
        continue;
      }
      const id4_pipeline_region_tensor_byte_range_t tensor_storage_range = {
          // Absolute byte offset into the local slab binding.
          .offset = record->tensor.offset,
          // Full logical tensor byte length.
          .length = record->tensor.length,
      };
      if (!id4_pipeline_region_tensor_byte_ranges_overlap(
              write_storage_range, tensor_storage_range)) {
        continue;
      }
      record->initialized = false;
      record->initialized_write_range_count = 0;
      record->consumed = true;
    }
  }
  ++builder->statistics.operation_count;
  ++builder->statistics.dispatch_count;
  return iree_ok_status();
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

static iree_status_t id4_pipeline_region_validate_loom_kernel(
    const id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_loom_kernel_t* kernel,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings) {
  if (!kernel) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region Loom dispatch kernel is required");
  }
  if (iree_string_view_is_empty(kernel->specialization_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region Loom dispatch specialization key is "
                            "required");
  }
  if (iree_string_view_is_empty(kernel->module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region Loom dispatch module path is required");
  }
  if (iree_string_view_is_empty(kernel->function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region Loom dispatch function name is required");
  }
  if (kernel->config_binding_count != 0 && !kernel->config_bindings) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "region Loom dispatch config bindings are required when count is "
        "nonzero");
  }
  for (iree_host_size_t i = 0; i < kernel->config_binding_count; ++i) {
    if (iree_string_view_is_empty(kernel->config_bindings[i].key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "region Loom dispatch config binding %" PRIhsz " key is required", i);
    }
    if (iree_string_view_is_empty(kernel->config_bindings[i].value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "region Loom dispatch config binding %" PRIhsz
                              " value is required",
                              i);
    }
  }
  id4_pipeline_region_kernel_t hal_kernel;
  memset(&hal_kernel, 0, sizeof(hal_kernel));
  hal_kernel.name = kernel->function_name;
  hal_kernel.executable = kernel->executable;
  hal_kernel.function = kernel->function;
  hal_kernel.binding_count = kernel->binding_count;
  hal_kernel.constant_byte_length = kernel->constant_byte_length;
  return id4_pipeline_region_validate_kernel(builder, &hal_kernel, constants,
                                             binding_count, bindings);
}

static iree_status_t id4_pipeline_region_append_loom_kernel_plan(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_loom_kernel_t* kernel) {
  IREE_RETURN_IF_ERROR(id4_pipeline_region_reserve_kernel_plans(
      builder, builder->kernel_plan_count + 1));
  id4_pipeline_region_kernel_plan_t* target =
      &builder->kernel_plans[builder->kernel_plan_count];
  memset(target, 0, sizeof(*target));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_copy_string(
      builder, kernel->specialization_key, &target->specialization_key));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_copy_string(
      builder, kernel->module_path, &target->module_path));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_copy_string(
      builder, kernel->function_name, &target->function_name));
  target->config_binding_count = kernel->config_binding_count;
  if (kernel->config_binding_count != 0) {
    id4_pipeline_kernel_config_binding_t* config_bindings = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &builder->arena, kernel->config_binding_count,
        sizeof(config_bindings[0]), (void**)&config_bindings));
    memset(config_bindings, 0,
           kernel->config_binding_count * sizeof(config_bindings[0]));
    target->config_bindings = config_bindings;
    for (iree_host_size_t i = 0; i < kernel->config_binding_count; ++i) {
      IREE_RETURN_IF_ERROR(id4_pipeline_region_copy_string(
          builder, kernel->config_bindings[i].key, &config_bindings[i].key));
      IREE_RETURN_IF_ERROR(id4_pipeline_region_copy_string(
          builder, kernel->config_bindings[i].value,
          &config_bindings[i].value));
    }
  }
  ++builder->kernel_plan_count;
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
  IREE_RETURN_IF_ERROR(id4_pipeline_region_collect_dispatch_records(
      builder, kernel, binding_count, bindings, &records));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_record_dispatch(
      builder, kernel, dispatch_config, constants, binding_count, bindings,
      flags));
  return id4_pipeline_region_commit_dispatch(builder, binding_count, records,
                                             bindings);
}

static iree_status_t id4_pipeline_region_validate_copy_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t source,
    id4_pipeline_tensor_t target,
    id4_pipeline_region_tensor_record_t** out_source_record,
    id4_pipeline_region_tensor_record_t** out_target_record) {
  IREE_RETURN_IF_ERROR(id4_pipeline_region_lookup_tensor_record(
      builder, source, out_source_record));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_lookup_tensor_record(
      builder, target, out_target_record));
  id4_pipeline_region_tensor_record_t* source_record = *out_source_record;
  id4_pipeline_region_tensor_record_t* target_record = *out_target_record;
  if (source_record->released) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION, "copy reads released tensor %.*s",
        (int)source_record->name.size, source_record->name.data);
  }
  if (target_record->released) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION, "copy writes released tensor %.*s",
        (int)target_record->name.size, target_record->name.data);
  }
  if (!source_record->initialized) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION, "copy reads uninitialized tensor %.*s",
        (int)source_record->name.size, source_record->name.data);
  }
  if (source.length != target.length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "copy length mismatch between %.*s and %.*s",
        (int)source_record->name.size, source_record->name.data,
        (int)target_record->name.size, target_record->name.data);
  }
  if (source_record->access_epoch == builder->statistics.current_epoch &&
      iree_any_bit_set(source_record->epoch_access,
                       ID4_PIPELINE_TENSOR_ACCESS_WRITE)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "copy reads tensor %.*s written in the same epoch",
                            (int)source_record->name.size,
                            source_record->name.data);
  }
  if (target_record->access_epoch == builder->statistics.current_epoch &&
      target_record->epoch_access != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "copy writes tensor %.*s already used in the same "
                            "epoch",
                            (int)target_record->name.size,
                            target_record->name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_region_record_copy_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t source,
    id4_pipeline_tensor_t target, iree_hal_copy_flags_t flags) {
  if (builder->mode != ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    return iree_ok_status();
  }
  return iree_hal_command_buffer_copy_buffer(
      builder->command_buffer,
      iree_hal_make_indirect_buffer_ref(source.binding_slot, source.offset,
                                        source.length),
      iree_hal_make_indirect_buffer_ref(target.binding_slot, target.offset,
                                        target.length),
      flags);
}

iree_status_t id4_pipeline_region_copy_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t source,
    id4_pipeline_tensor_t target, iree_hal_copy_flags_t flags) {
  IREE_ASSERT_ARGUMENT(builder);
  id4_pipeline_region_tensor_record_t* source_record = NULL;
  id4_pipeline_region_tensor_record_t* target_record = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_copy_tensor(
      builder, source, target, &source_record, &target_record));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_record_copy_tensor(builder, source, target, flags));
  const id4_pipeline_region_dispatch_binding_t source_binding = {
      // Tensor bound to the copy source.
      .tensor = source,
      // Copy reads the full source tensor.
      .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
  };
  const id4_pipeline_region_dispatch_binding_t target_binding = {
      // Tensor bound to the copy target.
      .tensor = target,
      // Copy writes the full target tensor.
      .access = ID4_PIPELINE_TENSOR_ACCESS_WRITE,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_apply_dispatch_binding_access(
      builder, source_record, &source_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_apply_dispatch_binding_access(
      builder, target_record, &target_binding));
  ++builder->statistics.operation_count;
  ++builder->statistics.copy_count;
  return iree_ok_status();
}

iree_status_t id4_pipeline_region_dispatch_loom(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_loom_kernel_t* kernel,
    iree_hal_dispatch_config_t dispatch_config,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_loom_kernel(
      builder, kernel, constants, binding_count, bindings));

  id4_pipeline_region_kernel_t hal_kernel;
  memset(&hal_kernel, 0, sizeof(hal_kernel));
  hal_kernel.name = kernel->function_name;
  hal_kernel.executable = kernel->executable;
  hal_kernel.function = kernel->function;
  hal_kernel.binding_count = kernel->binding_count;
  hal_kernel.constant_byte_length = kernel->constant_byte_length;

  id4_pipeline_region_tensor_record_t** records = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_collect_dispatch_records(
      builder, &hal_kernel, binding_count, bindings, &records));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_append_loom_kernel_plan(builder, kernel));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_record_dispatch(
      builder, &hal_kernel, dispatch_config, constants, binding_count, bindings,
      flags));
  return id4_pipeline_region_commit_dispatch(builder, binding_count, records,
                                             bindings);
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

static iree_status_t id4_pipeline_prepared_region_validate_create_options(
    const id4_pipeline_region_builder_t* builder,
    const id4_pipeline_prepared_region_create_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region builder is required");
  }
  if (builder->mode != ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared regions require a record-mode builder");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("prepared region")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "prepared region extension structures are not "
                            "supported");
  }
  if (!options->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region device group is required");
  }
  iree_host_size_t device_count =
      iree_hal_device_group_device_count(options->device_group);
  if (options->device_index >= device_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region device index %" PRIhsz
                            " out of range for device group with %" PRIhsz
                            " devices",
                            options->device_index, device_count);
  }
  if (!iree_hal_device_group_device_at(options->device_group,
                                       options->device_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region device is required");
  }
  if (!builder->command_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region command buffer is required");
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_prepared_region_create(
    const id4_pipeline_region_builder_t* builder,
    const id4_pipeline_prepared_region_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_prepared_region_t** out_prepared_region) {
  IREE_ASSERT_ARGUMENT(out_prepared_region);
  *out_prepared_region = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_prepared_region_validate_create_options(builder, options));

  id4_pipeline_prepared_region_t* prepared_region = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*prepared_region), (void**)&prepared_region));
  memset(prepared_region, 0, sizeof(*prepared_region));
  iree_atomic_ref_count_init(&prepared_region->ref_count);
  prepared_region->host_allocator = host_allocator;
  prepared_region->device_group = options->device_group;
  iree_hal_device_group_retain(prepared_region->device_group);
  prepared_region->device = iree_hal_device_group_device_at(
      options->device_group, options->device_index);
  prepared_region->command_buffer = builder->command_buffer;
  iree_hal_command_buffer_retain(prepared_region->command_buffer);
  prepared_region->local_slab_pool = options->local_slab_pool;
  iree_hal_pool_retain(prepared_region->local_slab_pool);
  prepared_region->queue_affinity = options->queue_affinity;
  prepared_region->local_slab_params = options->local_slab_params;
  prepared_region->local_slab_alloca_flags = options->local_slab_alloca_flags;
  prepared_region->local_slab_dealloca_flags =
      options->local_slab_dealloca_flags;
  prepared_region->binding_capacity = builder->binding_capacity;
  prepared_region->local_binding_slot = builder->local_binding_slot;
  prepared_region->statistics = builder->statistics;
  *out_prepared_region = prepared_region;
  return iree_ok_status();
}

static void id4_pipeline_prepared_region_destroy(
    id4_pipeline_prepared_region_t* prepared_region) {
  iree_allocator_t host_allocator = prepared_region->host_allocator;
  iree_hal_pool_release(prepared_region->local_slab_pool);
  iree_hal_command_buffer_release(prepared_region->command_buffer);
  iree_hal_device_group_release(prepared_region->device_group);
  iree_allocator_free(host_allocator, prepared_region);
}

void id4_pipeline_prepared_region_retain(
    id4_pipeline_prepared_region_t* prepared_region) {
  if (!prepared_region) return;
  iree_atomic_ref_count_inc(&prepared_region->ref_count);
}

void id4_pipeline_prepared_region_release(
    id4_pipeline_prepared_region_t* prepared_region) {
  if (prepared_region &&
      iree_atomic_ref_count_dec(&prepared_region->ref_count) == 1) {
    id4_pipeline_prepared_region_destroy(prepared_region);
  }
}

static iree_status_t id4_pipeline_prepared_region_validate_issue_options(
    const id4_pipeline_prepared_region_t* prepared_region,
    const id4_pipeline_prepared_region_issue_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("prepared region issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "prepared region issue extension structures are "
                            "not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("prepared region wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("prepared region signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region final signal is required");
  }
  const bool local_slot_is_trailing = prepared_region->local_binding_slot + 1 ==
                                      prepared_region->binding_capacity;
  const iree_host_size_t expected_binding_count =
      prepared_region->statistics.local_slab_byte_length == 0 &&
              local_slot_is_trailing
          ? prepared_region->local_binding_slot
          : prepared_region->binding_capacity;
  if (options->binding_table.count != expected_binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared region expected %" PRIhsz " bindings but got %" PRIhsz,
        expected_binding_count, options->binding_table.count);
  }
  if (options->binding_table.count != 0 && !options->binding_table.bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared region binding table is required");
  }
  if (prepared_region->statistics.local_slab_byte_length != 0) {
    const iree_hal_buffer_binding_t local_binding =
        options->binding_table.bindings[prepared_region->local_binding_slot];
    if (local_binding.buffer || local_binding.offset != 0 ||
        local_binding.length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "prepared region local binding slot must be "
                              "empty");
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_prepared_region_copy_binding_table(
    const id4_pipeline_prepared_region_t* prepared_region,
    iree_hal_buffer_binding_table_t source_table,
    iree_hal_buffer_t* local_slab_buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_binding_table_t* out_binding_table) {
  out_binding_table->count = 0;
  out_binding_table->bindings = NULL;
  if (source_table.count == 0) return iree_ok_status();

  iree_hal_buffer_binding_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, source_table.count,
                                  sizeof(bindings[0]), (void**)&bindings));
  memcpy(bindings, source_table.bindings,
         source_table.count * sizeof(*bindings));
  if (local_slab_buffer) {
    bindings[prepared_region->local_binding_slot] = (iree_hal_buffer_binding_t){
        // Local transient slab buffer for this issue.
        .buffer = local_slab_buffer,
        // The local slab binding starts at byte zero.
        .offset = 0,
        // Full local slab length visible to recorded binding refs.
        .length = prepared_region->statistics.local_slab_byte_length,
    };
  }
  out_binding_table->count = source_table.count;
  out_binding_table->bindings = bindings;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_prepared_region_create_semaphore(
    id4_pipeline_prepared_region_t* prepared_region,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      prepared_region->device, prepared_region->queue_affinity,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_hal_semaphore_list_t id4_pipeline_region_one_semaphore_list(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return (iree_hal_semaphore_list_t){
      // Number of semaphores in the list.
      .count = 1,
      // Semaphore pointer array.
      .semaphores = semaphore,
      // Payload value pointer array.
      .payload_values = payload_value,
  };
}

static iree_status_t id4_pipeline_prepared_region_issue_without_local_slab(
    id4_pipeline_prepared_region_t* prepared_region,
    const id4_pipeline_prepared_region_issue_options_t* options) {
  return iree_hal_device_queue_execute(
      prepared_region->device, prepared_region->queue_affinity,
      options->wait_semaphore_list, options->signal_semaphore_list,
      prepared_region->command_buffer, options->binding_table,
      options->execute_flags);
}

static iree_status_t id4_pipeline_prepared_region_issue_with_local_slab(
    id4_pipeline_prepared_region_t* prepared_region,
    const id4_pipeline_prepared_region_issue_options_t* options) {
  iree_hal_semaphore_t* alloca_semaphore = NULL;
  iree_hal_semaphore_t* execute_semaphore = NULL;
  iree_hal_buffer_t* local_slab_buffer = NULL;
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();

  iree_status_t status = id4_pipeline_prepared_region_create_semaphore(
      prepared_region, &alloca_semaphore);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_prepared_region_create_semaphore(prepared_region,
                                                           &execute_semaphore);
  }

  uint64_t alloca_payload_value = 1;
  uint64_t execute_payload_value = 1;
  iree_hal_semaphore_list_t alloca_signal_list =
      id4_pipeline_region_one_semaphore_list(&alloca_semaphore,
                                             &alloca_payload_value);
  iree_hal_semaphore_list_t execute_signal_list =
      id4_pipeline_region_one_semaphore_list(&execute_semaphore,
                                             &execute_payload_value);

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        prepared_region->device, prepared_region->queue_affinity,
        options->wait_semaphore_list, alloca_signal_list,
        prepared_region->local_slab_pool, prepared_region->local_slab_params,
        prepared_region->statistics.local_slab_byte_length,
        prepared_region->local_slab_alloca_flags, &local_slab_buffer);
  }
  const bool alloca_submitted = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_prepared_region_copy_binding_table(
        prepared_region, options->binding_table, local_slab_buffer,
        prepared_region->host_allocator, &binding_table);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        prepared_region->device, prepared_region->queue_affinity,
        alloca_signal_list, execute_signal_list,
        prepared_region->command_buffer, binding_table, options->execute_flags);
  }
  const bool execute_submitted = alloca_submitted && iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        prepared_region->device, prepared_region->queue_affinity,
        execute_signal_list, options->signal_semaphore_list, local_slab_buffer,
        prepared_region->local_slab_dealloca_flags);
  } else if (alloca_submitted && local_slab_buffer) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        execute_submitted ? execute_signal_list : alloca_signal_list;
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        prepared_region->device, prepared_region->queue_affinity,
        cleanup_wait_list, iree_hal_semaphore_list_empty(), local_slab_buffer,
        prepared_region->local_slab_dealloca_flags);
    status = iree_status_join(status, cleanup_status);
  }

  iree_allocator_free(prepared_region->host_allocator,
                      (void*)binding_table.bindings);
  iree_hal_buffer_release(local_slab_buffer);
  iree_hal_semaphore_release(execute_semaphore);
  iree_hal_semaphore_release(alloca_semaphore);
  return status;
}

iree_status_t id4_pipeline_prepared_region_issue(
    id4_pipeline_prepared_region_t* prepared_region,
    const id4_pipeline_prepared_region_issue_options_t* options) {
  IREE_ASSERT_ARGUMENT(prepared_region);
  IREE_RETURN_IF_ERROR(id4_pipeline_prepared_region_validate_issue_options(
      prepared_region, options));
  if (prepared_region->statistics.local_slab_byte_length == 0) {
    return id4_pipeline_prepared_region_issue_without_local_slab(
        prepared_region, options);
  }
  return id4_pipeline_prepared_region_issue_with_local_slab(prepared_region,
                                                            options);
}

void id4_pipeline_prepared_region_statistics(
    const id4_pipeline_prepared_region_t* prepared_region,
    id4_pipeline_region_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(prepared_region);
  IREE_ASSERT_ARGUMENT(out_statistics);
  *out_statistics = prepared_region->statistics;
}
