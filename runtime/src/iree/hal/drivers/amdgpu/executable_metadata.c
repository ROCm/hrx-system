// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_metadata.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_metadata_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_executable_metadata_storage_size(
    const iree_hal_amdgpu_executable_metadata_counts_t* counts,
    iree_host_size_t* out_storage_byte_length) {
  IREE_ASSERT_ARGUMENT(counts);
  IREE_ASSERT_ARGUMENT(out_storage_byte_length);
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_executable_metadata_t), out_storage_byte_length,
      IREE_STRUCT_FIELD(counts->export_count,
                        iree_hal_amdgpu_executable_export_t, NULL),
      IREE_STRUCT_FIELD(counts->export_count,
                        iree_hal_amdgpu_executable_reflection_t, NULL),
      IREE_STRUCT_FIELD(counts->parameter_count,
                        iree_hal_executable_function_parameter_t, NULL),
      IREE_STRUCT_FIELD_ALIGNED(counts->layout_blob_byte_length, uint8_t,
                                iree_alignof(iree_hal_amdgpu_kernarg_layout_t),
                                NULL));
}

iree_status_t iree_hal_amdgpu_executable_metadata_allocate(
    const iree_hal_amdgpu_executable_metadata_counts_t* counts,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_executable_metadata_t** out_metadata) {
  IREE_ASSERT_ARGUMENT(counts);
  IREE_ASSERT_ARGUMENT(out_metadata);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_metadata = NULL;

  iree_host_size_t exports_offset = 0;
  iree_host_size_t reflection_offset = 0;
  iree_host_size_t parameters_offset = 0;
  iree_host_size_t layout_blob_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_amdgpu_executable_metadata_t), &total_size,
              IREE_STRUCT_FIELD(counts->export_count,
                                iree_hal_amdgpu_executable_export_t,
                                &exports_offset),
              IREE_STRUCT_FIELD(counts->export_count,
                                iree_hal_amdgpu_executable_reflection_t,
                                &reflection_offset),
              IREE_STRUCT_FIELD(counts->parameter_count,
                                iree_hal_executable_function_parameter_t,
                                &parameters_offset),
              IREE_STRUCT_FIELD_ALIGNED(
                  counts->layout_blob_byte_length, uint8_t,
                  iree_alignof(iree_hal_amdgpu_kernarg_layout_t),
                  &layout_blob_offset)));

  iree_hal_amdgpu_executable_metadata_t* metadata = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&metadata);
  if (iree_status_is_ok(status)) {
    memset(metadata, 0, total_size);
    uint8_t* storage_base = (uint8_t*)metadata;
    metadata->host_allocator = host_allocator;
    metadata->source = IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_UNKNOWN;
    metadata->export_count = counts->export_count;
    metadata->exports =
        counts->export_count
            ? (iree_hal_amdgpu_executable_export_t*)(storage_base +
                                                     exports_offset)
            : NULL;
    metadata->reflection =
        counts->export_count
            ? (iree_hal_amdgpu_executable_reflection_t*)(storage_base +
                                                         reflection_offset)
            : NULL;
    metadata->parameter_count = counts->parameter_count;
    metadata->parameters =
        counts->parameter_count
            ? (iree_hal_executable_function_parameter_t*)(storage_base +
                                                          parameters_offset)
            : NULL;
    metadata->layout_blob_capacity = counts->layout_blob_byte_length;
    metadata->layout_blob = counts->layout_blob_byte_length
                                ? storage_base + layout_blob_offset
                                : NULL;
    for (iree_host_size_t i = 0; i < metadata->export_count; ++i) {
      metadata->exports[i].kernarg_layout =
          iree_hal_amdgpu_kernarg_layout_ref_invalid();
    }
    *out_metadata = metadata;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_executable_metadata_free(
    iree_hal_amdgpu_executable_metadata_t* metadata) {
  if (!metadata) return;
  iree_allocator_t host_allocator = metadata->host_allocator;
  iree_allocator_free(host_allocator, metadata);
}

iree_status_t iree_hal_amdgpu_executable_metadata_append_layout(
    iree_hal_amdgpu_executable_metadata_t* metadata,
    iree_host_size_t layout_byte_length,
    iree_hal_amdgpu_kernarg_layout_ref_t* out_ref,
    iree_byte_span_t* out_storage) {
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(out_ref);
  IREE_ASSERT_ARGUMENT(out_storage);

  *out_ref = iree_hal_amdgpu_kernarg_layout_ref_invalid();
  *out_storage = iree_byte_span_empty();

  if (IREE_UNLIKELY(layout_byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "layout byte length must be non-zero");
  }

  iree_host_size_t aligned_offset = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
          metadata->layout_blob_used,
          iree_alignof(iree_hal_amdgpu_kernarg_layout_t), &aligned_offset))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "layout blob offset alignment overflow");
  }
  iree_host_size_t end_offset = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          aligned_offset, layout_byte_length, &end_offset))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "layout blob end offset overflow");
  }
  if (IREE_UNLIKELY(end_offset > metadata->layout_blob_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "layout blob requires %" PRIhsz
                            " bytes but capacity is %" PRIhsz,
                            end_offset, metadata->layout_blob_capacity);
  }

  metadata->layout_blob_used = end_offset;
  *out_ref = (iree_hal_amdgpu_kernarg_layout_ref_t){
      .byte_offset = aligned_offset,
  };
  *out_storage = iree_make_byte_span(metadata->layout_blob + aligned_offset,
                                     layout_byte_length);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_metadata_resolve_layout(
    const iree_hal_amdgpu_executable_metadata_t* metadata,
    iree_hal_amdgpu_kernarg_layout_ref_t ref,
    const iree_hal_amdgpu_kernarg_layout_t** out_layout) {
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(out_layout);

  *out_layout = NULL;

  if (IREE_UNLIKELY(!iree_hal_amdgpu_kernarg_layout_ref_is_valid(ref))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid kernarg layout reference");
  }
  if (IREE_UNLIKELY(!iree_host_size_has_alignment(
          ref.byte_offset, iree_alignof(iree_hal_amdgpu_kernarg_layout_t)))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "layout reference offset %" PRIhsz " is not aligned", ref.byte_offset);
  }
  if (IREE_UNLIKELY(ref.byte_offset >= metadata->layout_blob_used)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "layout reference offset %" PRIhsz
                            " exceeds used layout blob length %" PRIhsz,
                            ref.byte_offset, metadata->layout_blob_used);
  }
  *out_layout =
      (const iree_hal_amdgpu_kernarg_layout_t*)(metadata->layout_blob +
                                                ref.byte_offset);
  return iree_ok_status();
}
