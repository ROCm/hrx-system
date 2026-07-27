// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/kernarg_layout.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_kernarg_layout_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_kernarg_byte_range_t {
  // Byte offset of the first byte in the range.
  iree_host_size_t offset;
  // Number of bytes covered by the range.
  iree_host_size_t length;
} iree_hal_amdgpu_kernarg_byte_range_t;

static bool iree_hal_amdgpu_kernarg_layout_range_end(
    iree_hal_amdgpu_kernarg_byte_range_t range, iree_host_size_t* out_end) {
  return iree_host_size_checked_add(range.offset, range.length, out_end);
}

static bool iree_hal_amdgpu_kernarg_layout_ranges_overlap(
    iree_hal_amdgpu_kernarg_byte_range_t lhs,
    iree_hal_amdgpu_kernarg_byte_range_t rhs) {
  iree_host_size_t lhs_end = 0;
  iree_host_size_t rhs_end = 0;
  if (!iree_hal_amdgpu_kernarg_layout_range_end(lhs, &lhs_end) ||
      !iree_hal_amdgpu_kernarg_layout_range_end(rhs, &rhs_end)) {
    return true;
  }
  return lhs.offset < rhs_end && rhs.offset < lhs_end;
}

static iree_hal_amdgpu_kernarg_byte_range_t
iree_hal_amdgpu_kernarg_layout_binding_range(
    iree_hal_amdgpu_kernarg_binding_slot_t binding_slot) {
  return (iree_hal_amdgpu_kernarg_byte_range_t){
      .offset =
          (iree_host_size_t)binding_slot.target_qword_index * sizeof(uint64_t),
      .length = sizeof(uint64_t),
  };
}

static iree_hal_amdgpu_kernarg_byte_range_t
iree_hal_amdgpu_kernarg_layout_constant_target_range(
    iree_hal_amdgpu_kernarg_constant_span_t constant_span) {
  return (iree_hal_amdgpu_kernarg_byte_range_t){
      .offset = constant_span.target_byte_offset,
      .length = constant_span.byte_length,
  };
}

static iree_hal_amdgpu_kernarg_byte_range_t
iree_hal_amdgpu_kernarg_layout_constant_source_range(
    iree_hal_amdgpu_kernarg_constant_span_t constant_span) {
  return (iree_hal_amdgpu_kernarg_byte_range_t){
      .offset = constant_span.source_byte_offset,
      .length = constant_span.byte_length,
  };
}

static iree_status_t iree_hal_amdgpu_kernarg_layout_validate_target_range(
    iree_host_size_t kernarg_byte_length,
    iree_hal_amdgpu_kernarg_byte_range_t range) {
  iree_host_size_t range_end = 0;
  if (IREE_UNLIKELY(
          !iree_hal_amdgpu_kernarg_layout_range_end(range, &range_end) ||
          range_end > kernarg_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernarg range [%" PRIhsz ", %" PRIhsz
                            ") exceeds kernarg byte length %" PRIhsz,
                            range.offset, range_end, kernarg_byte_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_kernarg_layout_validate_source_range(
    iree_host_size_t constant_byte_length,
    iree_hal_amdgpu_kernarg_byte_range_t range) {
  iree_host_size_t range_end = 0;
  if (IREE_UNLIKELY(
          !iree_hal_amdgpu_kernarg_layout_range_end(range, &range_end) ||
          range_end > constant_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "constant range [%" PRIhsz ", %" PRIhsz
                            ") exceeds constant byte length %" PRIhsz,
                            range.offset, range_end, constant_byte_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_kernarg_layout_validate_no_target_overlap(
    const iree_hal_amdgpu_kernarg_layout_params_t* params) {
  for (iree_host_size_t i = 0; i < params->binding_count; ++i) {
    const iree_hal_amdgpu_kernarg_byte_range_t lhs =
        iree_hal_amdgpu_kernarg_layout_binding_range(params->binding_slots[i]);
    for (iree_host_size_t j = i + 1; j < params->binding_count; ++j) {
      const iree_hal_amdgpu_kernarg_byte_range_t rhs =
          iree_hal_amdgpu_kernarg_layout_binding_range(
              params->binding_slots[j]);
      if (IREE_UNLIKELY(
              iree_hal_amdgpu_kernarg_layout_ranges_overlap(lhs, rhs))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "binding kernarg ranges overlap");
      }
    }
    for (iree_host_size_t j = 0; j < params->constant_span_count; ++j) {
      const iree_hal_amdgpu_kernarg_byte_range_t rhs =
          iree_hal_amdgpu_kernarg_layout_constant_target_range(
              params->constant_spans[j]);
      if (IREE_UNLIKELY(
              iree_hal_amdgpu_kernarg_layout_ranges_overlap(lhs, rhs))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "binding and constant kernarg ranges overlap");
      }
    }
  }
  for (iree_host_size_t i = 0; i < params->constant_span_count; ++i) {
    const iree_hal_amdgpu_kernarg_byte_range_t lhs =
        iree_hal_amdgpu_kernarg_layout_constant_target_range(
            params->constant_spans[i]);
    for (iree_host_size_t j = i + 1; j < params->constant_span_count; ++j) {
      const iree_hal_amdgpu_kernarg_byte_range_t rhs =
          iree_hal_amdgpu_kernarg_layout_constant_target_range(
              params->constant_spans[j]);
      if (IREE_UNLIKELY(
              iree_hal_amdgpu_kernarg_layout_ranges_overlap(lhs, rhs))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "constant kernarg ranges overlap");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_kernarg_layout_validate_no_source_overlap(
    const iree_hal_amdgpu_kernarg_layout_params_t* params,
    iree_host_size_t* out_constant_write_byte_length) {
  *out_constant_write_byte_length = 0;
  for (iree_host_size_t i = 0; i < params->constant_span_count; ++i) {
    const iree_hal_amdgpu_kernarg_byte_range_t lhs =
        iree_hal_amdgpu_kernarg_layout_constant_source_range(
            params->constant_spans[i]);
    iree_host_size_t total_write_byte_length = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            *out_constant_write_byte_length, lhs.length,
            &total_write_byte_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "constant write byte length overflow");
    }
    *out_constant_write_byte_length = total_write_byte_length;
    for (iree_host_size_t j = i + 1; j < params->constant_span_count; ++j) {
      const iree_hal_amdgpu_kernarg_byte_range_t rhs =
          iree_hal_amdgpu_kernarg_layout_constant_source_range(
              params->constant_spans[j]);
      if (IREE_UNLIKELY(
              iree_hal_amdgpu_kernarg_layout_ranges_overlap(lhs, rhs))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "constant source ranges overlap");
      }
    }
  }
  if (IREE_UNLIKELY(*out_constant_write_byte_length !=
                    params->constant_byte_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "constant spans cover %" PRIhsz
                            " bytes but constant stream has %" PRIhsz " bytes",
                            *out_constant_write_byte_length,
                            params->constant_byte_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_kernarg_layout_validate_params(
    const iree_hal_amdgpu_kernarg_layout_params_t* params) {
  if (IREE_UNLIKELY(params->kernarg_byte_length >
                    IREE_HAL_AMDGPU_KERNARG_LAYOUT_MAX_BYTE_LENGTH)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernarg byte length %" PRIhsz
                            " exceeds layout limit %" PRIu16,
                            params->kernarg_byte_length,
                            IREE_HAL_AMDGPU_KERNARG_LAYOUT_MAX_BYTE_LENGTH);
  }
  if (IREE_UNLIKELY(
          params->kernarg_alignment == 0 ||
          params->kernarg_alignment > UINT16_MAX ||
          !iree_host_size_is_power_of_two(params->kernarg_alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernarg alignment must be a non-zero power of two "
                            "that fits uint16_t; got %" PRIhsz,
                            params->kernarg_alignment);
  }
  if (IREE_UNLIKELY(params->constant_byte_length > UINT16_MAX ||
                    params->binding_count > UINT16_MAX ||
                    params->constant_span_count > UINT16_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "layout counts must fit uint16_t; constants=%" PRIhsz
        ", bindings=%" PRIhsz ", constant_spans=%" PRIhsz,
        params->constant_byte_length, params->binding_count,
        params->constant_span_count);
  }
  if (IREE_UNLIKELY(params->binding_count && !params->binding_slots)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binding slots are required when binding_count > 0");
  }
  if (IREE_UNLIKELY(params->constant_span_count && !params->constant_spans)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "constant spans are required when constant_span_count > 0");
  }
  if (params->implicit_args_byte_offset !=
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE) {
    if (IREE_UNLIKELY(params->implicit_args_byte_offset >
                      params->kernarg_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "implicit args offset %" PRIhsz
                              " exceeds kernarg byte length %" PRIhsz,
                              params->implicit_args_byte_offset,
                              params->kernarg_byte_length);
    }
    if (IREE_UNLIKELY(!iree_host_size_has_alignment(
            params->implicit_args_byte_offset, sizeof(uint64_t)))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "implicit args offset %" PRIhsz
                              " must be 8-byte aligned",
                              params->implicit_args_byte_offset);
    }
  }
  for (iree_host_size_t i = 0; i < params->binding_count; ++i) {
    const iree_hal_amdgpu_kernarg_byte_range_t range =
        iree_hal_amdgpu_kernarg_layout_binding_range(params->binding_slots[i]);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_validate_target_range(
        params->kernarg_byte_length, range));
  }
  for (iree_host_size_t i = 0; i < params->constant_span_count; ++i) {
    if (IREE_UNLIKELY(params->constant_spans[i].byte_length == 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant span byte length must be non-zero");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_validate_target_range(
        params->kernarg_byte_length,
        iree_hal_amdgpu_kernarg_layout_constant_target_range(
            params->constant_spans[i])));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_validate_source_range(
        params->constant_byte_length,
        iree_hal_amdgpu_kernarg_layout_constant_source_range(
            params->constant_spans[i])));
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_kernarg_layout_validate_no_target_overlap(params));
  iree_host_size_t constant_write_byte_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_kernarg_layout_validate_no_source_overlap(
          params, &constant_write_byte_length));
  return iree_ok_status();
}

static bool iree_hal_amdgpu_kernarg_layout_has_packed_binding_prefix(
    const iree_hal_amdgpu_kernarg_layout_params_t* params) {
  if (params->binding_count == 0) return false;
  for (iree_host_size_t i = 0; i < params->binding_count; ++i) {
    if (params->binding_slots[i].target_qword_index != i) return false;
  }
  return true;
}

static bool iree_hal_amdgpu_kernarg_layout_has_contiguous_constants(
    const iree_hal_amdgpu_kernarg_layout_params_t* params) {
  if (params->constant_byte_length == 0) return false;
  if (params->constant_span_count == 0) return false;
  iree_host_size_t next_source_offset = 0;
  for (iree_host_size_t i = 0; i < params->constant_span_count; ++i) {
    if (params->constant_spans[i].source_byte_offset != next_source_offset) {
      return false;
    }
    next_source_offset += params->constant_spans[i].byte_length;
  }
  return next_source_offset == params->constant_byte_length;
}

static iree_status_t iree_hal_amdgpu_kernarg_layout_total_written_bytes(
    const iree_hal_amdgpu_kernarg_layout_params_t* params,
    iree_host_size_t* out_written_byte_length) {
  iree_host_size_t written_byte_length =
      params->binding_count * sizeof(uint64_t);
  for (iree_host_size_t i = 0; i < params->constant_span_count; ++i) {
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            written_byte_length, params->constant_spans[i].byte_length,
            &written_byte_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kernarg written byte length overflow");
    }
  }
  *out_written_byte_length = written_byte_length;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_kernarg_layout_storage_size(
    iree_host_size_t binding_count, iree_host_size_t constant_span_count,
    iree_host_size_t* out_storage_byte_length) {
  IREE_ASSERT_ARGUMENT(out_storage_byte_length);
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_kernarg_layout_t), out_storage_byte_length,
      IREE_STRUCT_FIELD(binding_count, iree_hal_amdgpu_kernarg_binding_slot_t,
                        NULL),
      IREE_STRUCT_FIELD(constant_span_count,
                        iree_hal_amdgpu_kernarg_constant_span_t, NULL));
}

iree_status_t iree_hal_amdgpu_kernarg_layout_initialize(
    const iree_hal_amdgpu_kernarg_layout_params_t* params,
    iree_host_size_t storage_capacity,
    iree_hal_amdgpu_kernarg_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_layout);

  iree_host_size_t storage_byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_storage_size(
      params->binding_count, params->constant_span_count,
      &storage_byte_length));
  if (IREE_UNLIKELY(storage_capacity < storage_byte_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "layout storage capacity %" PRIhsz
                            " is smaller than required length %" PRIhsz,
                            storage_capacity, storage_byte_length);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_validate_params(params));

  iree_hal_amdgpu_kernarg_layout_flags_t flags =
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_NONE;
  if (params->implicit_args_byte_offset !=
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE) {
    flags |= IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS;
  }
  if (iree_hal_amdgpu_kernarg_layout_has_packed_binding_prefix(params)) {
    flags |= IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX;
  }
  if (iree_hal_amdgpu_kernarg_layout_has_contiguous_constants(params)) {
    flags |= IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_CONTIGUOUS_CONSTANTS;
  }

  iree_host_size_t written_byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_total_written_bytes(
      params, &written_byte_length));
  if (written_byte_length != params->kernarg_byte_length ||
      iree_any_bit_set(flags,
                       IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS)) {
    flags |= IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL;
  }

  out_layout->kernarg_byte_length = (uint16_t)params->kernarg_byte_length;
  out_layout->kernarg_alignment = (uint16_t)params->kernarg_alignment;
  out_layout->binding_count = (uint16_t)params->binding_count;
  out_layout->constant_span_count = (uint16_t)params->constant_span_count;
  out_layout->constant_byte_length = (uint16_t)params->constant_byte_length;
  out_layout->implicit_args_byte_offset =
      (uint16_t)params->implicit_args_byte_offset;
  out_layout->flags = flags;

  if (params->binding_count > 0) {
    memmove(out_layout->binding_slots, params->binding_slots,
            params->binding_count * sizeof(params->binding_slots[0]));
  }
  if (params->constant_span_count > 0) {
    iree_hal_amdgpu_kernarg_constant_span_t* constant_spans =
        (iree_hal_amdgpu_kernarg_constant_span_t*)(out_layout->binding_slots +
                                                   out_layout->binding_count);
    memmove(constant_spans, params->constant_spans,
            params->constant_span_count * sizeof(params->constant_spans[0]));
  }
  return iree_ok_status();
}

void iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
    const iree_hal_amdgpu_kernarg_layout_t* layout,
    const uint64_t* binding_ptrs, iree_const_byte_span_t constants,
    void* kernarg_ptr) {
  uint8_t* target = (uint8_t*)kernarg_ptr;
  if (iree_any_bit_set(
          layout->flags,
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL)) {
    memset(target, 0, layout->kernarg_byte_length);
  }

  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  if (iree_any_bit_set(
          layout->flags,
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX)) {
    memcpy(target, binding_ptrs, layout->binding_count * sizeof(uint64_t));
  } else {
    for (iree_host_size_t i = 0; i < layout->binding_count; ++i) {
      memcpy(target + binding_slots[i].target_qword_index * sizeof(uint64_t),
             &binding_ptrs[i], sizeof(uint64_t));
    }
  }

  const iree_hal_amdgpu_kernarg_constant_span_t* constant_spans =
      iree_hal_amdgpu_kernarg_layout_constant_spans(layout);
  for (iree_host_size_t i = 0; i < layout->constant_span_count; ++i) {
    const iree_hal_amdgpu_kernarg_constant_span_t span = constant_spans[i];
    memcpy(target + span.target_byte_offset,
           constants.data + span.source_byte_offset, span.byte_length);
  }
}
