// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"

#include <string.h>

static bool loom_aie2p_array_reject_binding_transfer(
    iree_string_view_t reason, iree_string_view_t* out_reason) {
  *out_reason = reason;
  return false;
}

static bool loom_aie2p_array_binding_append_dimension(
    loom_storage_geometry_axis_t axis, uint16_t element_bit_width,
    uint8_t address_granularity_bits, bool has_outer_dimension,
    const loom_xdna_dma_facts_t* dma_facts, uint8_t* dimension_count,
    loom_aie2p_array_binding_dma_dimension_t* dimensions,
    iree_string_view_t* out_reason) {
  if (*dimension_count == dma_facts->address_dimension_count) {
    return loom_aie2p_array_reject_binding_transfer(
        IREE_SV("the strided transfer requires more address dimensions "
                "than DMA provides"),
        out_reason);
  }
  uint64_t step_bits = 0;
  uint64_t wrap = 0;
  if (*dimension_count == 0 && element_bit_width < address_granularity_bits) {
    uint64_t packed_dimension_bits = 0;
    if (axis.element_stride != 1 ||
        !iree_checked_mul_u64(axis.extent, (uint64_t)element_bit_width,
                              &packed_dimension_bits) ||
        packed_dimension_bits % address_granularity_bits != 0) {
      return loom_aie2p_array_reject_binding_transfer(
          IREE_SV("the innermost transfer dimension must form packed native "
                  "address units"),
          out_reason);
    }
    step_bits = address_granularity_bits;
    if (has_outer_dimension) {
      wrap = packed_dimension_bits / address_granularity_bits;
    }
  } else {
    if (!iree_checked_mul_u64(axis.element_stride, (uint64_t)element_bit_width,
                              &step_bits) ||
        step_bits % address_granularity_bits != 0) {
      return loom_aie2p_array_reject_binding_transfer(
          IREE_SV("every transfer stride must form whole native address "
                  "units"),
          out_reason);
    }
    if (has_outer_dimension) {
      wrap = axis.extent;
    }
  }

  const uint64_t step_size = step_bits / address_granularity_bits;
  const uint64_t maximum_step_size = UINT64_C(1) << dma_facts->step_size_bits;
  const uint64_t maximum_wrap = (UINT64_C(1) << dma_facts->wrap_bits) - 1;
  if (step_size == 0 || step_size > maximum_step_size) {
    return loom_aie2p_array_reject_binding_transfer(
        IREE_SV("an address dimension exceeds the DMA step field"), out_reason);
  }
  // Step fields encode step_size - 1; wrap fields encode the actual count.
  // Zero disables wrapping and cannot represent one past the field maximum.
  if (has_outer_dimension && (wrap == 0 || wrap > maximum_wrap)) {
    return loom_aie2p_array_reject_binding_transfer(
        IREE_SV("an address dimension exceeds the nonzero DMA wrap field"),
        out_reason);
  }
  dimensions[(*dimension_count)++] = (loom_aie2p_array_binding_dma_dimension_t){
      .step_size = (uint32_t)step_size,
      .wrap = (uint32_t)wrap,
  };
  return true;
}

bool loom_aie2p_array_resolve_binding_transfer(
    const loom_storage_geometry_t* geometry, uint8_t record_rank,
    const loom_xdna_array_family_t* family, uint64_t binding_view_byte_offset,
    bool partitioned, uint32_t partition_lane,
    uint32_t logical_record_byte_length, uint32_t logical_record_count,
    const loom_xdna_dma_facts_t* dma_facts,
    loom_aie2p_array_binding_plan_t* binding_plan,
    iree_string_view_t* out_reason) {
  *out_reason = iree_string_view_empty();
  const uint64_t logical_transfer_byte_length =
      (uint64_t)logical_record_byte_length * logical_record_count;

  uint64_t binding_byte_offset = binding_view_byte_offset;
  uint64_t binding_span_byte_length = logical_transfer_byte_length;
  uint64_t transfer_byte_length = logical_transfer_byte_length;
  uint64_t task_repeat_count = 1;
  uint8_t dma_dimension_count = 0;
  loom_aie2p_array_binding_dma_dimension_t
      dma_dimensions[LOOM_AIE2P_ARRAY_BINDING_DMA_DIMENSION_CAPACITY] = {0};

  if (geometry->layout_kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    if (partitioned) {
      uint64_t partition_byte_offset = 0;
      if (!iree_checked_mul_u64(logical_transfer_byte_length, partition_lane,
                                &partition_byte_offset) ||
          !iree_checked_add_u64(binding_byte_offset, partition_byte_offset,
                                &binding_byte_offset)) {
        return loom_aie2p_array_reject_binding_transfer(
            IREE_SV("the selected partition offset must fit in 64 bits"),
            out_reason);
      }
    }
  } else {
    const uint8_t source_rank = geometry->rank;
    const uint8_t first_source_axis = partitioned ? 1 : 0;
    const uint8_t record_start = source_rank - record_rank;
    const uint16_t element_bit_width = geometry->element_bit_count;

    if (partitioned) {
      const uint64_t lane_element_stride = geometry->axes[0].element_stride;
      uint64_t binding_bit_offset = 0;
      if (!iree_checked_mul_u64(lane_element_stride, partition_lane,
                                &binding_bit_offset) ||
          !iree_checked_mul_u64(binding_bit_offset, (uint64_t)element_bit_width,
                                &binding_bit_offset) ||
          (binding_bit_offset & 7u) != 0) {
        return loom_aie2p_array_reject_binding_transfer(
            IREE_SV("the selected partition must begin at a whole-byte "
                    "representable offset"),
            out_reason);
      }
      if (!iree_checked_add_u64(binding_byte_offset, binding_bit_offset / 8u,
                                &binding_byte_offset)) {
        return loom_aie2p_array_reject_binding_transfer(
            IREE_SV("the selected partition offset must fit in 64 bits"),
            out_reason);
      }
    }

    uint8_t first_transfer_axis = first_source_axis;
    for (; first_transfer_axis < record_start; ++first_transfer_axis) {
      const loom_storage_geometry_axis_t axis =
          geometry->axes[first_transfer_axis];
      if (axis.extent == 1) {
        continue;
      }
      if (axis.element_stride != 0) {
        break;
      }
      // Logical topology admission already bounds the complete record count.
      task_repeat_count *= axis.extent;
    }

    uint8_t transfer_rank = 0;
    for (uint8_t axis = first_transfer_axis; axis < source_rank; ++axis) {
      // A single coordinate contributes no address advance, regardless of its
      // stride. It consumes neither a DMA dimension nor a repetition field.
      if (geometry->axes[axis].extent == 1) {
        continue;
      }
      ++transfer_rank;
      if (geometry->axes[axis].element_stride == 0) {
        return loom_aie2p_array_reject_binding_transfer(
            IREE_SV("only leading source sequence axes may use zero stride "
                    "for task repetition"),
            out_reason);
      }
    }
    loom_storage_geometry_span_t span = {0};
    if (!loom_storage_geometry_measure(geometry, first_transfer_axis, &span)) {
      return loom_aie2p_array_reject_binding_transfer(
          IREE_SV("the transfer shape and source address span must fit in "
                  "64 bits"),
          out_reason);
    }

    uint64_t transfer_bit_length = 0;
    uint64_t binding_span_bit_length = 0;
    if (!iree_checked_mul_u64(span.element_count, (uint64_t)element_bit_width,
                              &transfer_bit_length) ||
        !iree_checked_mul_u64(span.element_span, (uint64_t)element_bit_width,
                              &binding_span_bit_length) ||
        (transfer_bit_length & 7u) != 0 ||
        (binding_span_bit_length & 7u) != 0) {
      return loom_aie2p_array_reject_binding_transfer(
          IREE_SV("the transfer shape must form whole-byte transfer and "
                  "address spans"),
          out_reason);
    }
    transfer_byte_length = transfer_bit_length / 8u;
    binding_span_byte_length = binding_span_bit_length / 8u;

    if (!span.dense) {
      const uint8_t unit_bits = family->address_generation_granularity_bits;
      const bool wide = element_bit_width > unit_bits;
      const uint16_t axis_bits = wide ? unit_bits : element_bit_width;
      const uint64_t maximum_wrap = (UINT64_C(1) << dma_facts->wrap_bits) - 1;
      // A wide scalar contributes an innermost run of consecutive native
      // words. Keep it separate across gaps, or when a combined wrap would
      // exceed its field. The measured transfer and envelope bound the
      // products in native words below.
      const uint16_t words = wide ? element_bit_width / unit_bits : 0;
      loom_storage_geometry_axis_t inner = {.extent = words,
                                            .element_stride = 1};
      for (uint8_t axis = source_rank; axis-- > first_transfer_axis;) {
        loom_storage_geometry_axis_t source_axis = geometry->axes[axis];
        if (source_axis.extent == 1) {
          continue;
        }
        --transfer_rank;
        if (wide) {
          source_axis.element_stride *= words;
          if (source_axis.element_stride ==
                  inner.extent * inner.element_stride &&
              (!transfer_rank ||
               inner.extent * source_axis.extent <= maximum_wrap)) {
            inner.extent *= source_axis.extent;
            continue;
          }
        }
        if (inner.extent &&
            !loom_aie2p_array_binding_append_dimension(
                inner, axis_bits, unit_bits, true, dma_facts,
                &dma_dimension_count, dma_dimensions, out_reason)) {
          return false;
        }
        inner = source_axis;
      }
      if (!loom_aie2p_array_binding_append_dimension(
              inner, axis_bits, unit_bits, false, dma_facts,
              &dma_dimension_count, dma_dimensions, out_reason)) {
        return false;
      }
    }
  }

  const uint64_t scaled_transfer_length =
      transfer_byte_length / dma_facts->transfer_length_granularity;
  const uint64_t minimum_transfer_units =
      iree_max(1u, dma_facts->transfer_length_offset);
  const uint64_t maximum_transfer_units =
      (uint64_t)dma_facts->maximum_encoded_transfer_length +
      dma_facts->transfer_length_offset;
  if (transfer_byte_length == 0 || transfer_byte_length > UINT32_MAX ||
      transfer_byte_length % dma_facts->transfer_length_granularity != 0 ||
      scaled_transfer_length < minimum_transfer_units ||
      scaled_transfer_length > maximum_transfer_units) {
    return loom_aie2p_array_reject_binding_transfer(
        IREE_SV("the transfer length must fit the DMA length field in "
                "native units"),
        out_reason);
  }
  if (task_repeat_count > dma_facts->maximum_task_repeat_count) {
    return loom_aie2p_array_reject_binding_transfer(
        IREE_SV("the repeated task count exceeds the DMA queue field"),
        out_reason);
  }
  if (binding_byte_offset > INT64_MAX ||
      binding_byte_offset % dma_facts->address_alignment != 0 ||
      binding_span_byte_length > dma_facts->address_maximum ||
      binding_byte_offset >
          dma_facts->address_maximum - binding_span_byte_length) {
    return loom_aie2p_array_reject_binding_transfer(
        IREE_SV("the aligned binding offset and reachable span must fit the "
                "DMA address domain"),
        out_reason);
  }

  binding_plan->binding_byte_offset = binding_byte_offset;
  binding_plan->binding_span_byte_length = binding_span_byte_length;
  binding_plan->transfer_byte_length = (uint32_t)transfer_byte_length;
  binding_plan->task_repeat_count = (uint16_t)task_repeat_count;
  binding_plan->dma_dimension_count = dma_dimension_count;
  memcpy(binding_plan->dma_dimensions, dma_dimensions,
         sizeof(binding_plan->dma_dimensions));
  return true;
}
