// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/facts.h"
#include "loom/ops/encoding/storage.h"

static iree_status_t loom_aie2p_array_binding_dimension_value(
    const loom_value_fact_table_t* facts, loom_type_t type,
    iree_host_size_t dimension, uint32_t* out_value) {
  if (!loom_type_dim_is_dynamic_at(type, dimension)) {
    const int64_t value = loom_type_dim_static_size_at(type, dimension);
    if (value < 0 || value > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P binding dimension is out of range");
    }
    *out_value = (uint32_t)value;
    return iree_ok_status();
  }

  const loom_value_id_t value_id = loom_type_dim_value_id_at(type, dimension);
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t value = 0;
  if (!loom_value_facts_query_all_equal_element(
          &facts->context, loom_value_fact_table_lookup(facts, value_id),
          &element_facts) ||
      !loom_value_facts_as_exact_i64(element_facts, &value) || value < 0 ||
      value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P binding dimension must resolve to one exact non-negative u32 "
        "fact");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_binding_stride_value(
    const loom_value_fact_address_layout_t* layout, uint8_t dimension,
    uint64_t* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(layout->strides[dimension], &value) ||
      value < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P binding stride must resolve to one exact non-negative value");
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_binding_address_dimension(
    uint64_t element_stride, uint32_t dimension, int32_t element_bit_width,
    uint8_t address_granularity_bits, uint8_t dimension_index,
    uint8_t dimension_count,
    loom_aie2p_array_binding_dma_dimension_t* out_dimension) {
  uint64_t step_bits = 0;
  uint64_t wrap_bits = 0;
  const bool has_outer_dimension = dimension_index + 1u < dimension_count;
  if (dimension_index == 0 && element_bit_width < address_granularity_bits) {
    uint64_t packed_dimension_bits = 0;
    if (element_stride != 1 ||
        !iree_checked_mul_u64(dimension, (uint64_t)element_bit_width,
                              &packed_dimension_bits) ||
        packed_dimension_bits % address_granularity_bits != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding innermost dimension must form packed native "
          "address units");
    }
    step_bits = address_granularity_bits;
    if (has_outer_dimension) wrap_bits = packed_dimension_bits;
  } else {
    if (!iree_checked_mul_u64(element_stride, (uint64_t)element_bit_width,
                              &step_bits) ||
        step_bits % address_granularity_bits != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding stride must resolve to whole native address units");
    }
    if (has_outer_dimension) {
      wrap_bits = (uint64_t)dimension * address_granularity_bits;
    }
  }

  const uint64_t step_size = step_bits / address_granularity_bits;
  const uint64_t wrap = wrap_bits / address_granularity_bits;
  if (step_size == 0 || step_size > UINT32_MAX ||
      (has_outer_dimension && (wrap == 0 || wrap > UINT32_MAX))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P binding address dimension overflows");
  }
  *out_dimension = (loom_aie2p_array_binding_dma_dimension_t){
      .step_size = (uint32_t)step_size,
      .wrap = (uint32_t)wrap,
  };
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_plan_binding_transfer(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_xdna_array_family_t* family, loom_type_t source_type,
    loom_type_t record_type, bool partitioned, uint32_t partition_lane,
    uint32_t logical_record_byte_length, uint32_t logical_record_count,
    const loom_xdna_dma_facts_t* dma_facts,
    loom_aie2p_array_binding_plan_t* binding_plan) {
  uint64_t logical_transfer_byte_length = 0;
  if (!iree_checked_mul_u64(logical_record_byte_length, logical_record_count,
                            &logical_transfer_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P binding transfer size overflows");
  }

  uint64_t binding_byte_offset = 0;
  uint64_t binding_span_byte_length = logical_transfer_byte_length;
  uint64_t transfer_byte_length = logical_transfer_byte_length;
  uint64_t task_repeat_count = 1;
  uint8_t dma_dimension_count = 0;
  loom_aie2p_array_binding_dma_dimension_t
      dma_dimensions[LOOM_AIE2P_ARRAY_BINDING_DMA_DIMENSION_COUNT] = {0};

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          &facts->context, module, source_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P binding source requires one exact address layout");
  }
  if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    if (partitioned &&
        !iree_checked_mul_u64(logical_transfer_byte_length, partition_lane,
                              &binding_byte_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P binding partition offset overflows");
    }
  } else if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    const uint8_t source_rank = loom_type_rank(source_type);
    const uint8_t record_rank = loom_type_rank(record_type);
    const uint8_t first_source_axis = partitioned ? 1 : 0;
    if (source_rank != layout.rank || !layout.strides ||
        source_rank < record_rank ||
        first_source_axis > source_rank - record_rank) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding source has a malformed strided layout");
    }
    const uint8_t record_start = source_rank - record_rank;
    const int32_t element_bit_width =
        loom_scalar_type_bitwidth(loom_type_element_type(source_type));
    if (element_bit_width <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P binding element type has no bit width");
    }

    if (partitioned) {
      uint64_t lane_element_stride = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_stride_value(
          &layout, 0, &lane_element_stride));
      uint64_t binding_bit_offset = 0;
      if (!iree_checked_mul_u64(lane_element_stride, partition_lane,
                                &binding_bit_offset) ||
          !iree_checked_mul_u64(binding_bit_offset, (uint64_t)element_bit_width,
                                &binding_bit_offset) ||
          (binding_bit_offset & 7u) != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding partition offset is not whole-byte addressable");
      }
      binding_byte_offset = binding_bit_offset / 8u;
    }

    uint8_t first_transfer_axis = first_source_axis;
    for (; first_transfer_axis < record_start; ++first_transfer_axis) {
      uint64_t element_stride = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_stride_value(
          &layout, first_transfer_axis, &element_stride));
      if (element_stride != 0) break;
      uint32_t dimension = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_dimension_value(
          facts, source_type, first_transfer_axis, &dimension));
      if (!iree_checked_mul_u64(task_repeat_count, dimension,
                                &task_repeat_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P binding repeat count overflows");
      }
    }

    uint64_t transfer_element_count = 1;
    uint64_t maximum_element_offset = 0;
    bool transfer_is_dense = true;
    for (uint8_t axis = source_rank; axis-- > first_transfer_axis;) {
      uint32_t dimension = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_dimension_value(
          facts, source_type, axis, &dimension));
      uint64_t element_stride = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_stride_value(
          &layout, axis, &element_stride));
      if (dimension == 0 || element_stride == 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P shim DMA requires nonzero transfer dimensions beneath "
            "any repeated sequence prefix");
      }
      transfer_is_dense &= element_stride == transfer_element_count;
      if (!iree_checked_mul_u64(transfer_element_count, dimension,
                                &transfer_element_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P binding transfer shape overflows");
      }
      uint64_t axis_maximum_offset = 0;
      if (!iree_checked_mul_u64(element_stride, dimension - 1u,
                                &axis_maximum_offset) ||
          !iree_checked_add_u64(maximum_element_offset, axis_maximum_offset,
                                &maximum_element_offset)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P binding address span overflows");
      }
    }

    uint64_t transfer_bit_length = 0;
    uint64_t binding_span_bit_length = 0;
    if (!iree_checked_mul_u64(transfer_element_count,
                              (uint64_t)element_bit_width,
                              &transfer_bit_length) ||
        !iree_checked_add_u64(maximum_element_offset, 1u,
                              &maximum_element_offset) ||
        !iree_checked_mul_u64(maximum_element_offset,
                              (uint64_t)element_bit_width,
                              &binding_span_bit_length) ||
        (transfer_bit_length & 7u) != 0 ||
        (binding_span_bit_length & 7u) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding layout does not resolve to whole-byte spans");
    }
    transfer_byte_length = transfer_bit_length / 8u;
    binding_span_byte_length = binding_span_bit_length / 8u;

    uint64_t repeated_transfer_byte_length = 0;
    if (!iree_checked_mul_u64(transfer_byte_length, task_repeat_count,
                              &repeated_transfer_byte_length) ||
        repeated_transfer_byte_length != logical_transfer_byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding layout does not cover its logical record stream");
    }

    if (!transfer_is_dense) {
      const uint8_t transfer_rank = source_rank - first_transfer_axis;
      if (transfer_rank > dma_facts->address_dimension_count ||
          transfer_rank > LOOM_AIE2P_ARRAY_BINDING_DMA_DIMENSION_COUNT) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P binding requires more shim DMA address dimensions than "
            "the device provides");
      }
      dma_dimension_count = transfer_rank;
      for (uint8_t axis = source_rank; axis-- > first_transfer_axis;) {
        const uint8_t dimension_index = source_rank - axis - 1u;
        uint32_t dimension = 0;
        IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_dimension_value(
            facts, source_type, axis, &dimension));
        uint64_t element_stride = 0;
        IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_stride_value(
            &layout, axis, &element_stride));
        IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_address_dimension(
            element_stride, dimension, element_bit_width,
            family->address_generation_granularity_bits, dimension_index,
            transfer_rank, &dma_dimensions[dimension_index]));
      }

      const uint64_t maximum_step_size = UINT64_C(1)
                                         << dma_facts->step_size_bits;
      const uint64_t maximum_wrap = UINT64_C(1) << dma_facts->wrap_bits;
      for (uint8_t i = 0; i < dma_dimension_count; ++i) {
        if (dma_dimensions[i].step_size > maximum_step_size ||
            dma_dimensions[i].wrap > maximum_wrap) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "AIE2P binding address dimension exceeds hardware fields");
        }
      }
    }
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P binding address layout is unknown");
  }

  if (transfer_byte_length == 0 || transfer_byte_length > UINT32_MAX ||
      transfer_byte_length % dma_facts->transfer_length_granularity != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P shim DMA span is not representable");
  }
  if (task_repeat_count == 0 ||
      task_repeat_count > dma_facts->maximum_task_repeat_count ||
      task_repeat_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P shim DMA task repeat count exceeds hardware");
  }
  if (binding_byte_offset > INT64_MAX ||
      binding_byte_offset % dma_facts->address_alignment != 0 ||
      binding_span_byte_length > dma_facts->address_maximum ||
      transfer_byte_length > dma_facts->address_maximum) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P binding DMA address span is not representable");
  }

  binding_plan->binding_byte_offset = binding_byte_offset;
  binding_plan->binding_span_byte_length = binding_span_byte_length;
  binding_plan->transfer_byte_length = (uint32_t)transfer_byte_length;
  binding_plan->task_repeat_count = (uint16_t)task_repeat_count;
  binding_plan->dma_dimension_count = dma_dimension_count;
  memcpy(binding_plan->dma_dimensions, dma_dimensions,
         sizeof(binding_plan->dma_dimensions));
  return iree_ok_status();
}
