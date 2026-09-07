// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"

#include <inttypes.h>

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

iree_status_t loom_aie2p_array_plan_binding_transfer(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    loom_type_t partition_source_type, loom_type_t record_type,
    uint32_t partition_lane, uint32_t logical_record_byte_length,
    uint32_t logical_record_count, const loom_xdna_dma_facts_t* dma_facts,
    loom_aie2p_array_binding_plan_t* binding_plan) {
  uint64_t logical_transfer_byte_length = 0;
  if (!iree_checked_mul_u64(logical_record_byte_length, logical_record_count,
                            &logical_transfer_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P binding transfer size overflows");
  }

  uint64_t binding_byte_offset = 0;
  uint64_t transfer_byte_length = logical_transfer_byte_length;
  uint64_t task_repeat_count = 1;
  if (loom_type_kind(partition_source_type) != LOOM_TYPE_NONE) {
    loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
    loom_value_fact_address_layout_t layout = {0};
    if (!loom_encoding_query_type_address_layout(
            &facts->context, module, partition_source_type, stride_storage,
            IREE_ARRAYSIZE(stride_storage), &layout)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding partition requires an exact address layout");
    }
    if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
      if (!iree_checked_mul_u64(logical_transfer_byte_length, partition_lane,
                                &binding_byte_offset)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P binding partition offset overflows");
      }
    } else if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
      const uint8_t source_rank = loom_type_rank(partition_source_type);
      const uint8_t record_rank = loom_type_rank(record_type);
      if (source_rank == 0 || source_rank != layout.rank || !layout.strides ||
          source_rank <= record_rank) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding partition has a malformed strided layout");
      }
      const uint8_t record_start = source_rank - record_rank;
      uint64_t dense_inner_element_count = 1;
      bool repeating = false;
      for (uint8_t axis = source_rank; axis-- > 1;) {
        uint32_t dimension = 0;
        IREE_RETURN_IF_ERROR(loom_aie2p_array_binding_dimension_value(
            facts, partition_source_type, axis, &dimension));
        int64_t element_stride = 0;
        if (!loom_value_facts_as_exact_i64(layout.strides[axis],
                                           &element_stride) ||
            element_stride < 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AIE2P binding stride must be one exact non-negative value");
        }
        if (!repeating &&
            (uint64_t)element_stride == dense_inner_element_count) {
          if (!iree_checked_mul_u64(dense_inner_element_count, dimension,
                                    &dense_inner_element_count)) {
            return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "AIE2P binding dense span overflows");
          }
        } else if (axis < record_start && element_stride == 0) {
          repeating = true;
          if (!iree_checked_mul_u64(task_repeat_count, dimension,
                                    &task_repeat_count)) {
            return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "AIE2P binding repeat count overflows");
          }
        } else {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "AIE2P shim DMA currently requires a dense transfer suffix "
              "beneath any repeated sequence dimensions");
        }
      }

      int64_t lane_element_stride = 0;
      if (!loom_value_facts_as_exact_i64(layout.strides[0],
                                         &lane_element_stride) ||
          lane_element_stride < 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding lane stride must be one exact non-negative value");
      }
      const int32_t element_bit_width = loom_scalar_type_bitwidth(
          loom_type_element_type(partition_source_type));
      uint64_t binding_bit_offset = 0;
      uint64_t transfer_bit_length = 0;
      if (element_bit_width <= 0 ||
          !iree_checked_mul_u64((uint64_t)lane_element_stride, partition_lane,
                                &binding_bit_offset) ||
          !iree_checked_mul_u64(binding_bit_offset, (uint64_t)element_bit_width,
                                &binding_bit_offset) ||
          !iree_checked_mul_u64(dense_inner_element_count,
                                (uint64_t)element_bit_width,
                                &transfer_bit_length) ||
          (binding_bit_offset & 7u) != 0 || (transfer_bit_length & 7u) != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding layout does not resolve to whole-byte spans");
      }
      binding_byte_offset = binding_bit_offset / 8u;
      transfer_byte_length = transfer_bit_length / 8u;
      uint64_t repeated_transfer_byte_length = 0;
      if (!iree_checked_mul_u64(transfer_byte_length, task_repeat_count,
                                &repeated_transfer_byte_length) ||
          repeated_transfer_byte_length != logical_transfer_byte_length) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding layout does not cover its logical record stream");
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P binding address layout is unknown");
    }
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
      transfer_byte_length > dma_facts->address_maximum) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P binding DMA address span is not representable");
  }

  binding_plan->binding_byte_offset = binding_byte_offset;
  binding_plan->binding_span_byte_length = transfer_byte_length;
  binding_plan->transfer_byte_length = (uint32_t)transfer_byte_length;
  binding_plan->task_repeat_count = (uint16_t)task_repeat_count;
  return iree_ok_status();
}
