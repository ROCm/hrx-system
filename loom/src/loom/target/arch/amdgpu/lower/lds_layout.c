// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source workgroup-storage layout.
//
// Source-to-low lowering must know the final workgroup-storage base for packets
// that encode a workgroup-storage destination directly, such as
// global_load_lds. This layout mirrors the low.storage.reserve order emitted by
// lower_buffer.c: source block order, then source op order, with each
// workgroup buffer.alloca packed into the workgroup segment after applying its
// declared alignment.

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_source_lds_layout_checked_align(
    uint64_t value, uint64_t alignment, uint64_t* out_aligned_value) {
  *out_aligned_value = 0;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return false;
  }
  const uint64_t alignment_mask = alignment - 1;
  if (value > UINT64_MAX - alignment_mask) {
    return false;
  }
  *out_aligned_value = (value + alignment_mask) & ~alignment_mask;
  return true;
}

static bool loom_amdgpu_source_lds_layout_alloca_size(
    const loom_value_fact_table_t* fact_table, const loom_op_t* alloca_op,
    uint64_t* out_byte_length) {
  *out_byte_length = 0;
  int64_t byte_length = 0;
  if (!loom_amdgpu_value_facts_as_exact_non_negative_i64(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(alloca_op)),
          &byte_length) ||
      byte_length <= 0) {
    return false;
  }
  *out_byte_length = (uint64_t)byte_length;
  return true;
}

static bool loom_amdgpu_source_lds_layout_alloca_alignment(
    const loom_op_t* alloca_op, uint64_t* out_byte_alignment) {
  *out_byte_alignment = 0;
  const int64_t base_alignment = loom_buffer_alloca_base_alignment(alloca_op);
  if (base_alignment <= 0 || base_alignment > UINT32_MAX ||
      !loom_amdgpu_u32_is_power_of_two((uint32_t)base_alignment)) {
    return false;
  }
  *out_byte_alignment = (uint64_t)base_alignment;
  return true;
}

bool loom_amdgpu_source_lds_layout_lookup_root(
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_value_id_t root_value_id, uint64_t* out_byte_offset) {
  *out_byte_offset = 0;
  if (!loom_func_like_isa(source_function)) {
    return false;
  }
  loom_region_t* body = loom_func_like_body(source_function);
  if (body == NULL) {
    return false;
  }

  uint64_t lds_byte_size = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_buffer_alloca_isa(op) ||
          loom_buffer_alloca_memory_space(op) !=
              LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
        continue;
      }

      uint64_t byte_alignment = 0;
      if (!loom_amdgpu_source_lds_layout_alloca_alignment(op,
                                                          &byte_alignment)) {
        return false;
      }
      uint64_t slot_byte_offset = 0;
      if (!loom_amdgpu_source_lds_layout_checked_align(
              lds_byte_size, byte_alignment, &slot_byte_offset)) {
        return false;
      }
      uint64_t byte_length = 0;
      if (!loom_amdgpu_source_lds_layout_alloca_size(fact_table, op,
                                                     &byte_length)) {
        return false;
      }
      if (loom_buffer_alloca_result(op) == root_value_id) {
        *out_byte_offset = slot_byte_offset;
        return true;
      }

      if (slot_byte_offset > UINT64_MAX - byte_length) {
        return false;
      }
      lds_byte_size = slot_byte_offset + byte_length;
    }
  }
  return false;
}
