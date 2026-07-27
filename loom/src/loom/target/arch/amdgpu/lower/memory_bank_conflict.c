// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_bank_conflict.h"

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"

loom_amdgpu_memory_bank_geometry_t loom_amdgpu_memory_bank_default_lds_geometry(
    void) {
  return (loom_amdgpu_memory_bank_geometry_t){
      .bank_count = 32,
      .bank_width_bytes = 4,
  };
}

iree_string_view_t loom_amdgpu_memory_bank_conflict_kind_key(
    loom_amdgpu_memory_bank_conflict_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_FREE:
      return IREE_SV("bank-conflict-free");
    case LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_PADDED_FREE:
      return IREE_SV("padded-bank-conflict-free");
    case LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_RISK:
      return IREE_SV("bank-conflict-risk");
    case LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_SUBWORD_RISK:
      return IREE_SV("subword-bank-conflict-risk");
    case LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_UNKNOWN:
    default:
      return IREE_SV("bank-pattern-unknown");
  }
}

static uint32_t loom_amdgpu_gcd_u32(uint32_t lhs, uint32_t rhs) {
  while (rhs != 0) {
    const uint32_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

static uint32_t loom_amdgpu_memory_access_packet_footprint_bytes(
    const loom_amdgpu_memory_access_t* access) {
  if (access->packet_byte_count != 0) {
    return access->packet_byte_count;
  }
  const uint32_t payload_register_count =
      iree_max(access->payload_register_count, 1u);
  return access->source.element_byte_count * payload_register_count;
}

static uint32_t loom_amdgpu_memory_subword_conflict_degree(
    uint32_t byte_stride, uint32_t bank_width_bytes) {
  return bank_width_bytes / loom_amdgpu_gcd_u32(byte_stride, bank_width_bytes);
}

loom_amdgpu_memory_bank_conflict_summary_t
loom_amdgpu_memory_access_bank_conflict_summary(
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_bank_geometry_t geometry) {
  loom_amdgpu_memory_bank_conflict_summary_t summary = {
      .kind = LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_UNKNOWN,
  };
  if (!access || geometry.bank_count == 0 || geometry.bank_width_bytes == 0 ||
      access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return summary;
  }

  const loom_low_source_memory_dynamic_term_t* term =
      loom_low_source_memory_access_single_dynamic_term(&access->source);
  if (!term ||
      access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      term->byte_stride <= 0) {
    return summary;
  }
  if (term->byte_stride > UINT32_MAX) {
    return summary;
  }

  const uint32_t byte_stride = (uint32_t)term->byte_stride;
  if ((byte_stride % geometry.bank_width_bytes) != 0) {
    summary.conflict_degree = loom_amdgpu_memory_subword_conflict_degree(
        byte_stride, geometry.bank_width_bytes);
    summary.kind = LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_SUBWORD_RISK;
    return summary;
  }

  summary.bank_stride_words = byte_stride / geometry.bank_width_bytes;
  summary.conflict_degree =
      loom_amdgpu_gcd_u32(summary.bank_stride_words, geometry.bank_count);
  if (summary.conflict_degree == 1) {
    const uint32_t packet_footprint_bytes =
        loom_amdgpu_memory_access_packet_footprint_bytes(access);
    summary.kind = (uint64_t)term->byte_stride > packet_footprint_bytes
                       ? LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_PADDED_FREE
                       : LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_FREE;
  } else {
    summary.kind = LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_RISK;
  }
  return summary;
}
