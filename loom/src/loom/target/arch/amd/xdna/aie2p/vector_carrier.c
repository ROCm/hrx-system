// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"

#include "loom/ir/scalar_type.h"

uint16_t loom_aie2p_scalar_type_physical_bit_count(
    loom_scalar_type_t element_type) {
  switch (element_type) {
    case LOOM_SCALAR_TYPE_INDEX:
      return LOOM_AIE2P_INDEX_CARRIER_BIT_COUNT;
    case LOOM_SCALAR_TYPE_OFFSET:
      return LOOM_AIE2P_OFFSET_CARRIER_BIT_COUNT;
    default: {
      const int32_t bit_count = loom_scalar_type_bitwidth(element_type);
      return bit_count > 0 ? (uint16_t)bit_count : 0;
    }
  }
}

loom_aie2p_vector_carrier_t loom_aie2p_vector_carrier_for_type(
    loom_type_t type) {
  uint64_t element_count = 0;
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      !loom_type_static_element_count(type, &element_count) ||
      element_count == 0) {
    return (loom_aie2p_vector_carrier_t){0};
  }

  const loom_scalar_type_t element_type = loom_type_element_type(type);
  const bool is_rank_one = loom_type_rank(type) == 1;
  const uint16_t element_bit_count =
      loom_aie2p_scalar_type_physical_bit_count(element_type);
  if (is_rank_one && element_count == 32 &&
      element_type == LOOM_SCALAR_TYPE_F32) {
    return (loom_aie2p_vector_carrier_t){
        .kind = LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR,
        .unit_count = 2,
    };
  }

  // Accumulator values above 1024 bits retain the containing four-unit
  // physical view independently of logical rank. AIE2P has no allocatable
  // three-unit accumulator view; units beyond the logical extent are
  // unobservable.
  const bool has_accumulator_element_type =
      element_type == LOOM_SCALAR_TYPE_I32 ||
      element_type == LOOM_SCALAR_TYPE_I64 ||
      element_type == LOOM_SCALAR_TYPE_F32 ||
      element_type == LOOM_SCALAR_TYPE_INDEX ||
      element_type == LOOM_SCALAR_TYPE_OFFSET;
  if (has_accumulator_element_type && element_bit_count > 0 &&
      element_count > 1024 / (uint32_t)element_bit_count &&
      element_count <= 2048 / (uint32_t)element_bit_count) {
    return (loom_aie2p_vector_carrier_t){
        .kind = LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR,
        .unit_count = 4,
    };
  }

  if (element_type == LOOM_SCALAR_TYPE_I1) {
    if (element_count <= 128) {
      return (loom_aie2p_vector_carrier_t){
          .kind = LOOM_AIE2P_VECTOR_CARRIER_PREDICATE,
          .unit_count = (uint32_t)((element_count + 63u) / 64u),
      };
    }
    // Predicate payloads have no ordinary vector-register representation.
    return (loom_aie2p_vector_carrier_t){0};
  }

  if (element_bit_count > 0 &&
      element_count > 512 / (uint32_t)element_bit_count &&
      element_count <= 1024 / (uint32_t)element_bit_count) {
    return (loom_aie2p_vector_carrier_t){
        .kind = LOOM_AIE2P_VECTOR_CARRIER_ORDINARY,
        .unit_count = 4,
    };
  }
  if (element_bit_count > 0 &&
      element_count <= 512 / (uint32_t)element_bit_count) {
    return (loom_aie2p_vector_carrier_t){
        .kind = LOOM_AIE2P_VECTOR_CARRIER_ORDINARY,
        .unit_count = 2,
    };
  }
  return (loom_aie2p_vector_carrier_t){0};
}
