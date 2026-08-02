// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/carrier.h"

#include "loom/target/capability_facts.h"

static iree_string_view_t loom_index_target_carrier_bitwidth_key(
    loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_INDEX:
      return IREE_SV("index_bitwidth");
    case LOOM_SCALAR_TYPE_OFFSET:
      return IREE_SV("offset_bitwidth");
    default:
      return iree_string_view_empty();
  }
}

int32_t loom_index_target_carrier_bitwidth(const loom_fact_context_t* context,
                                           loom_scalar_type_t scalar_type) {
  iree_string_view_t bitwidth_key =
      loom_index_target_carrier_bitwidth_key(scalar_type);
  if (iree_string_view_is_empty(bitwidth_key)) return -1;

  uint64_t target_bitwidth = 0;
  if (!loom_target_fact_context_query_u64(context, IREE_SV("target"),
                                          bitwidth_key, &target_bitwidth)) {
    return 0;
  }
  if (target_bitwidth == 0 || target_bitwidth > 64) return -1;
  return (int32_t)target_bitwidth;
}

bool loom_index_value_facts_fit_signed_target_carrier(
    const loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    loom_value_facts_t facts) {
  int32_t target_bitwidth =
      loom_index_target_carrier_bitwidth(context, scalar_type);
  return target_bitwidth == 0 ||
         (target_bitwidth > 0 && loom_value_facts_fit_signed_bit_count(
                                     facts, (uint8_t)target_bitwidth));
}

bool loom_index_value_facts_fit_unsigned_target_carrier(
    const loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    loom_value_facts_t facts) {
  int32_t target_bitwidth =
      loom_index_target_carrier_bitwidth(context, scalar_type);
  return target_bitwidth == 0 ||
         (target_bitwidth > 0 && loom_value_facts_fit_unsigned_bit_count(
                                     facts, (uint8_t)target_bitwidth));
}
