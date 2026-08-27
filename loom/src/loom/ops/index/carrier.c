// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/carrier.h"

#include "loom/target/facts.h"
#include "loom/util/fact_table.h"

int32_t loom_index_target_carrier_bitwidth(const loom_fact_context_t* context,
                                           loom_scalar_type_t scalar_type) {
  if (scalar_type != LOOM_SCALAR_TYPE_INDEX &&
      scalar_type != LOOM_SCALAR_TYPE_OFFSET) {
    return -1;
  }
  if (!context || !context->target_facts) return 0;
  const loom_target_snapshot_t* snapshot =
      &context->target_facts->storage.snapshot;
  const uint32_t target_bitwidth = scalar_type == LOOM_SCALAR_TYPE_INDEX
                                       ? snapshot->index_bitwidth
                                       : snapshot->offset_bitwidth;
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
