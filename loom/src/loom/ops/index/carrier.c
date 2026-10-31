// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/carrier.h"

#include "loom/ir/module.h"
#include "loom/target/facts.h"
#include "loom/util/fact_table.h"

int32_t loom_index_target_carrier_bitwidth(const loom_fact_context_t* context,
                                           loom_scalar_type_t scalar_type) {
  if (scalar_type != LOOM_SCALAR_TYPE_INDEX &&
      scalar_type != LOOM_SCALAR_TYPE_OFFSET) {
    return -1;
  }
  if (!context || !context->target_facts) {
    return 0;
  }
  const loom_target_snapshot_t* snapshot =
      &context->target_facts->storage.snapshot;
  const uint32_t target_bitwidth = scalar_type == LOOM_SCALAR_TYPE_INDEX
                                       ? snapshot->index_bitwidth
                                       : snapshot->offset_bitwidth;
  if (target_bitwidth == 0 || target_bitwidth > 64) {
    return -1;
  }
  return (int32_t)target_bitwidth;
}

loom_value_facts_t loom_index_integer_carrier_domain_facts(
    const loom_fact_context_t* context, loom_scalar_type_t scalar_type) {
  const int32_t carrier_bitwidth =
      loom_index_target_carrier_bitwidth(context, scalar_type);
  if (carrier_bitwidth > 0 && scalar_type == LOOM_SCALAR_TYPE_INDEX) {
    return loom_value_facts_make_signed_bit_count_range(carrier_bitwidth);
  }
  if (carrier_bitwidth > 0 && carrier_bitwidth < 64 &&
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    return loom_value_facts_make_unsigned_bit_count_range(carrier_bitwidth);
  }
  int64_t range_lo = 0;
  int64_t range_hi = 0;
  if (loom_value_facts_scalar_type_domain(scalar_type, &range_lo, &range_hi)) {
    return loom_value_facts_make(range_lo, range_hi, 1);
  }
  return loom_value_facts_unknown();
}

loom_value_facts_t loom_index_wrap_offset_facts(
    const loom_fact_context_t* context, loom_value_facts_t mathematical_facts) {
  int32_t bit_count =
      loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_OFFSET);
  if (bit_count == 0) {
    // Targetless facts use the widest representable source carrier. This
    // preserves ordinary mathematical ranges while refusing to turn a
    // possible unsigned wrap into a clamped exact value.
    bit_count = 64;
  }
  if (bit_count < 0) {
    loom_value_facts_t result =
        loom_value_facts_make(0, INT64_MAX, /*known_divisor=*/1);
    loom_value_facts_propagate_unary_distribution(mathematical_facts, &result);
    return result;
  }
  return loom_value_facts_wrap_unsigned_integer(mathematical_facts, bit_count);
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

static bool loom_index_value_has_producer(const loom_module_t* module,
                                          loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? loom_value_def_block(value) != NULL
                                        : loom_value_def_op(value) != NULL;
}

int32_t loom_index_value_fact_carrier_bitwidth(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t value_id, loom_scalar_type_t scalar_type,
    loom_value_facts_t facts) {
  const int32_t native_bitwidth =
      loom_index_target_carrier_bitwidth(context, scalar_type);
  if (native_bitwidth <= 0 || native_bitwidth >= 64 ||
      scalar_type != LOOM_SCALAR_TYPE_INDEX ||
      loom_value_facts_is_float(facts) ||
      !loom_index_value_has_producer(module, value_id)) {
    return native_bitwidth;
  }

  // A native index carrier preserves its signed domain and nonnegative raw bit
  // patterns. Values proven wholly below the native signed domain are not
  // supported by the target's wide-coordinate carrier. A conservative
  // envelope crossing that boundary remains native: ordinary arithmetic over
  // unconstrained native inputs produces such an envelope without establishing
  // a wider representation. An established envelope extending above the raw
  // native payload requires the target's wider carrier.
  const int64_t native_signed_minimum = -(INT64_C(1) << (native_bitwidth - 1));
  const int64_t native_unsigned_maximum =
      (int64_t)((UINT64_C(1) << native_bitwidth) - 1);
  if (facts.range_hi < native_signed_minimum) {
    return -1;
  }
  if (facts.range_lo < native_signed_minimum ||
      facts.range_hi <= native_unsigned_maximum) {
    return native_bitwidth;
  }
  const uint32_t maximum_bitwidth = loom_target_snapshot_maximum_index_bitwidth(
      &context->target_facts->storage.snapshot);
  if (maximum_bitwidth <= (uint32_t)native_bitwidth || maximum_bitwidth > 64) {
    return -1;
  }
  if (maximum_bitwidth == 64) {
    return 64;
  }
  const int64_t maximum_signed_minimum =
      -(INT64_C(1) << (maximum_bitwidth - 1));
  const int64_t maximum_unsigned_maximum =
      (int64_t)((UINT64_C(1) << maximum_bitwidth) - 1);
  return facts.range_lo >= maximum_signed_minimum &&
                 facts.range_hi <= maximum_unsigned_maximum
             ? (int32_t)maximum_bitwidth
             : -1;
}
