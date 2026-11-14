// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/index.h"

#include "loom/codegen/low/lower/representation_observer.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/registers.h"

static bool loom_spirv_index_needs_wide(loom_value_facts_t facts) {
  return facts.range_lo >= INT32_MIN && facts.range_hi > INT32_MAX;
}

iree_status_t loom_spirv_map_index_value(loom_low_lower_context_t* context,
                                         loom_value_id_t value,
                                         loom_type_t* out_type) {
  loom_low_representation_id_t representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_lookup_if_ready(context, value,
                                                &representation);
  const bool wide = representation == LOOM_SPIRV_INDEX_REPRESENTATION_WIDE ||
                    loom_spirv_index_needs_wide(loom_value_fact_table_lookup(
                        loom_low_lower_context_fact_table(context), value));
  return loom_low_lower_make_typed_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1,
      loom_type_scalar(wide ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_INDEX),
      out_type);
}

static uint32_t loom_spirv_index_carrier_bitwidth(loom_type_t type) {
  if (loom_low_register_type_class_id(type) !=
          SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID ||
      loom_low_register_type_unit_count(type) != 1) {
    return 0;
  }
  const loom_type_t* payload = loom_type_register_value_type(type);
  if (payload == NULL || !loom_type_is_scalar(*payload)) {
    return 0;
  }
  switch (loom_type_element_type(*payload)) {
    case LOOM_SCALAR_TYPE_INDEX:
      return 32;
    case LOOM_SCALAR_TYPE_I64:
      return 64;
    default:
      return 0;
  }
}

loom_type_t loom_spirv_join_index_type(loom_type_t source_type, loom_type_t lhs,
                                       loom_type_t rhs) {
  if (!loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return loom_type_none();
  }
  const uint32_t left_width = loom_spirv_index_carrier_bitwidth(lhs);
  const uint32_t right_width = loom_spirv_index_carrier_bitwidth(rhs);
  if (left_width == 0 || right_width == 0) {
    return loom_type_none();
  }
  return right_width > left_width ? rhs : lhs;
}

bool loom_spirv_index_control_operand_accepts(loom_type_t source_type,
                                              loom_value_facts_t source_facts,
                                              loom_type_t actual_type,
                                              loom_type_t required_type) {
  const loom_type_t common_type =
      loom_spirv_join_index_type(source_type, actual_type, required_type);
  return loom_type_kind(common_type) != LOOM_TYPE_NONE &&
         (loom_type_equal(common_type, required_type) ||
          loom_value_facts_fit_signed_bit_count(source_facts, 32));
}
