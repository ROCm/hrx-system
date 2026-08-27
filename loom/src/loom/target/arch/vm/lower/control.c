// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/control.h"

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"

static const uint8_t kVmDescriptorRecordByteLengths[] = {
#define LOOM_VM_INSTRUCTION_ENCODING_LIMITS(maximum_record_byte_length)
#define LOOM_VM_INSTRUCTION_ENCODING_ROW(byte_length) byte_length,
#include "loom/target/arch/vm/encoding_rows.inl"
#undef LOOM_VM_INSTRUCTION_ENCODING_ROW
#undef LOOM_VM_INSTRUCTION_ENCODING_LIMITS
};

typedef enum loom_vm_switch_lowering_e {
  LOOM_VM_SWITCH_LOWERING_SUBTRACT_CHAIN = 0,
  LOOM_VM_SWITCH_LOWERING_DECREMENT_CHAIN = 1,
  LOOM_VM_SWITCH_LOWERING_DIRECT = 2,
} loom_vm_switch_lowering_t;

typedef struct loom_vm_switch_plan_t {
  // Physical lowering selected from local record and table bytes.
  loom_vm_switch_lowering_t lowering;
  // Smallest authored case key used to normalize direct and decrement forms.
  int64_t minimum_key;
  // Inclusive authored key interval length, saturated when it is 2^64.
  uint64_t key_span;
} loom_vm_switch_plan_t;

static uint8_t loom_vm_descriptor_record_byte_length(
    uint16_t descriptor_ordinal) {
  IREE_ASSERT_LT(descriptor_ordinal,
                 IREE_ARRAYSIZE(kVmDescriptorRecordByteLengths));
  return kVmDescriptorRecordByteLengths[descriptor_ordinal];
}

static uint8_t loom_vm_constant_record_byte_length(uint64_t bits) {
  return loom_vm_descriptor_record_byte_length(
      loom_vm_constant_descriptor_ordinal(bits));
}

static bool loom_vm_switch_plan(const loom_op_t* source_op,
                                loom_vm_switch_plan_t* out_plan) {
  *out_plan = (loom_vm_switch_plan_t){0};
  const loom_attribute_t case_keys = loom_cfg_switch_case_keys(source_op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY || case_keys.count == 0 ||
      case_keys.i64_array == NULL) {
    return false;
  }
  for (uint16_t i = 1; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i - 1] >= case_keys.i64_array[i]) return false;
  }

  const int64_t minimum_key = case_keys.i64_array[0];
  const int64_t maximum_key = case_keys.i64_array[case_keys.count - 1];
  const uint64_t span_minus_one = (uint64_t)maximum_key - (uint64_t)minimum_key;
  const uint64_t key_span =
      span_minus_one == UINT64_MAX ? UINT64_MAX : span_minus_one + 1;
  const uint64_t branch_byte_length =
      sizeof(iree_vm_isa_control_branch_unless_s16_record_t);
  const uint64_t subtract_byte_length = loom_vm_descriptor_record_byte_length(
      VM_CORE_DESCRIPTOR_REF_INTEGER_SUB_I64);

  uint64_t subtract_chain_byte_length = 0;
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    const int64_t key = case_keys.i64_array[i];
    subtract_chain_byte_length += branch_byte_length;
    if (key != 0) {
      subtract_chain_byte_length +=
          loom_vm_constant_record_byte_length((uint64_t)key) +
          subtract_byte_length;
    }
  }

  const uint64_t normalization_byte_length =
      minimum_key == 0
          ? 0
          : loom_vm_constant_record_byte_length((uint64_t)minimum_key) +
                subtract_byte_length;
  uint64_t scalar_byte_length = subtract_chain_byte_length;
  loom_vm_switch_lowering_t scalar_lowering =
      LOOM_VM_SWITCH_LOWERING_SUBTRACT_CHAIN;
  if (key_span == case_keys.count) {
    uint64_t decrement_chain_byte_length = normalization_byte_length;
    if (case_keys.count == 1) {
      decrement_chain_byte_length += branch_byte_length;
    } else {
      decrement_chain_byte_length +=
          loom_vm_constant_record_byte_length(UINT64_C(1)) +
          branch_byte_length +
          (uint64_t)(case_keys.count - 1) *
              (subtract_byte_length + branch_byte_length);
    }
    if (decrement_chain_byte_length <= scalar_byte_length) {
      scalar_byte_length = decrement_chain_byte_length;
      scalar_lowering = LOOM_VM_SWITCH_LOWERING_DECREMENT_CHAIN;
    }
  }

  loom_vm_switch_lowering_t lowering = scalar_lowering;
  if (case_keys.count >= 2 && key_span <= UINT16_MAX - 1u) {
    const uint64_t direct_byte_length =
        normalization_byte_length +
        sizeof(iree_vm_isa_control_switch_record_t) +
        key_span * sizeof(iree_vm_bytecode_v0_switch_target_entry_t);
    if (direct_byte_length <= scalar_byte_length) {
      lowering = LOOM_VM_SWITCH_LOWERING_DIRECT;
    }
  }
  *out_plan = (loom_vm_switch_plan_t){
      .lowering = lowering,
      .minimum_key = minimum_key,
      .key_span = key_span,
  };
  return true;
}

static iree_status_t loom_vm_emit_subtract(loom_low_lower_context_t* context,
                                           loom_location_id_t location,
                                           loom_value_id_t lhs,
                                           loom_value_id_t rhs,
                                           loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          descriptor_set, VM_CORE_DESCRIPTOR_REF_INTEGER_SUB_I64);
  IREE_ASSERT(descriptor != NULL);
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  loom_op_t* subtract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      loom_low_lower_context_builder(context), descriptor_set, descriptor,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      &result_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &subtract_op));
  *out_result = loom_value_slice_get(loom_low_op_results(subtract_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_vm_normalize_switch_selector(
    loom_low_lower_context_t* context, loom_location_id_t location,
    loom_value_id_t selector, int64_t minimum_key,
    loom_value_id_t* out_selector) {
  *out_selector = selector;
  if (minimum_key == 0) return iree_ok_status();
  const loom_type_t selector_type =
      loom_module_value_type(loom_low_lower_context_module(context), selector);
  loom_value_id_t minimum = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_constant_build(
      loom_low_lower_context_builder(context), (uint64_t)minimum_key,
      selector_type, location, &minimum));
  return loom_vm_emit_subtract(context, location, selector, minimum,
                               out_selector);
}

static iree_status_t loom_vm_insert_switch_dispatch_blocks(
    loom_low_lower_context_t* context, uint16_t dispatch_block_count,
    loom_block_t** out_blocks) {
  if (dispatch_block_count == 0) return iree_ok_status();
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_block_t* source_block = builder->ip.block;
  loom_region_t* body = source_block->parent_region;
  if (dispatch_block_count > UINT16_MAX - body->block_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "VM scalar switch expansion exceeds the function block limit");
  }
  const uint16_t first_block_index = (uint16_t)(source_block->region_index + 1);
  for (uint16_t i = 0; i < dispatch_block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_region_insert_block(
        loom_low_lower_context_module(context), body,
        (uint16_t)(first_block_index + i), &out_blocks[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_switch_subtract_chain(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t selector, loom_block_t* default_dest,
    loom_block_t* const* case_dests, uint16_t case_count) {
  loom_block_t** dispatch_blocks = NULL;
  if (case_count > 1) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, case_count - 1, sizeof(*dispatch_blocks),
        (void**)&dispatch_blocks));
    IREE_RETURN_IF_ERROR(loom_vm_insert_switch_dispatch_blocks(
        context, case_count - 1, dispatch_blocks));
  }

  const loom_attribute_t case_keys = loom_cfg_switch_case_keys(source_op);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_type_t selector_type =
      loom_module_value_type(loom_low_lower_context_module(context), selector);
  for (uint16_t i = 0; i < case_count; ++i) {
    if (i != 0) loom_builder_set_block(builder, dispatch_blocks[i - 1]);
    loom_value_id_t difference = selector;
    if (case_keys.i64_array[i] != 0) {
      loom_value_id_t key = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_vm_constant_build(builder, (uint64_t)case_keys.i64_array[i],
                                 selector_type, source_op->location, &key));
      IREE_RETURN_IF_ERROR(loom_vm_emit_subtract(context, source_op->location,
                                                 selector, key, &difference));
    }
    loom_block_t* nonzero_dest =
        i + 1 < case_count ? dispatch_blocks[i] : default_dest;
    loom_op_t* branch_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_cond_br_build(builder, difference, nonzero_dest, case_dests[i],
                               source_op->location, &branch_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_switch_decrement_chain(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t selector, int64_t minimum_key, loom_block_t* default_dest,
    loom_block_t* const* case_dests, uint16_t case_count) {
  loom_block_t** dispatch_blocks = NULL;
  if (case_count > 1) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, case_count - 1, sizeof(*dispatch_blocks),
        (void**)&dispatch_blocks));
    IREE_RETURN_IF_ERROR(loom_vm_insert_switch_dispatch_blocks(
        context, case_count - 1, dispatch_blocks));
  }

  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_normalize_switch_selector(
      context, source_op->location, selector, minimum_key, &value));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  if (case_count > 1) {
    IREE_RETURN_IF_ERROR(loom_vm_constant_build(
        builder, UINT64_C(1),
        loom_module_value_type(loom_low_lower_context_module(context), value),
        source_op->location, &one));
  }

  for (uint16_t i = 0; i < case_count; ++i) {
    if (i != 0) {
      loom_builder_set_block(builder, dispatch_blocks[i - 1]);
      IREE_RETURN_IF_ERROR(loom_vm_emit_subtract(context, source_op->location,
                                                 value, one, &value));
    }
    loom_block_t* nonzero_dest =
        i + 1 < case_count ? dispatch_blocks[i] : default_dest;
    loom_op_t* branch_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_cond_br_build(builder, value, nonzero_dest, case_dests[i],
                               source_op->location, &branch_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_switch_direct(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t selector, const loom_vm_switch_plan_t* plan,
    loom_block_t* default_dest, loom_block_t* const* case_dests,
    uint16_t case_count) {
  IREE_ASSERT_LE(plan->key_span, UINT16_MAX - 1u);
  const uint16_t target_count = (uint16_t)plan->key_span;
  loom_block_t** targets = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, target_count, sizeof(*targets), (void**)&targets));
  for (uint16_t i = 0; i < target_count; ++i) targets[i] = default_dest;

  const loom_attribute_t case_keys = loom_cfg_switch_case_keys(source_op);
  for (uint16_t i = 0; i < case_count; ++i) {
    const uint64_t target_index =
        (uint64_t)case_keys.i64_array[i] - (uint64_t)plan->minimum_key;
    IREE_ASSERT_LT(target_index, target_count);
    targets[target_index] = case_dests[i];
  }

  loom_value_id_t normalized_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_normalize_switch_selector(
      context, source_op->location, selector, plan->minimum_key,
      &normalized_selector));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          descriptor_set, VM_CORE_DESCRIPTOR_REF_CONTROL_SWITCH);
  IREE_ASSERT(descriptor != NULL);
  loom_op_t* switch_op = NULL;
  return loom_low_build_resolved_descriptor_switch(
      loom_low_lower_context_builder(context), descriptor_set, descriptor,
      normalized_selector, default_dest, targets, target_count,
      source_op->location, &switch_op);
}

bool loom_vm_switch_lowering_can_emit(void* user_data,
                                      const loom_module_t* module,
                                      const loom_op_t* source_op,
                                      const loom_target_facts_t* target_facts) {
  (void)user_data;
  (void)module;
  (void)target_facts;
  loom_vm_switch_plan_t plan = {0};
  return loom_vm_switch_plan(source_op, &plan);
}

iree_status_t loom_vm_switch_lowering_emit(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t selector,
                                           loom_block_t* default_dest,
                                           loom_block_t* const* case_dests,
                                           uint16_t case_count) {
  (void)user_data;
  const loom_attribute_t case_keys = loom_cfg_switch_case_keys(source_op);
  loom_vm_switch_plan_t plan = {0};
  if (case_keys.count != case_count || !loom_vm_switch_plan(source_op, &plan)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM switch lowering plan is stale");
  }
  switch (plan.lowering) {
    case LOOM_VM_SWITCH_LOWERING_SUBTRACT_CHAIN:
      return loom_vm_emit_switch_subtract_chain(
          context, source_op, selector, default_dest, case_dests, case_count);
    case LOOM_VM_SWITCH_LOWERING_DECREMENT_CHAIN:
      return loom_vm_emit_switch_decrement_chain(context, source_op, selector,
                                                 plan.minimum_key, default_dest,
                                                 case_dests, case_count);
    case LOOM_VM_SWITCH_LOWERING_DIRECT:
      return loom_vm_emit_switch_direct(context, source_op, selector, &plan,
                                        default_dest, case_dests, case_count);
    default:
      IREE_ASSERT_UNREACHABLE("VM switch lowering kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
