// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/source_integer_representation.h"

#include "iree/base/internal/math.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"

enum loom_amdgpu_narrow_integer_guarantee_bits_e {
  // The declared narrow payload is valid in the low carrier bits.
  LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS = 1u << 0,
  // Carrier bits above the declared width replicate its sign bit.
  LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_SIGN_EXTENDED = 1u << 1,
  // Carrier bits above the declared width are zero.
  LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_ZERO_EXTENDED = 1u << 2,
};
typedef uint8_t loom_amdgpu_narrow_integer_guarantees_t;

static bool loom_amdgpu_source_integer_representation_is_narrow(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_source_integer_representation_type_is_narrow(
      loom_module_value_type(module, value_id));
}

static void loom_amdgpu_source_integer_representation_record_flexible(
    loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_low_representation_candidate_t candidates[] = {
      {
          .representation = LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS,
      },
      {
          .representation =
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED,
          .cost = {.runtime = 1, .code_size = 1},
      },
      {
          .representation =
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
          .cost = {.runtime = 1, .code_size = 1},
      },
  };
  loom_low_lower_representation_record_candidates(
      recorder, value_id, candidates, IREE_ARRAYSIZE(candidates));
}

static void loom_amdgpu_source_integer_representation_record_guarantees(
    loom_value_id_t value_id,
    loom_amdgpu_narrow_integer_guarantees_t guarantees,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_low_representation_candidate_t candidates[3] = {0};
  iree_host_size_t candidate_count = 0;
  if (iree_any_bit_set(guarantees,
                       LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS)) {
    candidates[candidate_count++].representation =
        LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS;
  }
  if (iree_any_bit_set(guarantees,
                       LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_SIGN_EXTENDED)) {
    candidates[candidate_count++].representation =
        LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED;
  }
  if (iree_any_bit_set(guarantees,
                       LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_ZERO_EXTENDED)) {
    candidates[candidate_count++].representation =
        LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED;
  }
  IREE_ASSERT_GT(candidate_count, 0u);
  loom_low_lower_representation_record_candidates(recorder, value_id,
                                                  candidates, candidate_count);
}

static void loom_amdgpu_source_integer_representation_record_consumer(
    loom_value_id_t value_id,
    loom_low_representation_id_t required_representation,
    loom_low_lower_representation_recorder_t* recorder) {
  IREE_ASSERT(required_representation ==
                  LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED ||
              required_representation ==
                  LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED);
  const uint32_t sign_extended_cost =
      required_representation ==
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED
          ? 0u
          : 1u;
  const uint32_t zero_extended_cost =
      required_representation ==
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED
          ? 0u
          : 1u;
  const loom_low_representation_candidate_t candidates[] = {
      {
          .representation = LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS,
          .cost = {.runtime = 1, .code_size = 1},
      },
      {
          .representation =
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED,
          .cost = {.runtime = sign_extended_cost,
                   .code_size = sign_extended_cost},
      },
      {
          .representation =
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
          .cost = {.runtime = zero_extended_cost,
                   .code_size = zero_extended_cost},
      },
  };
  loom_low_lower_representation_record_costs(recorder, value_id, candidates,
                                             IREE_ARRAYSIZE(candidates));
}

static void loom_amdgpu_source_integer_representation_record_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_representation_id_t required_representation,
    loom_amdgpu_narrow_integer_guarantees_t result_guarantees,
    loom_low_lower_representation_recorder_t* recorder) {
  IREE_ASSERT_EQ(source_op->operand_count, 1u);
  IREE_ASSERT_EQ(source_op->result_count, 1u);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t input = loom_op_const_operands(source_op)[0];
  if (loom_amdgpu_source_integer_representation_is_narrow(module, input)) {
    loom_amdgpu_source_integer_representation_record_consumer(
        input, required_representation, recorder);
  }
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  if (loom_amdgpu_source_integer_representation_is_narrow(module, result)) {
    loom_amdgpu_source_integer_representation_record_guarantees(
        result, result_guarantees, recorder);
  }
}

void loom_amdgpu_source_integer_representation_observe_boundary(
    loom_amdgpu_source_integer_representation_action_t action,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  switch (action) {
    case LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_TRANSPORT_PAYLOAD: {
      const loom_value_id_t input = loom_op_const_operands(source_op)[0];
      const loom_value_id_t result = loom_op_const_results(source_op)[0];
      if (loom_amdgpu_source_integer_representation_is_narrow(module, input)) {
        loom_low_lower_representation_record_union(recorder, input, result);
      }
      return;
    }
    case LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT:
    case LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT: {
      IREE_ASSERT_EQ(source_op->result_count, 1u);
      if (source_op->kind == LOOM_OP_INDEX_CAST) {
        IREE_ASSERT_EQ(source_op->operand_count, 1u);
        const loom_value_id_t input = loom_op_const_operands(source_op)[0];
        const loom_scalar_type_t result_scalar_type =
            loom_type_element_type(loom_module_value_type(
                module, loom_op_const_results(source_op)[0]));
        if (loom_amdgpu_source_integer_representation_is_narrow(module,
                                                                input) &&
            (result_scalar_type == LOOM_SCALAR_TYPE_INDEX ||
             result_scalar_type == LOOM_SCALAR_TYPE_OFFSET)) {
          loom_amdgpu_source_integer_representation_record_consumer(
              input,
              result_scalar_type == LOOM_SCALAR_TYPE_INDEX
                  ? LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED
                  : LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
              recorder);
        }
      }
      const loom_value_id_t result = loom_op_const_results(source_op)[0];
      if (!loom_amdgpu_source_integer_representation_is_narrow(module,
                                                               result)) {
        return;
      }
      if (action ==
          LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT) {
        loom_amdgpu_source_integer_representation_record_flexible(result,
                                                                  recorder);
      } else {
        loom_amdgpu_source_integer_representation_record_guarantees(
            result,
            LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS |
                LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_SIGN_EXTENDED,
            recorder);
      }
      return;
    }
    case LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_CONSTANT_RESULT: {
      IREE_ASSERT_EQ(source_op->result_count, 1u);
      const loom_value_id_t result = loom_op_const_results(source_op)[0];
      if (!loom_amdgpu_source_integer_representation_is_narrow(module,
                                                               result)) {
        return;
      }
      const loom_attribute_t value = loom_scalar_constant_value(source_op);
      IREE_ASSERT_EQ(value.kind, LOOM_ATTR_I64);
      loom_amdgpu_narrow_integer_guarantees_t guarantees =
          LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS |
          LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_SIGN_EXTENDED;
      if (value.i64 >= 0) {
        guarantees |= LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_ZERO_EXTENDED;
      }
      loom_amdgpu_source_integer_representation_record_guarantees(
          result, guarantees, recorder);
      return;
    }
    case LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGNED_CONVERSION:
      loom_amdgpu_source_integer_representation_record_conversion(
          context, source_op,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED,
          LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS |
              LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_SIGN_EXTENDED,
          recorder);
      return;
    case LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_UNSIGNED_CONVERSION:
      loom_amdgpu_source_integer_representation_record_conversion(
          context, source_op,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
          LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS |
              LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_ZERO_EXTENDED,
          recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU integer representation action");
}

static void loom_amdgpu_source_integer_representation_record_callable_inputs(
    loom_low_lower_context_t* context, const loom_value_id_t* value_ids,
    iree_host_size_t value_count,
    loom_amdgpu_narrow_integer_guarantees_t guarantees,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    if (loom_amdgpu_source_integer_representation_is_narrow(module,
                                                            value_ids[i])) {
      loom_amdgpu_source_integer_representation_record_guarantees(
          value_ids[i], guarantees, recorder);
    }
  }
}

void loom_amdgpu_source_integer_representation_observe_callable_boundary(
    loom_low_lower_representation_callable_boundary_kind_t kind,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  switch (kind) {
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_DEFINITION: {
      const loom_func_like_t function =
          loom_func_like_const_cast(module, source_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* argument_ids =
          loom_func_like_arg_ids(function, &argument_count);
      // Kernel metadata advertises payload bytes, not a zero-extended carrier.
      // Both kernel and ordinary callable inputs guarantee only their low bits.
      loom_amdgpu_source_integer_representation_record_callable_inputs(
          context, argument_ids, argument_count,
          LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_CALL: {
      const loom_call_like_t call =
          loom_call_like_const_cast(module, source_op);
      const loom_value_slice_t results = loom_call_like_results(call);
      loom_amdgpu_source_integer_representation_record_callable_inputs(
          context, results.values, results.count,
          LOOM_AMDGPU_NARROW_INTEGER_GUARANTEE_LOW_BITS, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_EXIT:
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown callable representation boundary kind");
}

loom_low_representation_id_t loom_amdgpu_source_integer_representation_lookup(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  loom_low_representation_id_t representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_lookup(context, source_value_id,
                                       &representation);
  IREE_ASSERT(
      representation == LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS ||
          representation ==
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED ||
          representation ==
              LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
      "narrow integer producer must select a carrier representation");
  return representation;
}

static iree_status_t loom_amdgpu_normalize_narrow_integer_to_lane_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_bit_count,
    loom_low_representation_id_t source_representation,
    loom_low_representation_id_t required_representation, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = low_source;
  if (source_representation == required_representation) {
    return iree_ok_status();
  }

  switch (required_representation) {
    case LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS:
      return iree_ok_status();
    case LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED:
      return loom_amdgpu_extract_register_bitfield(
          context, source_op, low_source, /*bit_offset=*/0, source_bit_count,
          LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND, lane_type,
          out_low_result);
    case LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED:
      if (loom_amdgpu_low_type_is_register_class(
              context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR) ||
          loom_amdgpu_low_value_defines_register_low16(context, low_source)) {
        return loom_amdgpu_extract_register_bitfield(
            context, source_op, low_source, /*bit_offset=*/0, source_bit_count,
            LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_ZERO_EXTEND, lane_type,
            out_low_result);
      }
      return loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          low_source,
          iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)source_bit_count),
          lane_type, out_low_result);
    case LOOM_LOW_REPRESENTATION_ID_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("invalid narrow integer representation");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_normalize_narrow_integer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_bit_count,
    loom_low_representation_id_t source_representation,
    loom_low_representation_id_t required_representation,
    loom_value_id_t* out_low_result) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  return loom_amdgpu_normalize_narrow_integer_to_lane_type(
      context, source_op, low_source, source_bit_count, source_representation,
      required_representation, lane_type, out_low_result);
}

iree_status_t loom_amdgpu_normalize_narrow_source_integer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t low_source,
    loom_low_representation_id_t required_representation, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_value);
  if (!loom_amdgpu_source_integer_representation_type_is_narrow(source_type)) {
    *out_low_result = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_normalize_narrow_integer_to_lane_type(
      context, source_op, low_source,
      loom_scalar_type_bitwidth(loom_type_element_type(source_type)),
      loom_amdgpu_source_integer_representation_lookup(context, source_value),
      required_representation, lane_type, out_low_result);
}
